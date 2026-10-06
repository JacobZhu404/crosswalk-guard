#!/usr/bin/env python3
"""违章01 FP 生产根因只读诊断 (plan-gate #3 前置, wb)。

方法: 复用生产入口 redlight.app.cli.run(零循环复制), 仅用**只读** monkeypatch 抓取引擎内部
逐帧状态:
  - TrafficLightDetector.observe: 每灯帧复算采样路径 + prior ROI 绿/红像素 + 全局亮斑绿/红面积,
    存入 OBS_LATEST(前向填充, 与生产的 stale obs 语义一致)。
  - BatchViolationEngine.accumulate: 每推理帧记录 (ts, obs, conf, last_sample, signal_prior,
    OBS_LATEST), 与生产 engine._light_obs 严格 1:1 对齐。
  - BatchViolationEngine.decide: 抓取引擎实例, 取 _light_obs/_occ_samples/_track_samples。
禁用 select_gtfree。GT 仅裁判(不进推理)。

唯一二分: 01 FP 走 :87(on_crosswalk+green/flashing→confirmed 误绿) 还是 :90(unknown+occluded→review)?
映射批处理引擎 decide_violations: confirmed=go(green/flashing 段) / review=review_light(unknown+occluded/inferred 段)。
裁判 GT: events.csv「违章01,0,66.5,red,occluded,0」→ 窗口内任何 green 读数=误绿。

红线: 只读, 不写生产码, 不建 worktree。产物仅诊断输出。
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config, project_root
from redlight.app import cli as cli_mod
from redlight.models.traffic_light import TrafficLightDetector
from redlight.pipeline import temporal_fusion as tf
from redlight.pipeline.violation_engine import BatchViolationEngine
from redlight.pipeline.decision import decide_violations

VIDEO = "违章01"
WINDOW_LO = 45.4
WINDOW_HI = 64.3

# ---------- 只读钩子状态 ----------
OBS_LATEST = {}          # 最近一次 observe 的归因(前向填充)
ACC_LOG = []             # 每推理帧归因(1:1 对齐 engine._light_obs)
_ENGINE_REF = {}
_TL_INSTANCES = []

_orig_tl_init = TrafficLightDetector.__init__


def _patched_tl_init(self, cfg, verbose=True):
    _orig_tl_init(self, cfg, verbose)
    _TL_INSTANCES.append(self)


_orig_observe = TrafficLightDetector.observe


def _patched_observe(self, frame, yolo_light_boxes=None):
    res = _orig_observe(self, frame, yolo_light_boxes)
    # 只读复算: 采样路径 + prior ROI 绿/红 + 全局亮斑绿/红 (不改生产 obs/conf)
    path = _classify_path_readonly(self, yolo_light_boxes)
    prior = _prior_roi_counts(frame, self.signal_prior, self.prior_roi_px)
    spots = res.get("candidates", []) if isinstance(res, dict) else []
    g_area = sum(s.get("area", 0) for s in spots if s.get("color") == "green")
    r_area = sum(s.get("area", 0) for s in spots if s.get("color") == "red")
    OBS_LATEST.clear()
    OBS_LATEST.update({
        "path": path,
        "prior_roi": prior,
        "global_g_area": g_area,
        "global_r_area": r_area,
        "last_sample_gn_rn": list(getattr(self, "_last_sample", (0, 0))),
        "signal_prior": list(self.signal_prior) if self.signal_prior else None,
        "n_yolo_light_boxes": len(yolo_light_boxes or []),
        "yolo_light_boxes": [[int(v) for v in b] for b in (yolo_light_boxes or [])],
        "_frame": frame,
    })
    return res


_orig_acc = BatchViolationEngine.accumulate


def _patched_accumulate(self, track_states, mask, light_observation, timestamp):
    obs = light_observation.get("obs") if isinstance(light_observation, dict) else None
    conf = light_observation.get("conf") if isinstance(light_observation, dict) else None
    ACC_LOG.append({
        "ts": round(float(timestamp), 3),
        "obs": obs, "conf": conf,
        "last_sample_gn_rn": OBS_LATEST.get("last_sample_gn_rn", [0, 0]),
        "signal_prior": OBS_LATEST.get("signal_prior"),
        "path": OBS_LATEST.get("path"),
        "prior_roi": OBS_LATEST.get("prior_roi"),
        "global_g_area": OBS_LATEST.get("global_g_area", 0),
        "global_r_area": OBS_LATEST.get("global_r_area", 0),
        "n_yolo_light_boxes": OBS_LATEST.get("n_yolo_light_boxes", 0),
        "yolo_light_boxes": OBS_LATEST.get("yolo_light_boxes", []),
    })
    return _orig_acc(self, track_states, mask, light_observation, timestamp)


_orig_decide = BatchViolationEngine.decide


def _patched_decide(self):
    _ENGINE_REF["eng"] = self
    return _orig_decide(self)


TrafficLightDetector.__init__ = _patched_tl_init
TrafficLightDetector.observe = _patched_observe
BatchViolationEngine.accumulate = _patched_accumulate
BatchViolationEngine.decide = _patched_decide


# ---------- 只读复算辅助 (不调用会改 _last_sample 的生产方法) ----------
def _classify_path_readonly(det, yb):
    sp = det.signal_prior
    h, w = (det._last_frame.shape[0], det._last_frame.shape[1]) if det._last_frame is not None else (1, 1)
    if yb and det._last_frame is not None:
        best = None
        best_dist = None
        for box in yb:
            x1, y1, x2, y2 = [int(v) for v in box]
            bcx = ((x1 + x2) / 2) / w
            bcy = ((y1 + y2) / 2) / h
            if bcy < det.yolo_cy_min:
                continue
            col, gn, rn = det._sample_box(box, w, h)
            if col is None:
                continue
            if sp is not None:
                px, py = sp
                dist = ((bcx - px) ** 2 + (bcy - py) ** 2) ** 0.5
                score = -dist
            else:
                dist = None
                score = max(gn, rn)
            if best is None or score > best[1]:
                best = (col, score, gn, rn)
                best_dist = dist
        use_yolo = best is not None and (sp is None or best_dist is None or best_dist > det.yolo_prior_near)
        if use_yolo:
            return "yolo_box"
    if sp is not None and det._last_frame is not None:
        # 只读复算 _sample_prior_color 的决策(不调用以避副作用)
        sampled = _prior_sample_color_readonly(det)
        if sampled is not None:
            return "prior_direct_sample"
        px, py = sp
        r = det.prior_search_radius
        near = [s for s in det._candidates(det._last_frame)
                if ((s["cx"] - px) ** 2 + (s["cy"] - py) ** 2) ** 0.5 <= r
                and s["color"] in ("green", "red")]
        if near:
            return "prior_radius_candidates"
        return "prior_hold_old_color"
    return "global_brightspot"


def _prior_sample_color_readonly(det):
    if det.signal_prior is None or det._last_frame is None:
        return None
    roi_px = det.prior_roi_px
    res = _sample_roi_readonly(det._last_frame, det.signal_prior[0], det.signal_prior[1], roi_px)
    if res is not None:
        return res
    if roi_px == det.prior_roi_px and det.prior_roi_expand_factor > 1.0:
        big = int(det.prior_roi_px * det.prior_roi_expand_factor)
        return _sample_roi_readonly(det._last_frame, det.signal_prior[0], det.signal_prior[1], big)
    return None


def _sample_roi_readonly(frame, px, py, roi_px):
    h, w = frame.shape[:2]
    cx_i, cy_i = int(px * w), int(py * h)
    x1 = max(0, cx_i - roi_px // 2)
    y1 = max(0, cy_i - roi_px // 2)
    x2 = min(w, cx_i + roi_px // 2)
    y2 = min(h, cy_i + roi_px // 2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g_mask = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([95, 255, 255]))
    r1 = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([12, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([158, 60, 40]), np.array([180, 255, 255]))
    r_mask = r1 | r2
    g_n = int(cv2.countNonZero(g_mask))
    r_n = int(cv2.countNonZero(r_mask))
    total = (x2 - x1) * (y2 - y1)
    if total == 0:
        return None
    g_frac, r_frac = g_n / total, r_n / total
    if g_frac < 0.002 and r_frac < 0.002:
        return None
    if g_frac > r_frac * 1.3:
        return "green"
    if r_frac > g_frac * 1.3:
        return "red"
    return "green" if g_frac >= r_frac else "red"


def _prior_roi_counts(frame, signal_prior, roi_px):
    if frame is None or signal_prior is None:
        return None
    h, w = frame.shape[:2]
    px, py = signal_prior
    cx_i, cy_i = int(px * w), int(py * h)
    x1 = max(0, cx_i - roi_px // 2)
    y1 = max(0, cy_i - roi_px // 2)
    x2 = min(w, cx_i + roi_px // 2)
    y2 = min(h, cy_i + roi_px // 2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g_mask = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([95, 255, 255]))
    r1 = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([12, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([158, 60, 40]), np.array([180, 255, 255]))
    r_mask = r1 | r2
    total = (x2 - x1) * (y2 - y1)
    gn = int(cv2.countNonZero(g_mask))
    rn = int(cv2.countNonZero(r_mask))
    # 绿斑质心(相对 ROI 归一化): 集中紧凑≈灯泡, 弥散≈环境绿(树叶/车漆/反光)
    g_cx = g_cy = None
    if gn > 0:
        mom = cv2.moments(g_mask)
        if mom["m00"] > 0:
            g_cx = round((mom["m10"] / mom["m00"]) / max(1, (x2 - x1)), 3)
            g_cy = round((mom["m01"] / mom["m00"]) / max(1, (y2 - y1)), 3)
    return {
        "roi": [x1, y1, x2, y2],
        "g_n": gn, "r_n": rn, "total": total,
        "g_frac": round(gn / total, 4) if total else 0.0,
        "r_frac": round(rn / total, 4) if total else 0.0,
        "g_cx": g_cx, "g_cy": g_cy,
    }


def main():
    ap = argparse.ArgumentParser(description="违章01 FP 生产根因只读诊断")
    ap.add_argument("--config", default=os.path.join(project_root(), "configs", "config.yaml"))
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--output", default=os.path.join(project_root(), "data", "output", "diag_01fp"))
    ap.add_argument("--evidence", action="store_true", help="导出窗口内绿帧叠框证据图")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.output.annotated_video = False
    cfg.output.evidence_images = False
    cfg.output.csv_report = True

    video_path = os.path.join(project_root(), "input_video", VIDEO + ".mp4")
    os.makedirs(args.output, exist_ok=True)

    # 清空钩子状态
    OBS_LATEST.clear(); ACC_LOG.clear(); _ENGINE_REF.clear(); _TL_INSTANCES.clear()

    events, track_samples = cli_mod.run(cfg, video_path, args.output, args.preset,
                                        return_track_samples=True)

    eng = _ENGINE_REF["eng"]
    light_obs = eng._light_obs
    occ_samples = eng._occ_samples

    # 断言: ACC_LOG 与 engine._light_obs 同序同长(均每推理帧一次)
    assert len(ACC_LOG) == len(light_obs), \
        f"ACC_LOG({len(ACC_LOG)}) != light_obs({len(light_obs)}) 对齐失败"

    # 重新融合灯态段(与生产 decide 同源), 判定分支
    tl = getattr(cfg, "traffic_light", None)
    fuse_kwargs = dict(
        window=int(getattr(tl, "smoothing_window", 24)),
        hysteresis=float(getattr(tl, "hysteresis", 0.68)),
        flicker_toggle=int(getattr(tl, "flicker_toggle_count", 4)),
        unknown_hold=int(getattr(tl, "anchor_hold", 30)),
    )
    light_segments = tf.fuse_light(light_obs, **fuse_kwargs)
    light_segments = tf.tag_evidence(light_segments, occ_samples)

    # 分支判定: 01 事件 status + 对应灯段
    fp_events = [e for e in events if e["status"] in ("confirmed", "review")]
    branch = None
    seg_evidence = []
    for e in fp_events:
        s, en = e["start_ts"], e["end_ts"]
        seg = [sg for sg in light_segments if sg["start_s"] < en and sg["end_s"] > s]
        seg_evidence.append({
            "status": e["status"], "start_s": s, "end_s": en,
            "light_state": e["light_state"], "track_id": e["track_id"],
            "overlapping_segments": [
                {"state": sg["state"], "evidence": sg.get("evidence"),
                 "start_s": round(sg["start_s"], 2), "end_s": round(sg["end_s"], 2)}
                for sg in seg],
        })
        if e["status"] == "confirmed":
            branch = "P87_confirmed_green"   # :87 on_crosswalk+green/flashing
        elif e["status"] == "review":
            branch = "P90_review_unknown_occluded"

    # 窗口内逐帧统计
    win = [a for a in ACC_LOG if WINDOW_LO <= a["ts"] <= WINDOW_HI]
    green_frames = [a for a in win if a["obs"] == "green"]
    prior_green_frames = [a for a in win
                          if a.get("prior_roi") and a["prior_roi"]["g_frac"] > a["prior_roi"]["r_frac"] * 1.3]
    prior_strong_green_frames = [a for a in win
                                 if a.get("prior_roi") and a["prior_roi"]["g_frac"] >= 0.10]
    global_green_frames = [a for a in win if a["global_g_area"] > a["global_r_area"]]

    summary = {
        "video": VIDEO,
        "gt": "events.csv: 违章01,0,66.5,red,occluded,0 (全程红, 真负例)",
        "window": [WINDOW_LO, WINDOW_HI],
        "production_events": [
            {k: e[k] for k in ("event_id", "track_id", "status", "start_ts", "end_ts",
                               "vehicle_class", "confidence", "light_state")}
            for e in events],
        "branch_verdict": branch,
        "fp_event_detail": seg_evidence,
        "window_frame_count": len(win),
        "window_green_obs_frames": len(green_frames),
        "window_prior_roi_green_frames": len(prior_green_frames),
        "window_prior_roi_strong_green_frames": len(prior_strong_green_frames),
        "window_global_brightspot_green_frames": len(global_green_frames),
        "window_green_ts_list": [a["ts"] for a in green_frames],
        "sample_window_rows": win[:6] + win[-6:],
        "prior_used": any(a["signal_prior"] for a in win),
        "prior_value": next((a["signal_prior"] for a in win if a["signal_prior"]), None),
    }

    out_json = os.path.join(args.output, "diag_01fp.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "acc_log": ACC_LOG,
                   "light_segments": [
                       {"state": s["state"], "evidence": s.get("evidence"),
                        "start_s": round(s["start_s"], 2), "end_s": round(s["end_s"], 2),
                        "conf": s.get("conf")}
                       for s in light_segments]}, f, ensure_ascii=False, indent=2)

    # 交叉核对: 重跑产出的 violations.csv 应与 cc_baseline 一致
    reproduced_csv = os.path.join(args.output, "violations.csv")
    print(f"[diag] 产物: {out_json}")
    print(f"[diag] 重跑 violations.csv: {reproduced_csv}")
    print(f"[diag] 分支判定 = {branch}")
    print(f"[diag] 窗口 [{WINDOW_LO},{WINDOW_HI}] 帧数={len(win)} "
          f"obs==green {len(green_frames)} / prior_roi绿 {len(prior_green_frames)} "
          f"/ 全局亮斑绿 {len(global_green_frames)}")
    print(f"[diag] prior 启用 = {summary['prior_used']} 值={summary['prior_value']}")

    # 可选: 导出窗口内绿帧叠框证据图
    if args.evidence:
        ev_dir = os.path.join(args.output, "evidence")
        os.makedirs(ev_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sp = summary["prior_value"]
        roi_px = 160
        for a in green_frames:
            ts = a["ts"]
            cap.set(cv2.CAP_PROP_POS_MSEC, int(ts * 1000))
            ret, frame = cap.read()
            if not ret:
                continue
            if sp:
                cx_i, cy_i = int(sp[0] * frame.shape[1]), int(sp[1] * frame.shape[0])
                x1, y1 = max(0, cx_i - roi_px // 2), max(0, cy_i - roi_px // 2)
                x2, y2 = min(frame.shape[1], cx_i + roi_px // 2), min(frame.shape[0], cy_i + roi_px // 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for b in a.get("yolo_light_boxes", []):
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
            fname = os.path.join(ev_dir, f"green_ts{ts:.2f}.jpg")
            cv2.imwrite(fname, frame)
        cap.release()
        print(f"[diag] 证据图已导出: {ev_dir} ({len(green_frames)} 张)")


if __name__ == "__main__":
    main()
