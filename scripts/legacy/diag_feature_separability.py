#!/usr/bin/env python3
"""特征可分性横测 (01 假绿 vs 05/06/07/08/09 真绿), wb, 只读。

cc 裁定(本轮): 修法#1/#2 及一切外观阈值/形状变体出局, 转结构路线 #3(遮挡降级 unknown)。
强制门: 先证明"所选机制"真能分开「01 假绿」与「05/06/07/08/09 真绿」再选特征——wb 上次错在预锁 solidity 再调阈值。

本脚本只测特征可分性, 不改生产码, 不建 worktree。
方法: 复用 cli.run(零循环复制), monkeypatch observe 抓取每帧生产 obs 及"决定绿的区域"的特征电池。
对所有 6 视频(light_priors.json 均有 prior, 路径统一)统计绿帧特征分布, 比对 01(假) vs 其余(真)。
硬试金石: 06 低饱和真绿 S 86-120 不得被任何机制屠真绿 → 报告 06 绿帧 S 分布。

红线: 只读诊断; 禁 select_gtfree; 权重不入库; 不碰 ped_signal.pt; 过 gate 前不写生产码。
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
from redlight.pipeline.violation_engine import BatchViolationEngine

VIDEOS = ["违章01", "违章05", "违章06", "违章07", "违章08", "违章09"]

# ---------- 只读钩子状态 ----------
OBS_LATEST = {}
ACC_LOG = []
_ENGINE_REF = {}
_TL_INSTANCES = []

_orig_tl_init = TrafficLightDetector.__init__


def _patched_tl_init(self, cfg, verbose=True):
    _orig_tl_init(self, cfg, verbose)
    _TL_INSTANCES.append(self)


_orig_observe = TrafficLightDetector.observe


def _patched_observe(self, frame, yolo_light_boxes=None):
    res = _orig_observe(self, frame, yolo_light_boxes)
    path = _classify_path(self, yolo_light_boxes)
    feat = _region_feature_battery(frame, self, path, yolo_light_boxes)
    OBS_LATEST.clear()
    OBS_LATEST.update({
        "obs": res.get("obs") if isinstance(res, dict) else None,
        "path": path,
        "feat": feat,
        "signal_prior": list(self.signal_prior) if self.signal_prior else None,
        "n_yolo": len(yolo_light_boxes or []),
    })
    return res


_orig_acc = BatchViolationEngine.accumulate


def _patched_accumulate(self, track_states, mask, light_observation, timestamp):
    obs = light_observation.get("obs") if isinstance(light_observation, dict) else None
    ACC_LOG.append({
        "ts": round(float(timestamp), 3),
        "obs": obs,
        "path": OBS_LATEST.get("path"),
        "n_yolo": OBS_LATEST.get("n_yolo"),
        "feat": OBS_LATEST.get("feat") if obs == "green" else None,
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


# ---------- 只读复算(不改生产与 _last_sample) ----------
def _classify_path(det, yb):
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
        # 只读复算 _sample_prior_color 决策
        roi_px = det.prior_roi_px
        if _sample_roi_readonly(det._last_frame, sp[0], sp[1], roi_px) is not None:
            return "prior_direct_sample"
        r = det.prior_search_radius
        near = [s for s in det._candidates(det._last_frame)
                if ((s["cx"] - sp[0]) ** 2 + (s["cy"] - sp[1]) ** 2) ** 0.5 <= r
                and s["color"] in ("green", "red")]
        if near:
            return "prior_radius_candidates"
        return "prior_hold_old_color"
    return "global_brightspot"


def _sample_roi_readonly(frame, px, py, roi_px):
    h, w = frame.shape[:2]
    cx_i, cy_i = int(px * w), int(py * h)
    x1 = max(0, cx_i - roi_px // 2); y1 = max(0, cy_i - roi_px // 2)
    x2 = min(w, cx_i + roi_px // 2); y2 = min(h, cy_i + roi_px // 2)
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
    gn = int(cv2.countNonZero(g_mask)); rn = int(cv2.countNonZero(r_mask))
    total = (x2 - x1) * (y2 - y1)
    if total == 0:
        return None
    g_frac, r_frac = gn / total, rn / total
    if g_frac < 0.002 and r_frac < 0.002:
        return None
    if g_frac > r_frac * 1.3:
        return "green"
    if r_frac > g_frac * 1.3:
        return "red"
    return "green" if g_frac >= r_frac else "red"


def _choose_box(det, yb):
    sp = det.signal_prior
    h, w = (det._last_frame.shape[0], det._last_frame.shape[1]) if det._last_frame is not None else (1, 1)
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
            best = (col, score, gn, rn, box)
            best_dist = dist
    return best[4] if best else None


_FEATURE_KEYS = [
    "g_frac", "r_frac", "gn",
    "s_med", "s_mean", "s_p10", "s_min", "frac_s_lt120", "frac_s_lt130", "v_med",
    "max_sol", "max_area", "max_fill", "n_comp", "disp_x", "disp_y", "dist_c",
]


def _region_feature_battery(frame, det, path, yb):
    """从"决定绿的区域"提取特征电池。统一 prior ROI / YOLO 框 / 全局最大绿连通域。"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if path == "yolo_box" and yb:
        box = _choose_box(det, yb)
        if not box:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
    elif path in ("prior_direct_sample", "prior_radius_candidates") and det.signal_prior:
        px, py = det.signal_prior
        roi_px = det.prior_roi_px
        cx, cy = int(px * w), int(py * h)
        x1 = max(0, cx - roi_px // 2); y1 = max(0, cy - roi_px // 2)
        x2 = min(w, cx + roi_px // 2); y2 = min(h, cy + roi_px // 2)
    elif path == "global_brightspot":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        g_mask_full = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([95, 255, 255]))
        nb, lbl, stats, _ = cv2.connectedComponentsWithStats(g_mask_full, 8)
        if nb <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        bi = 1 + int(np.argmax(areas))
        x1 = int(stats[bi, cv2.CC_STAT_LEFT]); y1 = int(stats[bi, cv2.CC_STAT_TOP])
        x2 = x1 + int(stats[bi, cv2.CC_STAT_WIDTH]); y2 = y1 + int(stats[bi, cv2.CC_STAT_HEIGHT])
    else:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g_mask = cv2.inRange(hsv_roi, np.array([35, 60, 40]), np.array([95, 255, 255]))
    r_mask = (cv2.inRange(hsv_roi, np.array([0, 60, 40]), np.array([12, 255, 255]))
             | cv2.inRange(hsv_roi, np.array([158, 60, 40]), np.array([180, 255, 255])))
    total = (x2 - x1) * (y2 - y1)
    gn = int(cv2.countNonZero(g_mask)); rn = int(cv2.countNonZero(r_mask))
    g_frac = gn / total; r_frac = rn / total
    if gn == 0:
        return {"region": path, "g_frac": round(g_frac, 4), "r_frac": round(r_frac, 4), "gn": 0}
    S = hsv_roi[:, :, 1][g_mask > 0].astype(np.float32)
    V = hsv_roi[:, :, 2][g_mask > 0].astype(np.float32)
    s_med = float(np.median(S)); s_mean = float(np.mean(S))
    s_p10 = float(np.percentile(S, 10)); s_min = float(np.min(S))
    frac_s_lt120 = float(np.mean(S < 120)); frac_s_lt130 = float(np.mean(S < 130))
    v_med = float(np.median(V))
    nb, lbl, stats, _ = cv2.connectedComponentsWithStats(g_mask, 8)
    if nb <= 1:
        max_sol = 0.0; max_area = 0; max_fill = 0.0; n_comp = 0
    else:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        ws = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
        hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        bbs = np.maximum(ws * hs, 1.0)
        solids = areas / bbs
        bi = int(np.argmax(areas))
        max_sol = float(solids[bi]); max_area = int(areas[bi])
        max_fill = float(areas[bi] / max(1, gn))
        n_comp = int(nb - 1)
    ys, xs = np.where(g_mask > 0)
    if len(xs) > 1:
        disp_x = float(np.std(xs) / (x2 - x1)); disp_y = float(np.std(ys) / (y2 - y1))
    else:
        disp_x = disp_y = 0.0
    mom = cv2.moments(g_mask)
    gcx = float(mom["m10"] / mom["m00"]) / max(1, (x2 - x1)) if mom["m00"] > 0 else 0.5
    gcy = float(mom["m01"] / mom["m00"]) / max(1, (y2 - y1)) if mom["m00"] > 0 else 0.5
    dist_c = float(((gcx - 0.5) ** 2 + (gcy - 0.5) ** 2) ** 0.5)
    return {
        "region": path, "g_frac": round(g_frac, 4), "r_frac": round(r_frac, 4), "gn": gn,
        "s_med": round(s_med, 1), "s_mean": round(s_mean, 1), "s_p10": round(s_p10, 1),
        "s_min": round(s_min, 1), "frac_s_lt120": round(frac_s_lt120, 3),
        "frac_s_lt130": round(frac_s_lt130, 3), "v_med": round(v_med, 1),
        "max_sol": round(max_sol, 3), "max_area": max_area, "max_fill": round(max_fill, 3),
        "n_comp": n_comp, "disp_x": round(disp_x, 3), "disp_y": round(disp_y, 3),
        "dist_c": round(dist_c, 3),
    }


def _aggregate(feats):
    """feats: list[dict]; 返回每特征 median/p10/p90/n。"""
    if not feats:
        return {"n": 0}
    out = {"n": len(feats)}
    for k in _FEATURE_KEYS:
        vals = [f[k] for f in feats if k in f and f.get("gn", 0) > 0]
        if not vals:
            out[k] = None
            continue
        arr = np.array(vals, dtype=np.float32)
        out[k] = {
            "median": round(float(np.median(arr)), 3),
            "p10": round(float(np.percentile(arr, 10)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="特征可分性横测 (01 假绿 vs 05/06/07/08/09 真绿)")
    ap.add_argument("--config", default=os.path.join(project_root(), "configs", "config.yaml"))
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--output", default=os.path.join(project_root(), "data", "output", "diag_separability"))
    ap.add_argument("--videos", default=",".join(VIDEOS), help="逗号分隔视频名子集")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.output.annotated_video = False
    cfg.output.evidence_images = False
    cfg.output.csv_report = True

    os.makedirs(args.output, exist_ok=True)
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    aggregates = {}
    per_video_green = {}
    for v in videos:
        OBS_LATEST.clear(); ACC_LOG.clear(); _ENGINE_REF.clear(); _TL_INSTANCES.clear()
        video_path = os.path.join(project_root(), "input_video", v + ".mp4")
        if not os.path.exists(video_path):
            print(f"[sep] 跳过(缺视频): {video_path}")
            aggregates[v] = {"error": "missing_video"}
            continue
        try:
            cli_mod.run(cfg, video_path, args.output, args.preset, return_track_samples=True)
        except Exception as e:  # 单视频失败不阻断其余
            print(f"[sep] {v} 运行异常: {e!r}")
            aggregates[v] = {"error": str(e)}
            continue
        greens = [a["feat"] for a in ACC_LOG if a.get("feat")]
        aggregates[v] = _aggregate(greens)
        per_video_green[v] = greens
        print(f"[sep] {v}: 绿帧 {len(greens)} | s_med median="
              f"{aggregates[v].get('s_med')} max_sol median={aggregates[v].get('max_sol')} "
              f"frac_s_lt120 median={aggregates[v].get('frac_s_lt120')}")

    # 横测: 01(假) vs 其余(真) 每特征 median 对比
    false_cls = aggregates.get("违章01", {})
    real_cls = []
    for v in videos:
        if v == "违章01":
            continue
        a = aggregates.get(v)
        if a and "n" in a and a["n"] > 0:
            real_cls.append((v, a))
    sep_rows = []
    if false_cls.get("n"):
        for k in _FEATURE_KEYS:
            fv = false_cls.get(k)
            if not fv:
                continue
            real_meds = [ra[k]["median"] for (_v, ra) in real_cls if ra.get(k)]
            if not real_meds:
                continue
            real_med = float(np.median(real_meds))
            sep_rows.append({
                "feature": k,
                "false_01_median": fv["median"],
                "real_median": round(real_med, 3),
                "real_p10": round(float(np.min([ra[k]["p10"] for (_v, ra) in real_cls if ra.get(k)])), 3),
                "real_p90": round(float(np.max([ra[k]["p90"] for (_v, ra) in real_cls if ra.get(k)])), 3),
                "gap": round(fv["median"] - real_med, 3),
            })

    out = {"aggregates": aggregates, "sep_rows": sep_rows,
           "videos": videos, "real_class_videos": [v for (v, _) in real_cls]}
    out_json = os.path.join(args.output, "diag_separability.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[sep] 产物: {out_json}")
    print(f"[sep] 横测 (01 假绿 vs 其余真绿) median 对比:")
    for r in sep_rows:
        print(f"  {r['feature']:>14}: 01={r['false_01_median']:>8}  真绿中位={r['real_median']:>8}  "
              f"(真绿 p10={r['real_p10']} p90={r['real_p90']})  gap={r['gap']}")


if __name__ == "__main__":
    main()
