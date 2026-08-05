#!/usr/bin/env python3
"""违章01 FP — 最后一轮时序正交轴可分性横测 (brief 33965cf, wb, 只读)。

四路已死(外观 sat/solidity + 结构 遮挡/YOLO, cc 已亲验), Jacob 拍板只测两条**时序**正交轴,
且这是 01 FP 快修的最后一轮; 硬停止规则: 两轴都不能留 margin 分开 01 假绿 vs 全部真绿
(且过不了 06 低饱和石 S86-120) → de-scope 为已知限制(同 04 FN), 治本并入 retrain 线。

两条候选时序轴(不预锁, 横测取胜者):
  A. 绿持续性/闪烁 (raw observe 层, 融合前): 逐帧 raw obs 序列的 green run 时长分布 +
     红打断频次/最长红缝隙。真绿=单段长 run; 01 假绿=短 run + ~2.3s 红缝隙(过路车/反光瞬态)。
     ⚠️防栽坑①②: 只在融合前 raw 层测; 必须同时量真绿(尤其 06/09)的红缝隙, 不能只量 01。
  B. 绿斑位置稳定性: prior ROI 内绿斑质心 (g_cx,g_cy) 逐帧漂移(方差/最大步移)。
     真灯定点(低漂移); 过路车绿漂移(高)。⚠️防栽坑③: 报告逐视频, 真绿视频若机位更抖会反被
     误判漂移 → 只要任一真绿视频漂移 ≥ 01, 轴 B 即无 margin 死亡(诚实报重叠)。

红线: 只读, 不写生产码, 不建 worktree。禁用 select_gtfree。产物仅诊断输出(gitignored)。
方法: 复用生产入口 cli.run(零循环复制), 仅用只读 monkeypatch 抓逐帧 raw obs + prior ROI 绿斑质心。
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

# ---------- 只读钩子状态 ----------
OBS_LATEST = {}          # 最近一次 observe 归因(前向填充)
ACC_LOG = []             # 每推理帧归因
_ENGINE_REF = {}
_TL_INSTANCES = []

_orig_tl_init = TrafficLightDetector.__init__


def _patched_tl_init(self, cfg, verbose=True):
    _orig_tl_init(self, cfg, verbose)
    _TL_INSTANCES.append(self)


_orig_observe = TrafficLightDetector.observe


def _patched_observe(self, frame, yolo_light_boxes=None):
    res = _orig_observe(self, frame, yolo_light_boxes)
    # 只读复算 prior ROI 绿斑质心(不改生产 obs/conf, 不调有副作用的生产方法)
    prior = _prior_roi_counts(frame, self.signal_prior, self.prior_roi_px)
    OBS_LATEST.clear()
    OBS_LATEST.update({
        "signal_prior": list(self.signal_prior) if self.signal_prior else None,
        "g_cx": prior["g_cx"] if prior else None,
        "g_cy": prior["g_cy"] if prior else None,
        "g_n": prior["g_n"] if prior else 0,
    })
    return res


_orig_acc = BatchViolationEngine.accumulate


def _patched_accumulate(self, track_states, mask, light_observation, timestamp):
    obs = light_observation.get("obs") if isinstance(light_observation, dict) else None
    ACC_LOG.append({
        "ts": round(float(timestamp), 3),
        "obs": obs,
        "signal_prior": OBS_LATEST.get("signal_prior"),
        "g_cx": OBS_LATEST.get("g_cx"),
        "g_cy": OBS_LATEST.get("g_cy"),
        "g_n": OBS_LATEST.get("g_n", 0),
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


def _prior_roi_counts(frame, signal_prior, roi_px):
    """只读复算 prior ROI 内绿/红像素 + 绿斑质心(相对 ROI 归一化)。
    与 diag_01fp_repro._prior_roi_counts 同源, 仅保留本脚本需要的字段。"""
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
    gn = int(cv2.countNonZero(g_mask))
    g_cx = g_cy = None
    if gn > 0:
        mom = cv2.moments(g_mask)
        if mom["m00"] > 0:
            g_cx = round((mom["m10"] / mom["m00"]) / max(1, (x2 - x1)), 3)
            g_cy = round((mom["m01"] / mom["m00"]) / max(1, (y2 - y1)), 3)
    return {"g_n": gn, "g_cx": g_cx, "g_cy": g_cy}


def _run_one_video(cfg, video, out_dir, preset):
    OBS_LATEST.clear(); ACC_LOG.clear()
    video_path = os.path.join(project_root(), "input_video", video + ".mp4")
    if not os.path.exists(video_path):
        print(f"[warn] 缺失视频: {video_path}")
        return []
    cli_mod.run(cfg, video_path, out_dir, preset, return_track_samples=True)
    return [dict(r, video=video) for r in ACC_LOG]


def _axis_a_green_runs(rows):
    """在 raw observe 层(融合前)测绿 run + 红缝隙。rows 已按 ts 排序。
    返回: max_green_run_s, n_green_runs, longest_red_gap_s(被绿 flank 的红缝隙)。"""
    # 仅取有 obs 的推理帧, 按 ts 排序
    seq = sorted([(r["ts"], r["obs"]) for r in rows if r["obs"] is not None],
                 key=lambda x: x[0])
    if not seq:
        return {"max_green_run_s": 0.0, "n_green_runs": 0, "longest_red_gap_s": 0.0,
                "total_green_s": 0.0}
    # 绿 run 划分: 连续 obs=='green' 为一段
    green_runs = []
    red_gaps = []
    i = 0
    n = len(seq)
    last_green_end = None
    while i < n:
        ts, obs = seq[i]
        if obs == "green":
            j = i
            while j < n and seq[j][1] == "green":
                j += 1
            run_start = seq[i][0]
            run_end = seq[j - 1][0]
            green_runs.append((run_start, run_end, round(run_end - run_start, 3)))
            last_green_end = run_end
            i = j
        elif obs == "red":
            # 若前一段是绿, 且下一段不久后是绿 → 记为红缝隙
            j = i
            while j < n and seq[j][1] == "red":
                j += 1
            gap_start = seq[i][0]
            gap_end = seq[j - 1][0]
            gap_dur = round(gap_end - gap_start, 3)
            # 只记被绿 flank 的红缝隙(前或后有绿, 且间隔 < 10s 视为同一绿活跃期)
            prev_green = last_green_end is not None and (gap_start - last_green_end) < 10.0
            next_is_green = j < n and seq[j][1] == "green"
            if prev_green or next_is_green:
                red_gaps.append(gap_dur)
            i = j
        else:
            i += 1
    max_run = max((g[2] for g in green_runs), default=0.0)
    total_green = round(sum(g[2] for g in green_runs), 3)
    longest_red = max(red_gaps, default=0.0)
    return {
        "max_green_run_s": round(max_run, 3),
        "n_green_runs": len(green_runs),
        "longest_red_gap_s": round(longest_red, 3),
        "total_green_s": total_green,
        "_green_runs": green_runs,
        "_red_gaps": red_gaps,
    }


def _axis_b_centroid_drift(rows):
    """绿斑质心逐帧漂移(相对 prior ROI 归一化)。仅在 prior ROI 有绿像素的帧上测。"""
    pts = [(r["ts"], r["g_cx"], r["g_cy"]) for r in rows
           if r.get("g_cx") is not None and r.get("g_cy") is not None]
    pts.sort(key=lambda x: x[0])
    if len(pts) < 2:
        return {"n": len(pts), "cx_std": 0.0, "cy_std": 0.0,
                "max_step_drift": 0.0, "mean_step_drift": 0.0}
    cxs = np.array([p[1] for p in pts])
    cys = np.array([p[2] for p in pts])
    step_drifts = []
    for k in range(1, len(pts)):
        d = ((pts[k][1] - pts[k - 1][1]) ** 2 + (pts[k][2] - pts[k - 1][2]) ** 2) ** 0.5
        step_drifts.append(round(float(d), 4))
    return {
        "n": len(pts),
        "cx_std": round(float(np.std(cxs)), 4),
        "cy_std": round(float(np.std(cys)), 4),
        "max_step_drift": max(step_drifts),
        "mean_step_drift": round(float(np.mean(step_drifts)), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="违章01 FP 时序轴可分性横测 (raw层绿run/红缝隙 + 绿斑漂移)")
    ap.add_argument("--config", default=os.path.join(project_root(), "configs", "config.yaml"))
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--videos", required=True,
                    help="逗号分隔视频名, 例: 违章01,违章05,违章06,违章07,违章08,违章09")
    ap.add_argument("--false-video", default="违章01",
                    help="已知假绿视频(全程红, 任何绿=误绿)")
    ap.add_argument("--output", default=os.path.join(project_root(), "data", "output", "diag_temporal_sep"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.output.annotated_video = False
    cfg.output.evidence_images = False
    cfg.output.csv_report = True

    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    os.makedirs(args.output, exist_ok=True)

    all_rows = []
    for v in videos:
        out_dir = os.path.join(args.output, "runs", v)
        rows = _run_one_video(cfg, v, out_dir, args.preset)
        all_rows.extend(rows)

    # 逐视频算两轴
    per_video = {}
    for v in videos:
        vrows = [r for r in all_rows if r["video"] == v]
        a = _axis_a_green_runs(vrows)
        b = _axis_b_centroid_drift(vrows)
        label = "false_green" if v == args.false_video else "true_green"
        per_video[v] = {"label": label, "axis_a": a, "axis_b": b}

    # 分离度判定 (01 假绿 vs 全部真绿聚合)
    fg_a = per_video[args.false_video]["axis_a"]
    tg_a_runs = [per_video[v]["axis_a"]["max_green_run_s"]
                 for v in videos if v != args.false_video]
    tg_a_gaps = [per_video[v]["axis_a"]["longest_red_gap_s"]
                 for v in videos if v != args.false_video]
    tg_a_nruns = [per_video[v]["axis_a"]["n_green_runs"]
                  for v in videos if v != args.false_video]

    fg_b = per_video[args.false_video]["axis_b"]
    tg_b_step = [per_video[v]["axis_b"]["max_step_drift"]
                 for v in videos if v != args.false_video]
    tg_b_mean = [per_video[v]["axis_b"]["mean_step_drift"]
                 for v in videos if v != args.false_video]

    # 轴A 判定: 01 的最大绿 run 应明显短于真绿, 且 01 红缝隙应明显多于真绿
    # 防栽坑①: 真绿(尤其06/09)若也有短 run / 红缝隙 → 轴A 无 margin
    a_verdict = _axis_a_verdict(fg_a, tg_a_runs, tg_a_gaps, tg_a_nruns, per_video, args.false_video, videos)

    # 轴B 判定: 01 漂移应明显大于真绿; 防栽坑③: 任一真绿漂移 ≥ 01 → 无 margin
    b_verdict = _axis_b_verdict(fg_b, tg_b_step, tg_b_mean, per_video, args.false_video, videos)

    overall = _hard_stop(fg_a, fg_b, a_verdict, b_verdict)

    report = {
        "videos": videos,
        "false_video": args.false_video,
        "per_video": {v: {"label": d["label"],
                          "axis_a": {k: val for k, val in d["axis_a"].items() if not k.startswith("_")},
                          "axis_b": d["axis_b"]}
                      for v, d in per_video.items()},
        "axis_a_verdict": a_verdict,
        "axis_b_verdict": b_verdict,
        "hard_stop_rule": overall,
    }

    out_json = os.path.join(args.output, "diag_temporal_sep.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台表
    print("=" * 82)
    print("[sep] 时序轴可分性横测 — 轴A=raw层绿run/红缝隙, 轴B=prior ROI绿斑质心漂移")
    print("-" * 82)
    print(f"[sep] {'video':<8} {'label':<11} {'maxGreenRun':>12} {'nRuns':>6} {'maxRedGap':>10} {'maxStepDrift':>13} {'meanStepDrift':>14}")
    for v in videos:
        d = per_video[v]
        a, b = d["axis_a"], d["axis_b"]
        print(f"[sep] {v:<8} {d['label']:<11} {a['max_green_run_s']:>11.2f}s "
              f"{a['n_green_runs']:>6} {a['longest_red_gap_s']:>9.2f}s "
              f"{b['max_step_drift']:>13.3f} {b['mean_step_drift']:>14.3f}")
    print("-" * 82)
    print(f"[sep] 轴A 判定: {a_verdict['verdict']}")
    print(f"[sep]   01 maxGreenRun={fg_a['max_green_run_s']}s nRuns={fg_a['n_green_runs']} "
          f"maxRedGap={fg_a['longest_red_gap_s']}s")
    print(f"[sep]   真绿 maxGreenRun 集合={tg_a_runs}  maxRedGap 集合={tg_a_gaps}")
    print(f"[sep] 轴B 判定: {b_verdict['verdict']}")
    print(f"[sep]   01 maxStepDrift={fg_b['max_step_drift']} meanStepDrift={fg_b['mean_step_drift']}")
    print(f"[sep]   真绿 maxStepDrift 集合={tg_b_step}")
    print("-" * 82)
    print(f"[sep] 硬停止规则(Jacob): {overall['verdict']}")
    print("=" * 82)
    print(f"[sep] 产物: {out_json}")


def _axis_a_verdict(fg_a, tg_runs, tg_gaps, tg_nruns, per_video, fv, videos):
    """轴A: 判别核心是「绿 run 持续性」(brief 立足点: 瞬态 vs 持续), 不是红缝隙。

    ⚠️ 红缝隙不能作为硬判别 —— 防栽坑①实证: 真绿视频也有长红缝隙(红灯期),
       如 07 maxRedGap=34.3s ≫ 01 的 18.2s, 用红缝隙反而会把真绿判成"闪"。
       所以主判别只用 max_green_run_s(最长单段绿 run): 假绿瞬态(碎短 run),
       真绿持续(单段很长)。
    """
    # 01 指标
    fg_run = fg_a["max_green_run_s"]
    fg_gap = fg_a["longest_red_gap_s"]
    fg_nrun = fg_a["n_green_runs"]
    # 真绿指标(最小 max-run 代表最脆弱, 06 石在此)
    min_tg_run = min(tg_runs) if tg_runs else None
    max_tg_gap = max(tg_gaps) if tg_gaps else None
    max_tg_nrun = max(tg_nruns) if tg_nruns else None
    # 主判别: 01 最长绿 run 远短于真绿最短者(留足 margin, *0.5)
    run_margin = (min_tg_run is not None) and (fg_run < min_tg_run * 0.5)
    # 佐证: 01 绿碎成更多段
    nrun_margin = (fg_nrun >= 2) and (max_tg_nrun is not None) and (fg_nrun > max_tg_nrun)
    notes = []
    if not run_margin:
        notes.append(f"01 maxRun({fg_run})不短于真绿最短({min_tg_run}) → 无margin")
    if max_tg_gap is not None and max_tg_gap > fg_gap:
        notes.append(f"参考: 真绿最长红缝隙({max_tg_gap}s, 红灯期)≫01({fg_gap}s) "
                      f"→ 红缝隙不能单独判别(防栽坑①)")
    sep = run_margin
    return {
        "false_max_green_run_s": fg_run,
        "false_n_green_runs": fg_nrun,
        "false_longest_red_gap_s": fg_gap,
        "true_min_green_run_s": min_tg_run,
        "true_max_red_gap_s": max_tg_gap,
        "true_max_n_green_runs": max_tg_nrun,
        "run_margin": run_margin,
        "nrun_margin": nrun_margin,
        "separable": sep,
        "verdict": ("活: 01最长绿run({fg_run}s)远短于真绿最短({min_run}s), 瞬态vs持续可分"
                    .format(fg_run=fg_run, min_run=min_tg_run)
                    if sep else "死: 无margin(真绿也短run, 过06石失败)"),
        "notes": notes,
    }


def _axis_b_verdict(fg_b, tg_step, tg_mean, per_video, fv, videos):
    """轴B: 01 漂移 > 真绿。防栽坑③: 任一真绿漂移 ≥ 01 → 无 margin。"""
    fg_step_max = fg_b["max_step_drift"]
    fg_step_mean = fg_b["mean_step_drift"]
    # 真绿中最脆弱: 最大 step 漂移(若某真绿视频机位更抖)
    max_tg_step = max(tg_step) if tg_step else None
    # 分离判据: 01 漂移显著大于真绿最大者
    margin = (max_tg_step is not None) and (fg_step_max > max_tg_step * 1.5)
    notes = []
    tie_videos = [v for v in videos if v != fv and per_video[v]["axis_b"]["max_step_drift"] >= fg_step_max]
    if tie_videos:
        notes.append(f"真绿视频 {tie_videos} 漂移≥01 → 无margin(防栽坑③手持抖动混淆)")
    if max_tg_step is None:
        margin = False
        notes.append("无真绿质心数据")
    return {
        "false_max_step_drift": fg_step_max,
        "false_mean_step_drift": fg_step_mean,
        "true_max_step_drift": max_tg_step,
        "margin": margin,
        "separable": margin,
        "verdict": ("活: 01漂移≫真绿(定点灯)" if margin
                    else "死: 无margin(真绿也漂移/机位抖动混淆)"),
        "notes": notes,
    }


def _hard_stop(fg_a, fg_b, a_verdict, b_verdict):
    """Jacob 硬停止规则: 两轴都不可分 → de-scope; 任一可分 → 出 #3 时序门控方案。"""
    a_sep = a_verdict.get("separable")
    b_sep = b_verdict.get("separable")
    if a_sep or b_sep:
        winner = []
        if a_sep:
            winner.append("A(绿run/红缝隙)")
        if b_sep:
            winner.append("B(绿斑漂移)")
        return {
            "verdict": "可分: 出 #3 时序门控方案走 plan-gate #5",
            "separable_axes": winner,
        }
    return {
        "verdict": "不可分: 触发硬停止规则 → de-scope 01 FP(同04FN), 治本并入 light-classifier-retrain 线",
        "separable_axes": [],
    }


if __name__ == "__main__":
    main()
