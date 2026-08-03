#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sweep_box_overlap.py — box_overlap 阈值扫描(9正+2负, v2+denom=box)

对每个 box_overlap 值, 跑全 11 视频(9正例+2负例01/10), 量 F1 + 负例 FP。
选最大化 F1 且满足: 负例 01/10 FP=0、7 好视频 TP 不降。

用法: PYTHONPATH=src ./.venv/bin/python scripts/sweep_box_overlap.py
"""
import os, sys, json, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.pipeline.tracker import SENSITIVITY_PRESETS
from redlight.models.crosswalk_v2 import CrosswalkDetectorV2
from redlight.evaluation.violation_eval import (
    load_violation_gt, match_violation_events, load_video_metadata, aggregate,
)
from redlight.evaluation.gt_lookup import load_light_segments

GT_EVENTS = ROOT / "datasets" / "gt" / "events.csv"
VIDEOS_CSV = ROOT / "datasets" / "gt" / "videos.csv"
OUT_DIR = ROOT / "data" / "output" / "qw"
REPORT = ROOT / "docs" / "reports" / "2026-08-03-qw-box-overlap-sweep.md"

SWEEP_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30]
PRESET = "balanced"


def run_sweep(cfg, videos, gt_violations, video_meta, box_overlap_val):
    """用指定 box_overlap 跑全 11 视频, 返回 per-video events。"""
    # 临时修改 SENSITIVITY_PRESETS 的 box_overlap
    original = SENSITIVITY_PRESETS[PRESET].get("box_overlap")
    SENSITIVITY_PRESETS[PRESET]["box_overlap"] = box_overlap_val

    results = {}
    for video in videos:
        video_path = str(ROOT / "input_video" / f"{video}.mp4")
        if not os.path.isfile(video_path):
            print(f"  [跳过] {video}")
            continue
        out_dir = str(OUT_DIR / f"sweep_{video}_bo{box_overlap_val}")
        det = CrosswalkDetectorV2(cfg)
        events = cli.run(cfg, video_path, out_dir, preset=PRESET,
                         crosswalk_detector=det, occ_denom="box")
        confirmed = [e for e in events if e.get("status") == "confirmed"]
        results[video] = confirmed
        print(f"  [{video}] confirmed={len(confirmed)}", flush=True)

    # 恢复原值
    if original is not None:
        SENSITIVITY_PRESETS[PRESET]["box_overlap"] = original
    return results


def main():
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    # 禁用标注视频输出(加速)
    cfg.output.annotated_video = False

    gt_violations = load_violation_gt(str(GT_EVENTS))
    video_meta = load_video_metadata(str(VIDEOS_CSV))
    all_videos = sorted(video_meta.keys())

    print(f"视频: {all_videos}")
    print(f"扫阈值: {SWEEP_VALUES}")
    print(f"检测器: v2(时序聚合) + denom=box")
    print()

    sweep_results = {}
    for bo in SWEEP_VALUES:
        print(f"\n=== box_overlap={bo} ===")
        results = run_sweep(cfg, all_videos, gt_violations, video_meta, bo)
        sweep_results[bo] = results

        # 聚合 F1
        all_tp, all_fp, all_fn = 0, 0, 0
        neg_fp = {}
        for v in all_videos:
            preds = results.get(v, [])
            gts = gt_violations.get(v, [])
            is_neg = not video_meta.get(v, {}).get("has_violation", False)
            m = match_violation_events(preds, gts, min_overlap_s=0.5)
            tp = len(m["matches"])
            fp = m["fp"]
            fn = m["fn"]
            all_tp += tp
            all_fp += fp
            all_fn += fn
            if is_neg:
                neg_fp[v] = fp
        prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  TP={all_tp} FP={all_fp} FN={all_fn} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        print(f"  负例 FP: {neg_fp}")

    # 选最优
    print(f"\n=== 汇总 ===")
    print(f"{'box_overlap':>12} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6} {'neg_FP':>7}")
    best_bo, best_f1 = None, 0
    for bo in SWEEP_VALUES:
        all_tp, all_fp, all_fn = 0, 0, 0
        neg_fp_total = 0
        for v in all_videos:
            preds = sweep_results[bo].get(v, [])
            gts = gt_violations.get(v, [])
            is_neg = not video_meta.get(v, {}).get("has_violation", False)
            m = match_violation_events(preds, gts, min_overlap_s=0.5)
            all_tp += len(m["matches"])
            all_fp += m["fp"]
            all_fn += m["fn"]
            if is_neg:
                neg_fp_total += m["fp"]
        prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{bo:>12.2f} {all_tp:>4} {all_fp:>4} {all_fn:>4} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {neg_fp_total:>7}")
        # gate: 负例 FP=0
        if neg_fp_total == 0 and f1 > best_f1:
            best_f1 = f1
            best_bo = bo

    print(f"\n最优(box_overlap={best_bo}, F1={best_f1:.3f}, 负例FP=0)")

    # 写报告
    _write_report(sweep_results, all_videos, gt_violations, video_meta, best_bo, best_f1)


def _write_report(sweep_results, all_videos, gt_violations, video_meta, best_bo, best_f1):
    L = [
        "# box_overlap 阈值扫描报告(qw)\n",
        f"> 检测器: v2(时序聚合) + denom=box, preset={PRESET}\n",
        f"> 扫描值: {SWEEP_VALUES}\n",
        f"> 最优: box_overlap={best_bo}, F1={best_f1:.3f}\n\n",
        "## 汇总\n",
        "| box_overlap | TP | FP | FN | P | R | F1 | 负例FP |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for bo in SWEEP_VALUES:
        all_tp, all_fp, all_fn = 0, 0, 0
        neg_fp_total = 0
        for v in all_videos:
            preds = sweep_results[bo].get(v, [])
            gts = gt_violations.get(v, [])
            is_neg = not video_meta.get(v, {}).get("has_violation", False)
            m = match_violation_events(preds, gts, min_overlap_s=0.5)
            all_tp += len(m["matches"])
            all_fp += m["fp"]
            all_fn += m["fn"]
            if is_neg:
                neg_fp_total += m["fp"]
        prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        marker = " **←最优**" if bo == best_bo else ""
        L.append(f"| {bo:.2f} | {all_tp} | {all_fp} | {all_fn} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {neg_fp_total} |{marker}")

    L.append(f"\n## 逐视频(最优 box_overlap={best_bo})\n")
    L.append("| 视频 | has_violation | confirmed | TP | FP | FN |")
    L.append("|---|---|---|---|---|---|")
    for v in all_videos:
        preds = sweep_results[best_bo].get(v, [])
        gts = gt_violations.get(v, [])
        is_viol = video_meta.get(v, {}).get("has_violation", False)
        m = match_violation_events(preds, gts, min_overlap_s=0.5)
        L.append(f"| {v} | {is_viol} | {len(preds)} | {len(m['matches'])} | {m['fp']} | {m['fn']} |")

    L.append(f"\n## 方法学\n")
    L.append("- 临时修改 `SENSITIVITY_PRESETS[preset]['box_overlap']` 后调 `cli.run`。\n")
    L.append("- v2 检测器每视频新建(时序聚合天然跑), `occ_denom='box'`。\n")
    L.append("- 禁用标注视频输出(加速), 单次非多 seed。\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {REPORT}")


if __name__ == "__main__":
    main()
