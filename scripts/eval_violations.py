#!/usr/bin/env python3
"""端到端违章事件评测 (eval-e2e): 跑整条流水线 -> 违章事件, 对比 datasets/gt/events.csv。

区别于 eval-b(只评②层灯态) / eval-plate(只评车牌): 本脚本评**最终违章结论**
(灯态×静止×压线×判定的乘积), 用事件级重叠匹配算 P/R/F1, 并报覆盖率(暴露欠检)。

用法:
  python scripts/eval_violations.py                      # 跑全部有 GT 违章的视频(fresh)
  python scripts/eval_violations.py --videos 违章04 违章09
  python scripts/eval_violations.py --reuse              # 复用已有 run_*/violations.csv, 不重跑
  python scripts/eval_violations.py --preset balanced --min-overlap 0.5
"""
import os
import sys
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.evaluation.violation_eval import (
    load_violation_gt, match_violation_events, aggregate,
)


def _read_violations_csv(path):
    """从已有 violations.csv 读回事件(用于 --reuse)。"""
    events = []
    if not os.path.exists(path):
        return events
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                events.append({
                    "status": (row.get("status") or "").strip(),
                    "start_ts": float(row["start_ts"]),
                    "end_ts": float(row["end_ts"]),
                    "plate": (row.get("plate") or "").strip(),
                    "track_id": row.get("track_id", ""),
                })
            except (KeyError, ValueError):
                continue
    return events


def _run_pipeline(cfg, video, preset, out_root):
    """跑一遍流水线, 返回事件 list。关标注视频加速(证据截图保留以回填车牌)。"""
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return None
    cfg.output.annotated_video = False   # eval 不需要 114MB 标注视频
    out_dir = os.path.join(out_root, f"run_{video}")
    return cli.run(cfg, video_path, out_dir, preset=preset)


def _confirmed(events):
    return [e for e in events if e.get("status") == "confirmed"]


def _fmt_span(x, key_s="start_ts", key_e="end_ts"):
    return f"[{x[key_s]:.1f}-{x[key_e]:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None, help="默认: events.csv 里所有含违章的视频")
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--events", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--min-overlap", type=float, default=0.5, help="判为TP的最小时间重叠(秒)")
    ap.add_argument("--reuse", action="store_true",
                    help="复用已有 data/output/run_<video>_<preset>/violations.csv, 不重跑流水线")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "output", "eval_violations"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)
    gt = load_violation_gt(args.events)

    videos = args.videos or sorted(gt.keys())
    print(f"=== 端到端违章评测 (eval-e2e): preset={args.preset} min_overlap={args.min_overlap}s ===")
    print(f"视频(含GT违章段): {videos}\n")

    results, rows = [], []
    for v in videos:
        gt_v = gt.get(v, [])
        # 取事件
        if args.reuse:
            csv_path = os.path.join(ROOT, "data", "output", f"run_{v}_{args.preset}", "violations.csv")
            events = _read_violations_csv(csv_path)
            if not events and not os.path.exists(csv_path):
                print(f"  [跳过 {v}] 无 {csv_path} (先不加 --reuse 跑一遍)")
                continue
        else:
            events = _run_pipeline(cfg, v, args.preset, args.out)
            if events is None:
                continue

        conf = _confirmed(events)
        r = match_violation_events(conf, gt_v, min_overlap_s=args.min_overlap)
        results.append(r)
        rows.append((v, r))

        # 逐视频时间线
        pred_line = " ".join(f"{_fmt_span(e)}{'✓' if any(m['pred_idx']==i for m in r['matches']) else '✗FP'}"
                             for i, e in enumerate(conf)) or "(无confirmed事件)"
        gt_line = " ".join(f"[{g['start_s']:.0f}-{g['end_s']:.0f}]cov={r['gt_coverage'][i]:.2f}"
                           + ("" if i not in r["fn_gts"] else "✗漏")
                           for i, g in enumerate(gt_v)) or "(无GT违章)"
        print(f"[{v}] P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f} "
              f"(tp={r['tp']} fp={r['fp']} fn={r['fn']}) 覆盖={r['mean_coverage']:.2f} "
              f"车牌={r['plate_hits']}/{r['plate_total']}")
        print(f"    预测confirmed: {pred_line}")
        print(f"    GT违章段:     {gt_line}")

    if not results:
        print("无评测结果")
        return

    agg = aggregate(results)
    print(f"\n=== 总体事件级: P={agg['precision']:.3f} R={agg['recall']:.3f} F1={agg['f1']:.3f} "
          f"(tp={agg['tp']} fp={agg['fp']} fn={agg['fn']}) ===")
    print(f"=== 命中违章段平均覆盖率={agg['mean_coverage']:.3f} "
          f"车牌命中={agg['plate_hits']}/{agg['plate_total']} ===")


if __name__ == "__main__":
    main()
