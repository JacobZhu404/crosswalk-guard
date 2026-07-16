#!/usr/bin/env python3
"""Part B2 — 跟踪评测: 违章车 ID 碎片化 + 静止判定准确率(不做全 MOTA)。

锚定: 用 datasets/gt/tracking/{video}.json 的违章窗 + 锚帧框(Jacob 标)。
取数: cli.run(return_track_samples=True) 返回引擎已累积的逐帧 track 样本(加法形参, 零行为改动)。
归属(写死, cc 裁定): 窗内任一帧中框与任一锚框 IoU>=T 的所有 track_id 之**并集**(理想=1);
                     碎片化数 = 并集大小。T 默认 0.5。

用法:
  python scripts/eval_tracking.py                          # 跑全部已标视频
  python scripts/eval_tracking.py --videos 违章02           # 单视频
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.evaluation.module_metrics import (
    attribution_union, stationary_accuracy, iou_box,
)


def eval_video(video, gt_anchors, cfg, preset, T=0.5):
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return None
    out_dir = os.path.join(ROOT, "data", "output", f"run_{video}_{preset}")
    events, track_samples = cli.run(cfg, video_path, out_dir, preset=preset,
                                     return_track_samples=True)
    per_anchor = []
    for a in gt_anchors:
        window = tuple(a["window"])
        anchor_boxes = [fr["box"] for fr in a["frames"] if fr.get("box")]
        if not anchor_boxes:
            print(f"    [跳过 {video} 窗{window}] 锚框未标注")
            continue
        attr = attribution_union(track_samples, anchor_boxes, window, T=T)
        stat_acc = stationary_accuracy(track_samples, attr, window)
        per_anchor.append({
            "window": list(window), "track_ids": sorted(attr),
            "frag_count": len(attr), "stationary_acc": round(stat_acc, 3),
        })
    if not per_anchor:
        return None
    return {"video": video, "anchors": per_anchor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--T", type=float, default=0.5, help="归属 IoU 阈值(默认 0.5)")
    ap.add_argument("--gt-tracking", default=os.path.join(ROOT, "datasets", "gt", "tracking"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    files = []
    if args.videos:
        files = [(v, os.path.join(args.gt_tracking, f"{v}.json")) for v in args.videos]
    else:
        if not os.path.isdir(args.gt_tracking):
            print("无 datasets/gt/tracking/ 目录, 先跑 gen_gt_skeleton.py 并由 Jacob 标")
            return
        for fn in sorted(os.listdir(args.gt_tracking)):
            if fn.endswith(".json"):
                files.append((fn[:-5], os.path.join(args.gt_tracking, fn)))

    print(f"=== B2 跟踪评测(preset={args.preset}, T={args.T}) ===\n")
    for video, path in files:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            gt = json.load(f)
        r = eval_video(video, gt.get("anchors", []), cfg, args.preset, T=args.T)
        if r is None:
            continue
        print(f"[{video}]")
        for a in r["anchors"]:
            flag = "⚠️碎片化" if a["frag_count"] > 1 else "✓"
            print(f"    窗{a['window']} {flag} track_ids={a['track_ids']} "
                  f"碎片化数={a['frag_count']} 静止准确率={a['stationary_acc']:.3f}")


if __name__ == "__main__":
    main()
