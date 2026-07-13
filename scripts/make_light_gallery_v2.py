"""生成灯态误差确认画廊 (HTML) —— 基于 BaseGalleryBuilder 的轻量包装。

用法:
  python scripts/make_light_gallery_v2.py
  python scripts/make_light_gallery_v2.py --videos 违章04
"""
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import argparse

from redlight.evaluation import LightGalleryBuilder


def load_pred_csv(pred_csv: str) -> list:
    rows = []
    with open(pred_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_gt_csv(gt_csv: str) -> dict:
    g = {}
    with open(gt_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g.setdefault(r["video"], []).append(
                (float(r["start_s"]), float(r["end_s"]), r["state"], r.get("confidence", "confirmed"))
            )
    for v in g:
        g[v].sort(key=lambda x: x[0])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "output", "light_eval"))
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--priors", default=os.path.join(ROOT, "configs", "light_priors.json"))
    ap.add_argument("--feedback", default=os.path.join(ROOT, "data", "output", "annotated", "light_feedback.csv"))
    ap.add_argument("--max-crops", type=int, default=10)
    args = ap.parse_args()

    gt = load_gt_csv(args.gt)
    pred_files = sorted(glob.glob(os.path.join(args.eval_dir, "pred_*.csv")))
    videos = args.videos or [os.path.basename(p)[5:-4] for p in pred_files]

    video_items = {}
    for v in videos:
        pred_csv = os.path.join(args.eval_dir, f"pred_{v}.csv")
        if not os.path.exists(pred_csv):
            continue
        preds = load_pred_csv(pred_csv)
        if not preds:
            continue
        # 标记 video 到每条记录, 方便 builder 使用
        for p in preds:
            p["video"] = v
        video_items[v] = preds

    config_path = os.path.join(ROOT, "configs", "config.yaml")
    builder = LightGalleryBuilder(
        eval_dir=args.eval_dir,
        frames_dir=args.frames_dir,
        feedback_path=args.feedback,
        max_crops=args.max_crops,
        priors_path=args.priors,
        config_path=config_path,
    )

    out_path = builder.build(video_items, gt)
    print(f"[OK] 画廊 -> {out_path}")


if __name__ == "__main__":
    main()
