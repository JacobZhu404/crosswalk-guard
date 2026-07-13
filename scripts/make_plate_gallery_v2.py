"""生成车牌识别误差确认画廊 (HTML) —— 基于 BaseGalleryBuilder 的轻量包装。

用法:
  python scripts/make_plate_gallery_v2.py
  python scripts/make_plate_gallery_v2.py --videos 违章02
"""
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import argparse

from redlight.evaluation import PlateGalleryBuilder


def load_gt_csv(gt_csv: str) -> dict:
    g = {}
    try:
        with open(gt_csv, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                video = r["video"]
                plates = []
                for field in ["violating_plates", "other_plates"]:
                    if r.get(field):
                        for p in r[field].split(";"):
                            p = p.strip()
                            if p and p != "?" and p != "无牌":
                                plates.append(p)
                if plates:
                    g[video] = list(set(plates))
    except (PermissionError, FileNotFoundError):
        print(f"警告: 无法读取GT文件 {gt_csv}")
    return g


def load_pred_csv(pred_csv: str, video: str) -> list:
    rows = []
    with open(pred_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["video"] = video
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "output", "plate_eval"))
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--gt-plates", nargs="*", default=None, help="GT车牌列表(如: 京LNE560 京ADH9206)")
    ap.add_argument("--feedback", default=os.path.join(ROOT, "data", "output", "annotated", "plate_feedback.csv"))
    ap.add_argument("--max-crops", type=int, default=10)
    args = ap.parse_args()

    gt = load_gt_csv(args.gt)

    if args.gt_plates and args.videos:
        for v in args.videos:
            gt[v] = args.gt_plates

    videos = args.videos or list(gt.keys())

    # 优先读取预计算 pred CSV; 若无则跳过(未来可接入 VideoSampler 实时检测)
    video_items = {}
    for v in videos:
        pred_csv = os.path.join(args.eval_dir, f"pred_{v}.csv")
        if os.path.exists(pred_csv):
            preds = load_pred_csv(pred_csv, v)
            if preds:
                video_items[v] = preds
        else:
            print(f"[skip] {v}: 无 pred CSV, 跳过(实时检测模式待接入)")

    if not video_items:
        print("[warn] 无可用预测数据, 画廊为空。请先生成 pred_{video}.csv 或接入实时检测。")
        return

    builder = PlateGalleryBuilder(
        eval_dir=args.eval_dir,
        frames_dir=args.frames_dir,
        feedback_path=args.feedback,
        max_crops=args.max_crops,
    )

    out_path = builder.build(video_items, gt)
    print(f"[OK] 画廊 -> {out_path}")


if __name__ == "__main__":
    main()
