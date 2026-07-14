"""生成灯态误差确认画廊 (HTML) — 使用 LightGalleryBuilder。

用法:
  python scripts/make_light_gallery.py
  python scripts/make_light_gallery.py --videos 违章04
"""
import os
import sys
import csv
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.light_gallery import LightGalleryBuilder


def load_preds(pred_csv):
    # video 从文件名 pred_<video>.csv 取(确保 item 带 video, 画廊 annotate_crop 才能按视频查 prior)
    video = os.path.splitext(os.path.basename(pred_csv))[0][len("pred_"):]
    rows = []
    with open(pred_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "video": r.get("video", video) or video,
                "frame_idx": int(r["frame_idx"]),
                "t_sec": float(r["t_sec"]),
                "pred": r["pred"],
                "gt": r["gt"],
                "gt_conf": r.get("gt_conf", ""),
                "conf": float(r.get("conf", 0.0)),
            })
    return rows


def load_gt(gt_csv):
    g = {}
    with open(gt_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g.setdefault(r["video"], []).append(
                (float(r["start_s"]), float(r["end_s"]), r["state"],
                 r.get("confidence", "confirmed"))
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

    gt = load_gt(args.gt)

    pred_files = sorted(glob.glob(os.path.join(args.eval_dir, "pred_*.csv")))
    videos = args.videos or [os.path.basename(p)[5:-4] for p in pred_files]

    video_items = {}
    for v in videos:
        pred_csv = os.path.join(args.eval_dir, f"pred_{v}.csv")
        if not os.path.exists(pred_csv):
            continue
        items = load_preds(pred_csv)
        if items:
            video_items[v] = items

    builder = LightGalleryBuilder(
        eval_dir=args.eval_dir,
        frames_dir=args.frames_dir,
        feedback_path=args.feedback,
        max_crops=args.max_crops,
        priors_path=args.priors,
        config_path=os.path.join(ROOT, "configs", "config.yaml"),
    )
    out = builder.build(video_items, gt)
    print(f"[OK] 画廊 -> {out}")


if __name__ == "__main__":
    main()
