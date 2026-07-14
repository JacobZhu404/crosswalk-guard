"""生成车牌识别误差确认画廊 (HTML) — 使用 PlateGalleryBuilder。

用法:
  python scripts/make_plate_gallery.py
  python scripts/make_plate_gallery.py --videos 违章02
"""
import os
import sys
import csv
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer
from redlight.evaluation.plate_gallery import PlateGalleryBuilder


def load_gt(gt_csv):
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
        print(f"警告: 无法读取GT文件 {gt_csv}, 使用命令行指定的GT车牌")
    return g


def analyze_video(video_name, gt_plates, max_crops=10):
    """实时检测视频, 按优先级返回关键帧 items。"""
    video_path = os.path.join(ROOT, "input_video", f"{video_name}.mp4")
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    plate = PlateRecognizer(cfg, verbose=False)

    candidates = []
    seen_plates = set()
    frame_idx = 0

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % 8 == 0:
            ret, frame = cap.retrieve()
            if not ret:
                frame_idx += 1
                continue

            ts = frame_idx / src_fps
            plates = plate.detect(frame)

            for p in plates:
                if not p.get("text"):
                    continue

                text = p["text"]
                conf = p.get("conf", 0.0)
                matched = any(text == gt for gt in gt_plates)

                priority = 0
                if not matched:
                    priority += 3
                if conf < 0.6:
                    priority += 2
                if text not in seen_plates:
                    priority += 1
                    seen_plates.add(text)

                candidates.append({
                    "frame_idx": frame_idx,
                    "t_sec": ts,
                    "detected": text,
                    "conf": conf,
                    "plate_info": p,
                    "all_plates": plates,
                    "frame": frame.copy(),
                    "priority": priority,
                    "matched": matched,
                })

        frame_idx += 1

    cap.release()

    candidates.sort(key=lambda x: (-x["priority"], x["t_sec"]))
    return candidates[:max_crops]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "output", "plate_eval"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--gt-plates", nargs="*", default=None, help="GT车牌列表(如: 京LNE560 京ADH9206)")
    ap.add_argument("--feedback", default=os.path.join(ROOT, "data", "output", "annotated", "plate_feedback.csv"))
    ap.add_argument("--max-crops", type=int, default=10)
    args = ap.parse_args()

    gt = load_gt(args.gt)

    if args.gt_plates and args.videos:
        for v in args.videos:
            gt[v] = args.gt_plates

    videos = args.videos or list(gt.keys())

    video_items = {}
    for v in videos:
        gt_plates = gt.get(v, [])
        items = analyze_video(v, gt_plates, max_crops=args.max_crops)
        if items:
            # 按时间排序, 便于画廊展示
            items.sort(key=lambda x: x["t_sec"])
            video_items[v] = items

    builder = PlateGalleryBuilder(
        eval_dir=args.eval_dir,
        feedback_path=args.feedback,
        max_crops=args.max_crops,
    )
    out = builder.build(video_items, gt)
    print(f"[OK] 画廊 -> {out}")


if __name__ == "__main__":
    main()
