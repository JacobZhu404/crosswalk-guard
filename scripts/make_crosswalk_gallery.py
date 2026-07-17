#!/usr/bin/env python3
"""生成斑马线 y-band 标注画廊 (Part B1)。

读 datasets/gt/crosswalk/{video}.json 的关键帧 ts -> 用 CAP_PROP_POS_MSEC seek 帧
(与 eval_crosswalk_mask.py 同法, 保证标注帧=评测帧) -> 跑 CrosswalkDetector.detect
出预测带(青色参考) -> 建画廊(画布拖拽标 y0/y1)。

用法:
  python scripts/make_crosswalk_gallery.py                    # 全部已建骨架的视频
  python scripts/make_crosswalk_gallery.py --videos 违章11

标注闭环:
  1) 本脚本生成画廊到 data/output/crosswalk_eval/gallery.html
  2) python scripts/serve_gallery.py --eval-dir data/output/crosswalk_eval \
        --feedback-csv data/output/annotated/crosswalk_feedback.csv --port 8766
  3) 浏览器 http://localhost:8766 拖拽标注 -> 保存
  4) python scripts/apply_annotation_gt.py --kind crosswalk   # 反馈CSV -> 填骨架JSON
  5) python scripts/eval_crosswalk_mask.py                    # 出 band-IoU 基线
"""
import os
import sys
import json
import argparse

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.crosswalk import CrosswalkDetector
from redlight.evaluation.module_metrics import mask_band
from redlight.evaluation.crosswalk_gallery import CrosswalkMaskGalleryBuilder


def build_items(video, gt_frames, cfg):
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det = CrosswalkDetector(cfg)
    items = []
    for fr in gt_frames:
        ts = float(fr["ts"])
        cap.set(cv2.CAP_PROP_POS_MSEC, int(ts * 1000))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"    [跳过 {video}@{ts}] 取帧失败")
            continue
        mask = det.detect(frame)  # 不带 vehicle_boxes(与 eval 主指标一致)
        pb = mask_band(mask)  # (y0,y1) 或 None
        items.append({
            "t_sec": ts,
            "frame_idx": int(round(ts * fps)),
            "frame": frame,
            "pred_band": pb,
            "note": fr.get("note", ""),
        })
    cap.release()
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--gt-crosswalk", default=os.path.join(ROOT, "datasets", "gt", "crosswalk"))
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "output", "crosswalk_eval"))
    ap.add_argument("--feedback-csv",
                    default=os.path.join(ROOT, "data", "output", "annotated", "crosswalk_feedback.csv"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    videos = args.videos
    if not videos:
        if not os.path.isdir(args.gt_crosswalk):
            print("无 datasets/gt/crosswalk/ 骨架, 先跑 gen_gt_skeleton.py")
            return
        videos = [fn[:-5] for fn in sorted(os.listdir(args.gt_crosswalk)) if fn.endswith(".json")]

    video_items, gt_map = {}, {}
    for v in videos:
        path = os.path.join(args.gt_crosswalk, f"{v}.json")
        if not os.path.exists(path):
            print(f"  [跳过] 无骨架 {path}")
            continue
        with open(path, encoding="utf-8") as f:
            gt = json.load(f)
        frames = gt.get("frames", [])
        items = build_items(v, frames, cfg)
        if items:
            video_items[v] = items
            gt_map[v] = frames
            print(f"[{v}] {len(items)} 个关键帧")

    if not video_items:
        print("无可用关键帧")
        return

    builder = CrosswalkMaskGalleryBuilder(
        eval_dir=args.eval_dir, frames_dir=None,
        feedback_path=args.feedback_csv, max_crops=999,
    )
    out = builder.build(video_items, gt_map)
    print(f"\n画廊已生成: {out}")
    print(f"下一步: python scripts/serve_gallery.py --eval-dir {os.path.relpath(args.eval_dir, ROOT)} "
          f"--feedback-csv {os.path.relpath(args.feedback_csv, ROOT)} --port 8766")


if __name__ == "__main__":
    main()
