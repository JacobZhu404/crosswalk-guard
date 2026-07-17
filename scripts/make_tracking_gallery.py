#!/usr/bin/env python3
"""生成违章车 box 标注画廊 (Part B2)。

读 datasets/gt/tracking/{video}.json 的 anchors[].frames[].ts -> seek 帧 -> 建画廊(拖拽框车)。

用法:
  python scripts/make_tracking_gallery.py
  python scripts/make_tracking_gallery.py --videos 违章11

标注闭环:
  1) 本脚本 -> data/output/tracking_eval/gallery.html
  2) python scripts/serve_gallery.py --eval-dir data/output/tracking_eval \
        --feedback-csv data/output/annotated/tracking_feedback.csv --port 8767
  3) 浏览器 http://localhost:8767 拖拽框车 -> 保存
  4) python scripts/apply_annotation_gt.py --kind tracking
  5) python scripts/eval_tracking.py
"""
import os
import sys
import json
import argparse

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.tracking_gallery import TrackingGalleryBuilder


def build_items(video, skeleton):
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    items = []
    for anc in skeleton.get("anchors", []):
        win = anc.get("window", [])
        win_str = f"窗[{win[0]:.0f}-{win[1]:.0f}]" if len(win) == 2 else ""
        for fr in anc.get("frames", []):
            ts = float(fr["ts"])
            cap.set(cv2.CAP_PROP_POS_MSEC, int(ts * 1000))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"    [跳过 {video}@{ts}] 取帧失败")
                continue
            items.append({
                "t_sec": ts, "frame_idx": int(round(ts * fps)),
                "frame": frame, "note": win_str,
            })
    cap.release()
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--gt-tracking", default=os.path.join(ROOT, "datasets", "gt", "tracking"))
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "output", "tracking_eval"))
    ap.add_argument("--feedback-csv",
                    default=os.path.join(ROOT, "data", "output", "annotated", "tracking_feedback.csv"))
    args = ap.parse_args()

    videos = args.videos
    if not videos:
        if not os.path.isdir(args.gt_tracking):
            print("无 datasets/gt/tracking/ 骨架, 先跑 gen_gt_skeleton.py")
            return
        videos = [fn[:-5] for fn in sorted(os.listdir(args.gt_tracking)) if fn.endswith(".json")]

    video_items, gt_map = {}, {}
    for v in videos:
        path = os.path.join(args.gt_tracking, f"{v}.json")
        if not os.path.exists(path):
            print(f"  [跳过] 无骨架 {path}")
            continue
        with open(path, encoding="utf-8") as f:
            skel = json.load(f)
        items = build_items(v, skel)
        if items:
            video_items[v] = items
            gt_map[v] = skel
            print(f"[{v}] {len(items)} 个锚帧")

    if not video_items:
        print("无可用锚帧")
        return

    builder = TrackingGalleryBuilder(
        eval_dir=args.eval_dir, frames_dir=None,
        feedback_path=args.feedback_csv, max_crops=999,
    )
    out = builder.build(video_items, gt_map)
    print(f"\n画廊已生成: {out}")
    print(f"下一步: python scripts/serve_gallery.py --eval-dir {os.path.relpath(args.eval_dir, ROOT)} "
          f"--feedback-csv {os.path.relpath(args.feedback_csv, ROOT)} --port 8767")


if __name__ == "__main__":
    main()
