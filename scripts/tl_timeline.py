#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tl_timeline.py —— 帧序列快速调参工具 (配合 datasets/frames 抽帧数据)

用法:
  python scripts/tl_timeline.py --video 违章02
  python scripts/tl_timeline.py --video 违章03 --prior 0.82 0.12
  python scripts/tl_timeline.py --video 违章04 --prior 0.55 0.10 --every 2

说明:
  - 从 datasets/frames/{video}/ 读 jpg 序列, 用 manifest.csv 的 timestamp 对齐时间
  - 跑 TrafficLightDetector (可选 --prior 行人灯位置先验), 打印逐帧 (t, state, anchor_cx, anchor_cy)
  - 用 cv2.imdecode 读图, 规避 GBK 中文路径问题
  - 比视频解码快 10x+, 适合 prior 快速迭代
  - 生产数据仍用 draw_light_boxes.py 跑原视频 (保证跟踪连续+分辨率)
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redlight.models.traffic_light import TrafficLightDetector
from redlight.infrastructure.config import load_config


def robust_imread(path):
    """用字节流+imdecode 读图, 规避中文/GBK 路径编码问题。"""
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[WARN] 读图失败 {path}: {e}", file=sys.stderr)
        return None


def load_manifest(manifest_path, video_name):
    rows = []
    enc = "utf-8-sig"
    try:
        with open(manifest_path, encoding=enc) as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                if row.get("video", "").strip() == video_name:
                    try:
                        ts = float(row["timestamp"])
                    except (KeyError, ValueError):
                        ts = 0.0
                    rows.append((ts, row.get("file_path", "").strip()))
    except FileNotFoundError:
        print(f"[WARN] manifest 未找到: {manifest_path}", file=sys.stderr)
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="视频名(不含扩展名), 对应 datasets/frames/{video}/")
    ap.add_argument("--frames-dir", default="datasets/frames", help="抽帧根目录")
    ap.add_argument("--manifest", default="datasets/frames/manifest.csv")
    ap.add_argument("--prior", nargs=2, type=float, metavar=("CX", "CY"), default=None,
                    help="行人灯位置先验(归一化)")
    ap.add_argument("--roi", type=int, default=None, help="HSV直采ROI边长px(默认160, 小灯视频加大)")
    ap.add_argument("--every", type=int, default=1, help="每隔 N 帧采样一次(1=全采)")
    ap.add_argument("--max-sec", type=float, default=None, help="只跑到该秒数")
    ap.add_argument("--no-color", action="store_true", help="打印紧凑单行")
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    det = TrafficLightDetector(cfg, verbose=False)
    if args.prior:
        det.signal_prior = (args.prior[0], args.prior[1])
    if args.roi:
        det.prior_roi_px = args.roi
    if args.prior or args.roi:
        print(f"[INFO] 先验 signal_prior={det.signal_prior}, "
              f"prior_search_radius={det.prior_search_radius}, prior_roi_px={det.prior_roi_px}")

    frame_dir = os.path.join(args.frames_dir, args.video)
    if not os.path.isdir(frame_dir):
        print(f"[ERROR] 抽帧目录不存在: {frame_dir}", file=sys.stderr)
        sys.exit(1)

    manifest_rows = load_manifest(args.manifest, args.video)
    if manifest_rows:
        samples = manifest_rows
    else:
        # 回退: 直接列目录 jpg
        fs = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")])
        samples = [(i, os.path.join(frame_dir, f)) for i, f in enumerate(fs)]

    print(f"# 视频 {args.video}: {len(samples)} 帧 (来自抽帧)")
    print(f"# {'t(s)':>8}  {'state':<8}  {'cx':>6}  {'cy':>6}  {'g':>4} {'r':>4}")
    cnt = 0
    for idx, (ts, fpath) in enumerate(samples):
        if args.max_sec is not None and ts > args.max_sec:
            break
        if idx % max(1, args.every) != 0:
            continue
        img = robust_imread(fpath)
        if img is None:
            continue
        res = det.detect(img)
        a = det.anchor
        cx = round(a["cx"], 3) if a else None
        cy = round(a["cy"], 3) if a else None
        print(f"{ts:>8.2f}  {res['state']:<8}  {str(cx):>6}  {str(cy):>6}  "
              f"{res.get('g_px',0):>4} {res.get('r_px',0):>4}")
        cnt += 1
    print(f"# 共输出 {cnt} 行时间线")


if __name__ == "__main__":
    main()
