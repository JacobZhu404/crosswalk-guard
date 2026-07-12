"""视频抽帧预处理脚本: 将视频抽取为静态图像，保存到磁盘，生成索引文件。

这样可以先针对静态图像做好识别调优，再做动态推理，加快迭代速度。

用法:
    python scripts/extract_frames.py --video input_video/违章02.mp4 --fps 4
    python scripts/extract_frames.py --all --fps 4  (抽取所有视频)

输出结构:
    datasets/frames/
      违章02/
        frame_000001.jpg
        frame_000005.jpg
        ...
      违章03/
        frame_000001.jpg
        ...
      manifest.csv  (索引文件: video, frame_idx, timestamp, file_path, gt_plates)
"""
import os
import sys
import csv
import argparse
from pathlib import Path

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config


def parse_gt_plates():
    """解析GT文件，提取每个视频的车牌真值。"""
    gt = {}
    events_csv = os.path.join(ROOT, "datasets", "gt", "events.csv")
    with open(events_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row["video"]
            plates = row["violating_plates"] or ""
            other = row["other_plates"] or ""
            all_plates = []
            for p in plates.split(";") + other.split(";"):
                p = p.strip()
                if p and p != "?" and p != "无牌" and not p.startswith("["):
                    all_plates.append(p)
            if all_plates:
                gt[video] = list(set(all_plates))
    return gt


def extract_frames(video_path, output_dir, sample_fps=4, gt_plates=None):
    """抽取视频帧并保存到磁盘。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" ❌ 无法打开视频: {video_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    interval = max(1, int(round(src_fps / sample_fps)))
    num_samples = (total_frames // interval) + 1

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_out_dir, exist_ok=True)

    print(f"\n处理: {video_name}")
    print(f"  视频信息: {W}x{H} @ {src_fps:.1f}fps, 共 {total_frames} 帧")
    print(f"  采样参数: {sample_fps} fps, 间隔 {interval} 帧")
    print(f"  预计抽取: {num_samples} 帧")

    manifest_rows = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                frame_idx += 1
                continue

            timestamp = frame_idx / src_fps
            filename = f"frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(video_out_dir, filename)
            
            cv2.imencode('.jpg', frame)[1].tofile(filepath)
            
            manifest_rows.append({
                "video": video_name,
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 2),
                "file_path": filepath,
                "width": W,
                "height": H,
                "gt_plates": ",".join(gt_plates[video_name]) if gt_plates and video_name in gt_plates else "",
            })
            
            saved_count += 1
            if saved_count % 100 == 0:
                print(f"  进度: {saved_count}/{num_samples} ({saved_count/num_samples*100:.1f}%)")

        frame_idx += 1

    cap.release()
    print(f"  完成: 抽取 {saved_count} 帧")
    return manifest_rows


def main():
    ap = argparse.ArgumentParser(description="视频抽帧预处理")
    ap.add_argument("--video", help="单个视频路径")
    ap.add_argument("--all", action="store_true", help="处理所有视频")
    ap.add_argument("--fps", type=int, default=4, help="采样帧率")
    ap.add_argument("--output", default=None, help="输出目录")
    args = ap.parse_args()

    if not args.video and not args.all:
        print("请指定 --video 或 --all")
        return

    if args.output is None:
        args.output = os.path.join(ROOT, "datasets", "frames")
    os.makedirs(args.output, exist_ok=True)

    gt_plates = parse_gt_plates()
    manifest_path = os.path.join(args.output, "manifest.csv")

    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"视频不存在: {video_path}")
            return
        rows = extract_frames(video_path, args.output, args.fps, gt_plates)
        
        with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video", "frame_idx", "timestamp", 
                                              "file_path", "width", "height", "gt_plates"])
            w.writeheader()
            w.writerows(rows)

        print(f"\n索引文件已保存: {manifest_path}")
        return

    video_dir = os.path.join(ROOT, "input_video")
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

    print(f"视频目录: {video_dir}")
    print(f"待处理视频: {len(video_files)} 个")
    print(f"采样帧率: {args.fps} fps")
    print(f"输出目录: {args.output}")

    all_rows = []
    for vf in video_files:
        video_path = os.path.join(video_dir, vf)
        rows = extract_frames(video_path, args.output, args.fps, gt_plates)
        all_rows.extend(rows)

    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame_idx", "timestamp", 
                                          "file_path", "width", "height", "gt_plates"])
        w.writeheader()
        w.writerows(all_rows)

    total_frames = len(all_rows)
    total_size_mb = sum(os.path.getsize(r["file_path"]) for r in all_rows) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"抽帧完成!")
    print(f"{'='*60}")
    print(f"总视频数: {len(video_files)}")
    print(f"总帧数: {total_frames}")
    print(f"总大小: {total_size_mb:.1f} MB")
    print(f"索引文件: {manifest_path}")

    for video_name in gt_plates:
        count = sum(1 for r in all_rows if r["video"] == video_name)
        print(f"  {video_name}: {count} 帧, GT车牌: {gt_plates[video_name]}")


if __name__ == "__main__":
    main()
