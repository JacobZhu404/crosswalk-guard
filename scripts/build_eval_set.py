"""评测集构建脚本: 提取典型帧建立评测集，用于回归测试。

评测集结构:
    datasets/plate_eval_set/
    ├── meta.csv                    # 评测集索引
    ├── correct/                    # 正确识别的帧
    │   ├── 违章02_107.25s_京LNE560.jpg
    │   └── ...
    ├── incorrect/                  # 错误识别的帧
    │   ├── 违章02_47.14s_京A14672_实际_粤A14672.jpg
    │   └── ...
    └── missed/                     # 未识别的帧
        ├── 违章05_10.00s_京ADH9206.jpg
        └── ...

用法:
    python scripts/build_eval_set.py --video 违章02
    python scripts/build_eval_set.py --all
"""
import os
import sys
import argparse
import csv

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer


def extract_key_frames(video_name, gt_plates, output_dir, fps=4):
    video_path = os.path.join(ROOT, "input_video", f"{video_name}.mp4")
    if not os.path.exists(video_path):
        print(f"视频不存在: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(src_fps / fps)))

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    plate = PlateRecognizer(cfg, verbose=False)

    frame_idx = 0
    results = []

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % interval == 0:
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
                xyxy = p.get("xyxy", [0, 0, 0, 0])
                
                matched = False
                for gt in gt_plates:
                    if text == gt:
                        matched = True
                        break
                
                status = "correct" if matched else "incorrect"
                actual_plate = text if status == "correct" else "|".join(gt_plates)
                
                img_name = f"{frame_idx:06d}_{status}.jpg"
                img_path = os.path.join(output_dir, img_name)
                
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, 
                             (int(xyxy[0]), int(xyxy[1])), 
                             (int(xyxy[2]), int(xyxy[3])), 
                             (0, 255, 0) if status == "correct" else (0, 0, 255), 2)
                cv2.putText(frame_copy, f"{text} ({conf:.2f})", 
                           (int(xyxy[0]), int(xyxy[1])-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if status == "correct" else (0, 0, 255), 2)
                cv2.putText(frame_copy, f"GT: {','.join(gt_plates)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imwrite(img_path, frame_copy)

                results.append({
                    "video": video_name,
                    "frame_idx": frame_idx,
                    "time": ts,
                    "detected": text,
                    "confidence": conf,
                    "gt_plates": ",".join(gt_plates),
                    "status": status,
                    "image_path": img_path,
                })

        frame_idx += 1

    cap.release()
    return results


def extract_missed_frames(video_name, gt_plates, output_dir, fps=4):
    video_path = os.path.join(ROOT, "input_video", f"{video_name}.mp4")
    if not os.path.exists(video_path):
        print(f"视频不存在: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(src_fps / fps)))

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    plate = PlateRecognizer(cfg, verbose=False)

    frame_idx = 0
    results = []
    missed_timestamps = []

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                frame_idx += 1
                continue

            ts = frame_idx / src_fps
            plates = plate.detect(frame)

            detected_plates = set(p["text"] for p in plates if p.get("text"))
            missed_gt = [gt for gt in gt_plates if gt not in detected_plates]

            if missed_gt:
                missed_timestamps.append(ts)
                
                if len(missed_timestamps) <= 20:
                    img_name = f"{frame_idx:06d}_missed.jpg"
                    img_path = os.path.join(output_dir, img_name)
                    
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    
                    frame_copy = frame.copy()
                    cv2.putText(frame_copy, f"Missed: {','.join(missed_gt)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"Detected: {','.join(detected_plates)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.imwrite(img_path, frame_copy)

                    results.append({
                        "video": video_name,
                        "frame_idx": frame_idx,
                        "time": ts,
                        "detected": ",".join(detected_plates),
                        "confidence": 0,
                        "gt_plates": ",".join(missed_gt),
                        "status": "missed",
                        "image_path": img_path,
                    })

        frame_idx += 1

    cap.release()
    return results


def main():
    ap = argparse.ArgumentParser(description="构建车牌识别评测集")
    ap.add_argument("--video", help="视频名称(不含.mp4)")
    ap.add_argument("--all", action="store_true", help="处理所有视频")
    ap.add_argument("--fps", type=int, default=4, help="采样帧率")
    args = ap.parse_args()

    eval_set_dir = os.path.join(ROOT, "datasets", "plate_eval_set")
    os.makedirs(eval_set_dir, exist_ok=True)

    videos = {}
    gt_path = os.path.join(ROOT, "datasets", "gt", "events.csv")
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video = row["video"]
                plates = []
                for field in ["violating_plates", "other_plates"]:
                    if row.get(field):
                        for p in row[field].split(";"):
                            p = p.strip()
                            if p and p != "?" and p != "无牌":
                                plates.append(p)
                if plates:
                    videos[video] = list(set(plates))

    if args.video:
        videos = {k: v for k, v in videos.items() if k == args.video}
    elif not args.all:
        print("请指定 --video 或 --all")
        return

    all_results = []
    for video_name, gt_plates in videos.items():
        print(f"处理视频: {video_name}")
        print(f"  GT车牌: {gt_plates}")
        
        correct_results = extract_key_frames(video_name, gt_plates, eval_set_dir, args.fps)
        missed_results = extract_missed_frames(video_name, gt_plates, eval_set_dir, args.fps)
        
        all_results.extend(correct_results)
        all_results.extend(missed_results)
        
        print(f"  正确识别: {len([r for r in correct_results if r['status'] == 'correct'])}")
        print(f"  错误识别: {len([r for r in correct_results if r['status'] == 'incorrect'])}")
        print(f"  未识别: {len(missed_results)}")

    meta_path = os.path.join(eval_set_dir, "meta.csv")
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "frame_idx", "time", "detected", 
                                              "confidence", "gt_plates", "status", "image_path"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n评测集构建完成!")
    print(f"  总样本数: {len(all_results)}")
    print(f"  索引文件: {meta_path}")
    print(f"  正确识别: {len([r for r in all_results if r['status'] == 'correct'])}")
    print(f"  错误识别: {len([r for r in all_results if r['status'] == 'incorrect'])}")
    print(f"  未识别: {len([r for r in all_results if r['status'] == 'missed'])}")


if __name__ == "__main__":
    main()
