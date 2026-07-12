"""标注确认工具: 丢出有疑问的识别结果，让用户逐一确认。

用法:
    python scripts/confirm_plate_labels.py --video 违章05
    python scripts/confirm_plate_labels.py --all --output annotations.csv
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


def analyze_video(video_name, gt_plates, output_dir, fps=4):
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
    candidates = []

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

                if not matched:
                    img_name = f"{video_name}_{ts:.2f}s_{text}_conf{conf:.2f}.jpg"
                    img_path = os.path.join(output_dir, img_name)
                    
                    os.makedirs(output_dir, exist_ok=True)
                    
                    frame_copy = frame.copy()
                    cv2.rectangle(frame_copy, 
                                 (int(xyxy[0]), int(xyxy[1])), 
                                 (int(xyxy[2]), int(xyxy[3])), 
                                 (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"{text} ({conf:.2f})", 
                               (int(xyxy[0]), int(xyxy[1])-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"GT: {','.join(gt_plates)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame_copy, f"Time: {ts:.2f}s", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.imwrite(img_path, frame_copy)

                    candidates.append({
                        "video": video_name,
                        "frame_idx": frame_idx,
                        "time": ts,
                        "detected": text,
                        "confidence": conf,
                        "gt_plates": ",".join(gt_plates),
                        "image_path": img_path,
                        "confirmed_label": "",
                        "is_correct": "",
                        "notes": "",
                    })

        frame_idx += 1

    cap.release()
    return candidates


def main():
    ap = argparse.ArgumentParser(description="标注确认工具")
    ap.add_argument("--video", help="视频名称(不含.mp4)")
    ap.add_argument("--all", action="store_true", help="处理所有视频")
    ap.add_argument("--fps", type=int, default=4, help="采样帧率")
    ap.add_argument("--output", default="plate_annotations.csv", help="输出文件")
    args = ap.parse_args()

    output_dir = os.path.join(ROOT, "data", "output", "plate_confirmation")
    os.makedirs(output_dir, exist_ok=True)

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

    all_candidates = []
    for video_name, gt_plates in videos.items():
        print(f"处理视频: {video_name}")
        print(f"  GT车牌: {gt_plates}")
        
        candidates = analyze_video(video_name, gt_plates, output_dir, args.fps)
        all_candidates.extend(candidates)
        
        print(f"  待确认样本: {len(candidates)}")

    output_path = os.path.join(output_dir, args.output)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "frame_idx", "time", "detected", 
                                              "confidence", "gt_plates", "image_path",
                                              "confirmed_label", "is_correct", "notes"])
        writer.writeheader()
        writer.writerows(all_candidates)

    print(f"\n标注确认文件已生成:")
    print(f"  文件路径: {output_path}")
    print(f"  待确认样本: {len(all_candidates)}")
    print(f"  截图目录: {output_dir}")
    print("\n请打开CSV文件，填写以下字段:")
    print("  confirmed_label: 真实车牌（如果识别错误）")
    print("  is_correct: yes/no/partial（是否正确）")
    print("  notes: 备注（如遮挡、角度问题、难度太大等）")


if __name__ == "__main__":
    main()
