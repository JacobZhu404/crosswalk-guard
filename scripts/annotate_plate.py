"""车牌标注可视化脚本: 在视频中绘制车牌框和识别内容，输出带标注视频和关键帧截图。

用法:
    python scripts/annotate_plate.py <video.mp4> [--output_dir <dir>] [--sample_fps 4]
输出:
    - annotated.mp4: 带车牌标注的视频
    - key_frames/: 识别到车牌的关键帧截图
    - plate_detections.csv: 所有车牌检测结果
"""
import os
import sys
import csv
import argparse

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer


def draw_plate_annotation(frame, plates):
    """在帧上绘制车牌标注。"""
    disp = frame.copy()
    H, W = frame.shape[:2]
    
    for p in plates:
        x1, y1, x2, y2 = [int(v) for v in p["xyxy"]]
        text = p.get("text", "")
        conf = p.get("conf", 0.0)
        color = p.get("color", "blue")
        
        if color == "blue":
            box_color = (255, 150, 0)
            text_color = (255, 255, 255)
            bg_color = (255, 150, 0)
        elif color == "yellow":
            box_color = (0, 200, 255)
            text_color = (0, 0, 0)
            bg_color = (0, 200, 255)
        elif color == "green":
            box_color = (0, 200, 100)
            text_color = (0, 0, 0)
            bg_color = (0, 200, 100)
        else:
            box_color = (200, 200, 200)
            text_color = (0, 0, 0)
            bg_color = (200, 200, 200)
        
        cv2.rectangle(disp, (x1, y1), (x2, y2), box_color, 3)
        
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size
            
            label_y = max(y1 - 10, text_h + 4)
            label_x = x1
            
            bg_w = text_w + 8
            bg_h = text_h + 8
            bg_x1 = label_x - 4
            bg_y1 = label_y - text_h - 4
            
            cv2.rectangle(disp, (bg_x1, bg_y1), (bg_x1 + bg_w, bg_y1 + bg_h), bg_color, -1)
            cv2.putText(disp, text, (label_x, label_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
            
            conf_text = f"{conf:.2f}"
            conf_size, _ = cv2.getTextSize(conf_text, font, 0.5, 1)
            conf_w, conf_h = conf_size
            conf_x = x2 - conf_w - 4
            conf_y = y2 + conf_h + 4
            
            cv2.rectangle(disp, (conf_x - 2, conf_y - conf_h - 2), (conf_x + conf_w + 2, conf_y + 2), (0, 0, 0), -1)
            cv2.putText(disp, conf_text, (conf_x, conf_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return disp


def main():
    ap = argparse.ArgumentParser(description="车牌标注可视化")
    ap.add_argument("video", help="输入视频路径")
    ap.add_argument("--output_dir", default=None, help="输出目录")
    ap.add_argument("--sample_fps", type=int, default=4, help="采样帧率")
    ap.add_argument("--save_key_frames", action="store_true", help="保存识别到车牌的关键帧")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    if args.output_dir is None:
        args.output_dir = os.path.join(ROOT, "data", "output", f"plate_annotate_{video_name}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    key_frames_dir = os.path.join(args.output_dir, "key_frames")
    if args.save_key_frames:
        os.makedirs(key_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    interval = max(1, int(round(fps / args.sample_fps)))

    out_path = os.path.join(args.output_dir, "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    plate = PlateRecognizer(cfg, verbose=False)

    csv_rows = []
    frame_idx = 0
    key_frame_count = 0
    last_plates = []

    print(f"处理视频: {video_name}")
    print(f"视频尺寸: {W}x{H}")
    print(f"总帧数: {total_frames}")
    print(f"采样帧率: {args.sample_fps} fps")
    print(f"输出目录: {args.output_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            ts = frame_idx / fps
            plates = plate.detect(frame)
            last_plates = plates
            
            for p in plates:
                txt = p.get("text", "")
                conf = p.get("conf", 0.0)
                x1, y1, x2, y2 = p.get("xyxy", [0, 0, 0, 0])
                csv_rows.append({
                    "frame": frame_idx,
                    "time": round(ts, 2),
                    "plate": txt,
                    "confidence": round(conf, 4),
                    "color": p.get("color", "unknown"),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                })
                
                if txt:
                    print(f"[帧{frame_idx:5d} @{ts:6.2f}s] 车牌={txt:12s} conf={conf:.3f} 颜色={p.get('color','unknown')}")
            
            if args.save_key_frames and plates:
                annotated = draw_plate_annotation(frame, plates)
                plate_texts = "_".join([p["text"].replace(" ", "") for p in plates if p.get("text")])
                if plate_texts:
                    safe_name = f"frame_{frame_idx:06d}_time_{ts:.2f}s"
                    key_frame_path = os.path.join(key_frames_dir, f"{safe_name}.jpg")
                    cv2.imencode('.jpg', annotated)[1].tofile(key_frame_path)
                    key_frame_count += 1
                    print(f"  保存关键帧: {key_frame_path} ({plate_texts})")

        annotated_frame = draw_plate_annotation(frame, last_plates)
        video_writer.write(annotated_frame)
        
        if frame_idx % 500 == 0:
            print(f"进度: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
        
        frame_idx += 1

    cap.release()
    video_writer.release()

    csv_path = os.path.join(args.output_dir, "plate_detections.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time", "plate", "confidence", 
                                          "color", "x1", "y1", "x2", "y2"])
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n完成!")
    print(f"带标注视频: {out_path}")
    print(f"检测结果CSV: {csv_path}")
    if args.save_key_frames:
        print(f"关键帧截图: {key_frames_dir} ({key_frame_count} 张)")


if __name__ == "__main__":
    main()
