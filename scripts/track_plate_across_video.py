"""全视频车牌跟踪脚本: 跟踪车辆完整轨迹，跨时间段关联车牌。

用法:
    python scripts/track_plate_across_video.py --video 违章05
    python scripts/track_plate_across_video.py --video 违章07 --save_frames
"""
import os
import sys
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.vehicle import VehicleDetector
from redlight.models.plate import PlateRecognizer
from redlight.pipeline.tracker import TrackStateManagerV2


def main():
    ap = argparse.ArgumentParser(description="全视频车牌跟踪")
    ap.add_argument("--video", required=True, help="视频名称(不含.mp4)")
    ap.add_argument("--fps", type=int, default=8, help="分析帧率")
    ap.add_argument("--save_frames", action="store_true", help="保存关键帧")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    
    video_path = os.path.join(ROOT, "input_video", f"{args.video}.mp4")
    if not os.path.exists(video_path):
        print(f"视频不存在: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    interval = max(1, int(round(src_fps / args.fps)))

    vehicle = VehicleDetector(cfg)
    plate = PlateRecognizer(cfg, verbose=False)
    tracker = TrackStateManagerV2("balanced")

    output_dir = os.path.join(ROOT, "data", "output", f"plate_track_{args.video}")
    if args.save_frames:
        os.makedirs(output_dir, exist_ok=True)

    print(f"分析视频: {args.video}")
    print(f"视频信息: {W}x{H} @ {src_fps:.1f}fps, 共 {total_frames} 帧")
    print(f"分析帧率: {args.fps} fps")
    print("-" * 60)

    frame_idx = 0
    track_history = {}
    track_plates = {}
    track_first_seen = {}
    track_last_seen = {}

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
            
            dets = vehicle.detect(frame)
            states = tracker.update(dets, ts)
            
            plates = plate.detect(frame)
            
            for tid, st in states.items():
                if not st.get("active"):
                    continue
                
                if tid not in track_history:
                    track_history[tid] = []
                    track_first_seen[tid] = ts
                track_history[tid].append({
                    "time": ts,
                    "box": st.get("box", []),
                    "stationary": st.get("stationary", False),
                })
                track_last_seen[tid] = ts

                if tid not in track_plates:
                    track_plates[tid] = {}

            for p in plates:
                if not p.get("text"):
                    continue
                txt = p["text"]
                conf = p.get("conf", 0.0)
                px1, py1, px2, py2 = p.get("xyxy", [0, 0, 0, 0])
                p_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                
                for tid, st in states.items():
                    if st.get("active") and st.get("box"):
                        bx1, by1, bx2, by2 = st["box"]
                        if bx1 <= p_center[0] <= bx2 and by1 <= p_center[1] <= by2:
                            if txt not in track_plates[tid]:
                                track_plates[tid][txt] = {"count": 0, "times": [], "confidences": []}
                            track_plates[tid][txt]["count"] += 1
                            track_plates[tid][txt]["times"].append(ts)
                            track_plates[tid][txt]["confidences"].append(conf)
                            
                            if args.save_frames:
                                frame_copy = frame.copy()
                                cv2.rectangle(frame_copy, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)
                                cv2.rectangle(frame_copy, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                                cv2.putText(frame_copy, f"{txt} ({conf:.2f})", (int(px1), int(py1)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frame_copy, f"track_id={tid}", (int(bx1), int(by1)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.putText(frame_copy, f"time={ts:.1f}s", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                cv2.imwrite(os.path.join(output_dir, f"track_{tid}_frame_{frame_idx:06d}_time_{ts:.2f}s.jpg"), frame_copy)
                        break

        frame_idx += 1

    cap.release()

    print("\n=== 车辆轨迹汇总 ===")
    for tid in sorted(track_history.keys(), key=lambda x: len(track_history[x]), reverse=True):
        history = track_history[tid]
        duration = track_last_seen[tid] - track_first_seen[tid]
        plate_info = track_plates.get(tid, {})
        
        print(f"\n  track_id={tid}:")
        print(f"    出现时间: {track_first_seen[tid]:.1f}s - {track_last_seen[tid]:.1f}s (持续 {duration:.1f}s)")
        print(f"    跟踪帧数: {len(history)}")
        
        stationary_count = sum(1 for h in history if h.get("stationary"))
        print(f"    静止帧数: {stationary_count}/{len(history)}")
        
        if plate_info:
            print(f"    识别到的车牌:")
            for txt, info in sorted(plate_info.items(), key=lambda x: x[1]["count"], reverse=True):
                avg_conf = np.mean(info["confidences"]) if info["confidences"] else 0
                print(f"      {txt:12s}: {info['count']}次, 平均置信度={avg_conf:.3f}, 时间点: {[round(t, 1) for t in info['times'][:5]]}{'...' if len(info['times']) > 5 else ''}")
        else:
            print(f"    ❌ 未识别到任何车牌")

    print("\n=== 关键发现 ===")
    vehicles_without_plate = [tid for tid in track_history if not track_plates.get(tid)]
    vehicles_with_plate = [tid for tid in track_history if track_plates.get(tid)]
    
    print(f"总车辆数: {len(track_history)}")
    print(f"识别到车牌的车辆: {len(vehicles_with_plate)}")
    print(f"未识别到车牌的车辆: {len(vehicles_without_plate)}")
    
    if vehicles_without_plate:
        print(f"\n未识别车牌的车辆track_id: {vehicles_without_plate}")
        print("这些车辆可能需要在其他时间段寻找识别机会")


if __name__ == "__main__":
    main()
