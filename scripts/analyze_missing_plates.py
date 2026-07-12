"""未识别车牌追踪分析脚本: 追踪特定车辆在视频中的完整轨迹，查找其他角度的识别机会。

用法:
    python scripts/analyze_missing_plates.py --video 违章05 --gt_plate 京ADH9206
    python scripts/analyze_missing_plates.py --video 违章07 --gt_plate 京EJQ505
"""
import os
import sys
import argparse

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.vehicle import VehicleDetector
from redlight.models.plate import PlateRecognizer
from redlight.pipeline.tracker import TrackStateManagerV2
from redlight.pipeline.plate_consensus import PlateConsensus


def main():
    ap = argparse.ArgumentParser(description="未识别车牌追踪分析")
    ap.add_argument("--video", required=True, help="视频名称(不含.mp4)")
    ap.add_argument("--gt_plate", required=True, help="真值车牌")
    ap.add_argument("--fps", type=int, default=8, help="分析帧率")
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
    consensus = PlateConsensus(keep_history=300)

    print(f"分析视频: {args.video}")
    print(f"目标车牌: {args.gt_plate}")
    print(f"视频信息: {W}x{H} @ {src_fps:.1f}fps, 共 {total_frames} 帧")
    print(f"分析帧率: {args.fps} fps")
    print("-" * 60)

    frame_idx = 0
    track_plates = {}
    plate_timestamps = {}

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
            
            for p in plates:
                if not p.get("text"):
                    continue
                txt = p["text"]
                px1, py1, px2, py2 = p.get("xyxy", [0, 0, 0, 0])
                p_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                
                best_tid = None
                best_iou = 0.0
                for tid, st in states.items():
                    if st.get("active") and st.get("box"):
                        bx1, by1, bx2, by2 = st["box"]
                        if bx1 <= p_center[0] <= bx2 and by1 <= p_center[1] <= by2:
                            best_tid = tid
                            break
                
                if best_tid is not None:
                    consensus.update(best_tid, txt, p.get("conf", 0.0), ts)
                    
                    if best_tid not in track_plates:
                        track_plates[best_tid] = []
                    track_plates[best_tid].append({
                        "time": ts,
                        "plate": txt,
                        "conf": p.get("conf", 0.0),
                    })
                    
                    if txt not in plate_timestamps:
                        plate_timestamps[txt] = []
                    plate_timestamps[txt].append(ts)

        frame_idx += 1

    cap.release()

    print("\n=== 所有车牌识别时间点 ===")
    for txt, times in sorted(plate_timestamps.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {txt:12s}: {len(times)} 次, 时间点: {[round(t, 1) for t in times[:10]]}{'...' if len(times) > 10 else ''}")

    print("\n=== 车辆跟踪与车牌关联 ===")
    for tid, plates in sorted(track_plates.items(), key=lambda x: len(x[1]), reverse=True):
        plate_counts = {}
        for p in plates:
            plate_counts[p["plate"]] = plate_counts.get(p["plate"], 0) + 1
        
        best_plate = max(plate_counts, key=plate_counts.get)
        best_count = plate_counts[best_plate]
        times = [round(p["time"], 1) for p in plates[:10]]
        
        print(f"  track_id={tid}:")
        print(f"    识别到 {len(plates)} 次车牌")
        print(f"    最佳车牌: {best_plate} ({best_count}次)")
        print(f"    时间点: {times}{'...' if len(plates) > 10 else ''}")
        print(f"    所有识别: {plate_counts}")

    print("\n=== 目标车牌分析 ===")
    if args.gt_plate in plate_timestamps:
        times = plate_timestamps[args.gt_plate]
        print(f"  ✅ 成功识别到 {args.gt_plate}: {len(times)} 次")
        print(f"  时间点: {[round(t, 1) for t in times]}")
    else:
        print(f"  ❌ 未识别到 {args.gt_plate}")
        
        print("\n  ⚠️ 检查近似匹配...")
        import difflib
        for txt in plate_timestamps:
            similarity = difflib.SequenceMatcher(None, args.gt_plate, txt).ratio()
            if similarity >= 0.7:
                print(f"    近似车牌: {txt} (相似度={similarity:.2f}), 识别 {len(plate_timestamps[txt])} 次")
        
        print("\n  📊 可能的原因:")
        print("    1. 车辆角度: 侧面/背面拍摄，看不到车牌")
        print("    2. 遮挡: 被其他车辆、行人或障碍物遮挡")
        print("    3. 光线: 逆光、阴影导致对比度低")
        print("    4. 车辆颜色: 黑车在某些光线下难识别")
        print("\n  💡 建议:")
        print("    - 查看视频中该车辆的完整轨迹")
        print("    - 在车辆驶入/驶出路口时可能有更好角度")
        print("    - 检查车辆是否被其他物体遮挡")


if __name__ == "__main__":
    main()
