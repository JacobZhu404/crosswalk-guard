"""调试跟踪与车牌关联: 查看107秒左右各帧的track_id与车牌匹配情况。"""
import os
import sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.vehicle import VehicleDetector
from redlight.models.plate import PlateRecognizer
from redlight.pipeline.tracker import TrackStateManagerV2


def overlap(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    return inter / min(area1, area2)


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    det = VehicleDetector(cfg)
    plate = PlateRecognizer(cfg)
    tracker = TrackStateManagerV2("loose")

    print(f"{'帧':>6s} {'时间':>7s} | 车辆框 | 车牌(关联track_id, overlap)")
    print("-" * 90)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        if ts < 106.5 or ts > 109.5:
            frame_idx += 1
            continue
        if frame_idx % 4 != 0:
            frame_idx += 1
            continue

        dets = det.detect(frame)
        states = tracker.update(dets, ts)
        plates = plate.detect(frame)

        boxes_str = ",".join([f"tid{d.get('track_id','?')}" for d in dets])
        plate_strs = []
        for p in plates:
            if not p.get("text"):
                continue
            best_tid = None
            best_ov = 0.0
            for tid, st in states.items():
                if st.get("active") and st.get("box"):
                    ov = overlap(st["box"], p.get("xyxy", [0, 0, 0, 0]))
                    if ov > best_ov:
                        best_ov = ov
                        best_tid = tid
            plate_strs.append(f"{p['text']}(tid{best_tid},ov={best_ov:.2f})")

        if plate_strs:
            print(f"{frame_idx:6d} {ts:7.2f} | [{boxes_str}] | {', '.join(plate_strs)}")

        frame_idx += 1
    cap.release()


if __name__ == "__main__":
    main()
