"""检查#103车辆的速度"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.vehicle import VehicleDetector
from redlight.pipeline.tracker import TrackStateManagerV2


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / 8)))

    det = VehicleDetector(cfg)
    tracker = TrackStateManagerV2("very_loose")

    frame_idx = 0
    proc_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        if frame_idx % interval != 0:
            frame_idx += 1
            continue

        proc_idx += 1
        dets = det.detect(frame)
        states = tracker.update(dets, ts)

        if ts >= 107.0 and ts <= 109.0:
            for tid, st in states.items():
                if st.get("active"):
                    hist = st.get("history", [])
                    if len(hist) >= 2:
                        t0, x0, y0 = hist[-2]
                        t1, x1, y1 = hist[-1]
                        dt = max(t1 - t0, 1e-3)
                        speed = float(np.hypot(x1 - x0, y1 - y0) / dt)
                        print(f"[帧{frame_idx} @{ts:.2f}s] #{tid} 静止={st['stationary']} 速度={speed:.1f}px/s")
                    else:
                        print(f"[帧{frame_idx} @{ts:.2f}s] #{tid} 静止={st['stationary']} (历史不足)")

        frame_idx += 1
    cap.release()


if __name__ == "__main__":
    main()
