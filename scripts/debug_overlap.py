"""检查#163车辆的位置和overlap"""
import os
import sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.vehicle import VehicleDetector
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.traffic_light import TrafficLightDetector
from redlight.pipeline.tracker import TrackStateManagerV2
from redlight.pipeline.violation_engine import ViolationEngineV2
from redlight.infrastructure.geometry import compute_overlap_ratio


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / 8)))

    det = VehicleDetector(cfg)
    cw = CrosswalkDetector(cfg)
    tl = TrafficLightDetector(cfg)
    tracker = TrackStateManagerV2("loose")
    engine = ViolationEngineV2("loose", mode="red_light")

    cw_int = 4
    mask = None

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

        if proc_idx % cw_int == 0:
            mask = cw.detect(frame)

        dets = det.detect(frame)
        states = tracker.update(dets, ts)

        light = tl.detect(frame)
        light_state = light.get("state", "unknown") if isinstance(light, dict) else light

        if ts >= 107.5 and ts <= 108.5:
            ma = cv2.countNonZero(mask) if mask is not None else 0
            print(f"[帧{frame_idx} @{ts:.2f}s] 灯={light_state:8s} 掩膜={ma:6d}")
            for tid, st in states.items():
                if st.get("active"):
                    box = st["box"]
                    ov = compute_overlap_ratio(box, mask) if mask is not None else 0.0
                    print(f"  #{tid} 静止={st['stationary']} box={[int(v) for v in box]} overlap={ov:.4f}")

        frame_idx += 1
    cap.release()


if __name__ == "__main__":
    main()
