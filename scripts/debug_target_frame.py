"""查看107秒附近的违规条件"""
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


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    det = VehicleDetector(cfg)
    cw = CrosswalkDetector(cfg)
    tl = TrafficLightDetector(cfg)
    tracker = TrackStateManagerV2("loose")
    engine = ViolationEngineV2("loose")

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
        mask = cw.detect(frame)
        light = tl.detect(frame)
        light_state = light.get("state", "unknown") if isinstance(light, dict) else light

        events = engine.evaluate(states, mask, light_state, ts)

        mask_area = cv2.countNonZero(mask) if mask is not None else 0

        print(f"[帧{frame_idx:5d} @{ts:6.2f}s] 灯={light_state:8s} 掩膜={mask_area:6d}")
        for tid, st in states.items():
            if st.get("active"):
                overlap = st.get("overlap", 0)
                on_cw = overlap >= 0.15
                print(f"  #{tid} car 静止={st.get('stationary',False)} 压线={on_cw} overlap={overlap:.3f}")
        if events:
            print(f"  ⚠️ 违规事件: {len(events)}")

        frame_idx += 1
    cap.release()


if __name__ == "__main__":
    main()
