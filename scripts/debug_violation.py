import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.infrastructure.geometry import compute_overlap_ratio
from redlight.models.vehicle import VehicleDetector
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.traffic_light import TrafficLightDetector
from redlight.pipeline.tracker import TrackStateManagerV2
from redlight.pipeline.violation_engine import ViolationEngineV2


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, int(round(fps / cfg.inference.fps)))
    
    det = VehicleDetector(cfg)
    cw = CrosswalkDetector(cfg)
    tl = TrafficLightDetector(cfg)
    
    tracker = TrackStateManagerV2("loose")
    engine = ViolationEngineV2("loose")
    
    frame_idx = 0
    proc = 0
    mask = None
    light_state = "unknown"
    
    print(f"画面尺寸: {W}x{H}  推理间隔: {interval}帧")
    print(f"预设: loose (speed=50, sustain=3, duration=3, overlap=0.15)")
    print("=" * 80)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % interval == 0:
            proc += 1
            ts = frame_idx / fps
            
            dets = det.detect(frame)
            states = tracker.update(dets, ts)
            
            if proc % 4 == 0:
                mask = cw.detect(frame)
            
            if proc % 8 == 0:
                light = tl.detect(frame)
                light_state = light.get("state", "unknown") if isinstance(light, dict) else light
            
            new_events = engine.evaluate(states, mask, light_state, ts)
            
            if new_events:
                print(f"\n[帧{frame_idx:4d} @{ts:.1f}s]")
                print(f"  发现违规事件! 灯={light_state}")
                for ev in new_events:
                    print(f"    事件#{ev['event_id']} 车辆#{ev['track_id']} {ev['status']}")
            
            if proc % 5 == 0:
                mask_area = int(cv2.countNonZero(mask)) if mask is not None else 0
                print(f"\n[帧{frame_idx:4d} @{ts:.1f}s]")
                print(f"  灯={light_state}  掩膜面积={mask_area}")
                for tid, st in states.items():
                    if not st.get("active", False):
                        continue
                    stationary = st.get("stationary", False)
                    ratio = compute_overlap_ratio(st["box"], mask) if mask is not None else 0
                    on_cw = ratio >= 0.15
                    print(f"    #{tid} {st['cls']} 静止={stationary} 压线={on_cw} overlap={ratio:.2f}")
        
        frame_idx += 1
    
    cap.release()
    
    events = engine.events
    print(f"\n{'=' * 80}")
    print(f"总计事件: {len(events)}")
    for ev in events:
        print(f"  事件#{ev['event_id']} 车辆#{ev['track_id']} {ev['status']} "
              f"{ev['start_ts']:.1f}s~{ev['end_ts']:.1f}s {ev['vehicle_class']}")


if __name__ == "__main__":
    main()
