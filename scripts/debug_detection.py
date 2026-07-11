import os
import sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.vehicle import VehicleDetector
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.traffic_light import TrafficLightDetector
from redlight.models.plate import PlateRecognizer
from redlight.pipeline.tracker import TrackStateManagerV2, SENSITIVITY_PRESETS


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    interval = max(1, int(round(fps / cfg.inference.fps)))
    
    det = VehicleDetector(cfg)
    cw = CrosswalkDetector(cfg)
    tl = TrafficLightDetector(cfg)
    plate = PlateRecognizer(cfg)
    
    for preset_name in ["strict", "balanced", "loose"]:
        print(f"\n=== 预设: {preset_name} ===")
        p = SENSITIVITY_PRESETS[preset_name]
        print(f"  speed_thres={p['speed']} sustain={p['sustain']} duration={p['duration']} overlap={p['overlap']}")
    
    tracker = TrackStateManagerV2("loose")
    
    frame_idx = 0
    proc = 0
    sample_interval = int(fps)
    light_state = "unknown"
    
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
                mask_area = int(cv2.countNonZero(mask)) if mask is not None else 0
            else:
                mask = None
                mask_area = 0
            
            if proc % 8 == 0:
                light = tl.detect(frame)
                if isinstance(light, dict):
                    light_state = light.get("state", "unknown")
                else:
                    light_state = light
            
            if frame_idx % sample_interval == 0:
                print(f"\n[帧{frame_idx:4d} @{ts:.1f}s]")
                print(f"  车辆: {len(dets)} 辆")
                for d in dets:
                    tid = d["id"]
                    st = states.get(tid, {})
                    stationary = st.get("stationary", False)
                    print(f"    #{tid} {d['cls']} conf={d['conf']:.2f} 静止={stationary}")
                
                print(f"  斑马线掩膜面积: {mask_area}")
                print(f"  红绿灯状态: {light_state}")
                
                plates = plate.detect(frame)
                for p in plates:
                    if p.get("text"):
                        print(f"  车牌: {p['text']} ({p.get('color','?')}) conf={p.get('conf',0):.2f}")
        
        frame_idx += 1
    
    cap.release()


if __name__ == "__main__":
    main()
