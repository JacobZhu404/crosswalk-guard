import os
import sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import load_config
from src.vehicle_detector import VehicleDetector
from src.crosswalk_detector import CrosswalkDetector
from src.traffic_light import TrafficLightDetector
from src.plate_recognizer import PlateRecognizer
from src.tracker import TrackStateManager
from src.visualizer import Visualizer


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else r"E:\BaiduNetdiskDownload\违章02.mp4"
    
    cfg = load_config("configs/config.yaml")
    cfg.stationary.speed_px_per_sec = 30
    cfg.stationary.sustain_frames = 5
    cfg.crosswalk.overlap_ratio = 0.20
    
    out_dir = os.path.join(ROOT, "data", "output", "plate_inspect")
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频:", video_path)
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    
    det = VehicleDetector(cfg)
    cw = CrosswalkDetector(cfg)
    tl = TrafficLightDetector(cfg)
    plate = PlateRecognizer(cfg)
    tracker = TrackStateManager(cfg)
    viz = Visualizer(cfg)
    
    frame_idx = 0
    proc = 0
    plate_results = []
    interval = max(1, int(round(fps / cfg.inference.fps)))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % interval == 0:
            proc += 1
            ts = frame_idx / fps
            
            dets = det.detect(frame)
            tracker.update(dets, ts)
            
            mask = cw.detect(frame)
            light = tl.detect(frame)
            
            if proc % getattr(cfg.inference, "plate_interval", 3) == 0:
                plates = plate.detect(frame)
                for p in plates:
                    if p.get("text"):
                        plate_results.append({
                            "frame": frame_idx,
                            "time": round(ts, 2),
                            "plate": p["text"],
                            "color": p.get("color", "unknown"),
                            "conf": round(p.get("conf", 0), 3),
                            "light": light,
                            "vehicles": len(dets),
                        })
                        print(f"[帧{frame_idx:4d} @{ts:.1f}s] 车牌: {p['text']} ({p.get('color','?')}) conf={p.get('conf',0):.3f} 灯={light} 车辆数={len(dets)}")
            
            stationary_count = sum(1 for st in tracker.states.values() if st.get("stationary", False))
            if stationary_count > 0:
                for tid, st in tracker.states.items():
                    if st.get("stationary", False):
                        print(f"    车辆#{tid} 静止状态: {st['stationary']} 速度累计: {st['stationary_count']}")
        
        frame_idx += 1
    
    cap.release()
    
    print("\n=== 车牌识别汇总 ===")
    for pr in plate_results:
        print(f"  {pr['time']:6.2f}s 车牌={pr['plate']:10} 颜色={pr['color']:6} 置信度={pr['conf']:.3f} 灯={pr['light']:7}")
    
    if plate_results:
        import csv
        csv_path = os.path.join(out_dir, "plates.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time", "plate", "color", "confidence", "light_state", "vehicle_count"])
            for pr in plate_results:
                w.writerow([pr["frame"], pr["time"], pr["plate"], pr["color"], pr["conf"], pr["light"], pr["vehicles"]])
        print(f"\n结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()
