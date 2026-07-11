"""查看mask和车辆框的具体位置"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.vehicle import VehicleDetector
from redlight.models.crosswalk import CrosswalkDetector


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    det = VehicleDetector(cfg)
    cw = CrosswalkDetector(cfg)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        if ts < 107.0 or ts > 107.5:
            frame_idx += 1
            continue

        dets = det.detect(frame)
        mask = cw.detect(frame)

        h, w = frame.shape[:2]
        mask_area = cv2.countNonZero(mask) if mask is not None else 0
        
        print(f"\n[帧{frame_idx} @{ts:.2f}s] 画面={w}x{h} 掩膜面积={mask_area}")
        
        if mask is not None:
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                print(f"  mask边界: ({x_min},{y_min}) ~ ({x_max},{y_max})")

        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            print(f"  车辆框 #{d.get('id','?')}: ({x1:.0f},{y1:.0f}) ~ ({x2:.0f},{y2:.0f}) 中心=({cx:.0f},{cy:.0f})")

            if mask is not None:
                bx1, by1 = max(0, int(x1)), max(0, int(y1))
                bx2, by2 = min(w - 1, int(x2)), min(h - 1, int(y2))
                sub = mask[by1:by2, bx1:bx2]
                inside = int(np.count_nonzero(sub > 0))
                box_area = (bx2 - bx1) * (by2 - by1)
                ratio = inside / box_area if box_area > 0 else 0.0
                print(f"    overlap={ratio:.4f} (inside={inside}, box_area={box_area})")

        frame_idx += 1
    cap.release()


if __name__ == "__main__":
    main()
