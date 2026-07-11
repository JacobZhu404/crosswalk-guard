import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.crosswalk import CrosswalkDetector


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cw = CrosswalkDetector(cfg)
    
    out_dir = os.path.join(ROOT, "data", "output", "cw_debug")
    os.makedirs(out_dir, exist_ok=True)
    
    frame_idx = 0
    sample_frames = [116, 232, 348, 464, 580, 812, 1392, 1740, 2204, 2668]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in sample_frames:
            ts = frame_idx / fps
            
            mask = cw.detect(frame)
            mask_area = int(cv2.countNonZero(mask))
            
            h, w = frame.shape[:2]
            y0 = int(h * 0.45)
            roi = frame[y0:, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(gray_blur, 140, 255, cv2.THRESH_BINARY)
            
            print(f"\n[帧{frame_idx} @{ts:.1f}s]")
            print(f"  画面: {w}x{h}  ROI起点: y={y0}")
            print(f"  掩膜面积: {mask_area}")
            print(f"  ROI灰度均值: {gray.mean():.1f}  中值: {np.median(gray):.1f}")
            
            overlay = frame.copy()
            if mask_area > 0:
                overlay[mask > 0] = [0, 255, 255]
            cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx}_overlay.jpg"), overlay)
            
            cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx}_gray.jpg"), gray)
            cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx}_binary.jpg"), binary)
            cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx}_mask.jpg"), mask)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"  二值图轮廓数: {len(contours)}")
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                x, y, bw, bh = cv2.boundingRect(c)
                ar = bw / float(bh) if bh > 0 else 0
                print(f"    轮廓{i}: area={area:.0f} ar={ar:.2f} pos=({x},{y}) size=({bw}x{bh})")
        
        frame_idx += 1
    
    cap.release()
    print(f"\n调试图像已保存到: {out_dir}")


if __name__ == "__main__":
    main()
