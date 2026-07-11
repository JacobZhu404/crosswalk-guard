"""提取57.9秒帧并做车牌识别"""
import os
import sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    plate = PlateRecognizer(cfg)

    target_frame = int(107.5 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        return

    # 保存帧
    cv2.imwrite("data/output/debug_frame_57s.jpg", frame)
    print(f"已保存帧: data/output/debug_frame_57s.jpg")

    # 全帧车牌识别
    plates = plate.detect(frame)
    print("全帧车牌识别结果:")
    for p in plates:
        print(f"  车牌={p.get('text','')} conf={p.get('conf',0):.3f} color={p.get('color','')}")

    # 裁剪右下角区域 (根据之前的检测结果，#88车辆在右侧)
    h, w = frame.shape[:2]
    crop = frame[160:360, 1000:1280]
    cv2.imwrite("data/output/debug_crop_57s.jpg", crop)
    print(f"\n裁剪区域识别:")
    plates2 = plate.detect(crop)
    for p in plates2:
        print(f"  车牌={p.get('text','')} conf={p.get('conf',0):.3f}")

    cap.release()


if __name__ == "__main__":
    main()
