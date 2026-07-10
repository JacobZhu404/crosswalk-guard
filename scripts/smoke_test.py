"""临时冒烟测试: 在封网无权重环境下验证各纯CV模块是否work。
用法: python scripts/smoke_test.py <video> [frames_to_sample]
输出: data/output/smoke/ 下抽帧图(车辆/斑马线/红绿灯/车牌标注) + 控制台统计。
"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import load_config
from src.vehicle_detector import VehicleDetector
from src.crosswalk_detector import CrosswalkDetector
from src.traffic_light import TrafficLightDetector
from src.plate_recognizer import PlateRecognizer
from src.visualizer import Visualizer


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=r"E:\BaiduNetdiskDownload\违章10.mp4")
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    out_dir = os.path.join(ROOT, "data", "output", "smoke")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("无法打开视频:", args.video)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    det = VehicleDetector(cfg)
    cw = CrosswalkDetector(cfg)
    tl = TrafficLightDetector(cfg)
    plate = PlateRecognizer(cfg)
    viz = Visualizer(cfg)

    sampled = 0
    fi = 0
    stats = {"veh_frames": 0, "cw_frames": 0, "light": {}, "plate_total": 0,
             "light_unknown": 0}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 每 1秒 抽一帧测试
        if total and fi % max(1, int(fps)) != 0 and fi % max(1, int(total / max(1, args.frames))) != 0:
            fi += 1
            continue
        if sampled >= args.frames:
            break

        vehicles = det.detect(frame)
        mask = cw.detect(frame)
        light = tl.detect(frame)
        plates = plate.detect(frame, vehicle_boxes=[v["xyxy"] for v in vehicles])

        stats["veh_frames"] += (1 if vehicles else 0)
        if mask is not None:
            stats["cw_frames"] += 1
        stats["light"][light] = stats["light"].get(light, 0) + 1
        if light == "unknown":
            stats["light_unknown"] += 1
        stats["plate_total"] += len(plates)

        disp = viz.draw(frame, vehicles, {}, mask, light, plates=plates)
        # 中文路径/文件名安全, 用数字命名
        sp = os.path.join(out_dir, f"frame_{sampled:03d}.jpg")
        cv2.imwrite(sp, disp)
        print(f"[帧 {fi}] 车辆={len(vehicles)} 斑马线={'有' if mask is not None else '无'} "
              f"灯={light} 车牌={len(plates)} -> {sp}")
        for p in plates:
            print(f"    车牌 {p['color']} @ {[round(v) for v in p['xyxy']]} text='{p['text']}'")
        sampled += 1
        fi += 1

    cap.release()
    print("=== 统计 ===")
    print("抽样帧:", sampled)
    print("有车辆的帧:", stats["veh_frames"])
    print("检出斑马线的帧:", stats["cw_frames"])
    print("灯色分布:", stats["light"])
    print("定位车牌总数:", stats["plate_total"])


if __name__ == "__main__":
    main()
