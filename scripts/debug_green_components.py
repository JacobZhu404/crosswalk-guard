"""调试 (2026-07-11, E16): 直视 违章02 绿灯窗口内的红/绿灯候选几何。

目的:
  band 修正为 [0,0.95] 后, 红灯检测器现在能看到下部像素。但必须区分
  "行人绿灯(紧凑亮斑, 固定位置)" vs "绿色草地/植被(弥散大块)" (E15)。
  本脚本逐帧 dump 红/绿连通分量: 面积 / bbox / 质心cy / ped先验分,
  并叠加斑马线掩膜质心, 帮助决定 min_area / 位置先验 / 形态门限。

用法:
    python scripts/debug_green_components.py
"""
import sys
import os

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np

from redlight.infrastructure.config import load_config, project_root
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.traffic_light import TrafficLightDetector


VIDEO = os.path.join(ROOT, "input_video", "违章02.mp4")
GREEN_WIN = (21.0, 68.0)   # GT: 行人绿灯窗口
SAMPLE_FPS = 4


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    det = TrafficLightDetector(cfg, verbose=False)
    cw = CrosswalkDetector(cfg)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / SAMPLE_FPS)))
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = fi / fps
        if fi % interval == 0 and GREEN_WIN[0] - 5 <= ts <= GREEN_WIN[1] + 5:
            mask = cw.detect(frame)
            cands = det._candidates(frame, mask)
            # 也直算原始分量
            h, w = frame.shape[:2]
            y0, y1 = int(h * det.band[0]), int(h * det.band[1])
            roi = frame[y0:y1, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, np.array([43, 90, 90]), np.array([85, 255, 255]))
            red = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([170, 90, 90]), np.array([180, 255, 255])))
            min_area = max(det.min_pixels, int(det.min_area_ratio * (y1 - y0) * w))

            cw_cy = ""
            if mask is not None and mask.sum() > 0:
                ys, _ = np.where(mask > 0)
                cw_cy = f"{float(np.mean(ys)) / h:.3f}"

            gcomps = cv2.connectedComponentsWithStats(green, 8)
            rcomps = cv2.connectedComponentsWithStats(red, 8)

            def top3(comps, color):
                num = comps[0]
                stats = comps[2]
                cents = comps[3]
                items = []
                for i in range(1, num):
                    a = int(stats[i, cv2.CC_STAT_AREA])
                    if a < min_area:
                        continue
                    x = int(stats[i, cv2.CC_STAT_LEFT])
                    y = int(stats[i, cv2.CC_STAT_TOP])
                    bw = int(stats[i, cv2.CC_STAT_WIDTH])
                    bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                    cy = (y + bh / 2.0) / (y1 - y0)
                    ped = det._pedestrian_prior(cy, det.ped_peak)
                    items.append((a, x, y + y0, bw, bh, round(cy, 3), round(ped, 2)))
                items.sort(reverse=True)
                return items[:3]

            print(f"\n--- ts={ts:.1f}s band=({det.band[0]},{det.band[1]}) "
                  f"min_area={min_area} cw_cy={cw_cy} ---")
            print(f"  GREEN px={int(green.sum())}  RED px={int(red.sum())}")
            print("  green comps(top3):", top3(gcomps, "g"))
            print("  red   comps(top3):", top3(rcomps, "r"))
            # 也列出 detector 实际返回的候选
            print("  DETECTOR cands:", [(c["color"], c["area"], round(c["cy"], 3),
                                         round(c["ped"], 2)) for c in cands])
        fi += 1
    cap.release()
    print("\n[done]")


if __name__ == "__main__":
    main()
