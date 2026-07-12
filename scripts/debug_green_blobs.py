"""调试 (2026-07-11, E16): 02 绿灯窗口内绿色分量的真实几何 + 多组 HSV 阈值对比。

关键问题: 当前 green 分量只有 60-172 像素且碎片化 -> min_area=525 全滤掉。
需要确认:
  (1) 是否存在一个"一致性小亮斑"(行人绿灯本体), 还是纯噪声?
  (2) 当前 HSV 绿范围 [43,90,90]-[85,255,255] 是否漏掉了灯泡(饱和度偏低/偏青)?
  (3) 不同 S/V 下限 + 不同 hue 范围能召回多少绿像素, 是否形成单连通块?

输出: 每帧打印
  - 当前绿掩膜: 总像素 / 最大连通块面积 / 最大块质心cy / 绿质心(全掩膜均值)
  - 宽绿掩膜(S>=40): 同上
  - 青绿掩膜(hue 70-105): 同上
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
GREEN_WIN = (21.0, 68.0)
SAMPLE_FPS = 4


def blob_stats(mask, y0, y1, w):
    if mask.sum() == 0:
        return 0, 0, 0.0
    num, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best_a, best_cy = 0, 0.0
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > best_a:
            best_a = a
            yc = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
            best_cy = (yc + y0) / (y1)   # 全帧归一
    ys, xs = np.where(mask > 0)
    mean_cy = (float(np.mean(ys)) + y0) / y1 if len(ys) else 0.0
    return int(mask.sum()) // 255, best_a, round(mean_cy, 3)


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
        if fi % interval == 0 and GREEN_WIN[0] <= ts <= GREEN_WIN[1]:
            h, w = frame.shape[:2]
            y0, y1 = int(h * det.band[0]), int(h * det.band[1])
            roi = frame[y0:y1, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            g_cur = cv2.inRange(hsv, np.array([43, 90, 90]), np.array([85, 255, 255]))
            g_wide = cv2.inRange(hsv, np.array([35, 40, 60]), np.array([95, 255, 255]))
            g_cyang = cv2.inRange(hsv, np.array([60, 60, 60]), np.array([110, 255, 255]))

            print(f"ts={ts:5.1f}s | cur: cnt={blob_stats(g_cur,y0,y1,w)[0]:4d} "
                  f"maxblob={blob_stats(g_cur,y0,y1,w)[1]:4d} cy={blob_stats(g_cur,y0,y1,w)[2]} | "
                  f"wide: cnt={blob_stats(g_wide,y0,y1,w)[0]:4d} "
                  f"maxblob={blob_stats(g_wide,y0,y1,w)[1]:4d} cy={blob_stats(g_wide,y0,y1,w)[2]} | "
                  f"cyang: cnt={blob_stats(g_cyang,y0,y1,w)[0]:4d} "
                  f"maxblob={blob_stats(g_cyang,y0,y1,w)[1]:4d} cy={blob_stats(g_cyang,y0,y1,w)[2]}")
        fi += 1
    cap.release()
    print("[done]")


if __name__ == "__main__":
    main()
