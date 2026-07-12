"""实验 (2026-07-11): 用"高亮紧凑亮斑 + 均值色分类"替代极端像素阈值,

看能否在 02 绿灯窗口(21-68s)稳定检测到一个绿色信号灯 (green_persist 能否 >0.5)。

方法:
  1. lit = V>=v_thr & S>=s_thr  (只取被照亮的灯泡)
  2. 连通分量, 保留 紧凑(aspect<2.2) 且 面积在 [min,max] 且 在画面上部(ped>0)
  3. 对每个候选, 取其 bbox 内原图 HSV 均值, 按 mean_hue 分类 绿/红/其它
  4. 逐帧统计 green_present, 计算 21-68s 的 green_persist 与 red_persist
"""
import sys
import os

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np

from redlight.infrastructure.config import load_config, project_root

VIDEO = os.path.join(ROOT, "input_video", "违章02.mp4")
GREEN_WIN = (21.0, 68.0)
SAMPLE_FPS = 8


def lit_candidates(hsv, v_thr, s_thr, rh, rw, min_a, max_a, peak, width, cutoff):
    lit = cv2.inRange(hsv, np.array([0, s_thr, v_thr]), np.array([180, 255, 255]))
    # 形态学: 去掉毛刺
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, k)
    num, _, stats, cents = cv2.connectedComponentsWithStats(lit, 8)
    out = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_a or a > max_a:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 2.2:
            continue
        cy = (y + bh / 2.0) / rh
        if cy > cutoff:
            continue
        # 均值色分类
        patch = hsv[y:y + bh, x:x + bw]
        mean_h = float(np.mean(patch[:, :, 0]))
        mean_s = float(np.mean(patch[:, :, 1]))
        # 绿: hue 35-95 且 sat 够; 红: hue<15 或 >165
        color = None
        if 35 <= mean_h <= 95 and mean_s >= 30:
            color = "green"
        elif (mean_h <= 15 or mean_h >= 165) and mean_s >= 30:
            color = "red"
        if color:
            d = abs(cy - peak)
            ped = 1.0 - 0.4 * (d / width) if d <= width else 0.3
            out.append((color, a, round(cy, 3), round(ped, 2)))
    return out


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / SAMPLE_FPS)))
    fi = 0
    combos = [
        dict(v_thr=200, s_thr=50, min_a=15, max_a=2500, peak=0.18, width=0.30, cutoff=0.60),
        dict(v_thr=180, s_thr=40, min_a=15, max_a=2500, peak=0.18, width=0.30, cutoff=0.60),
        dict(v_thr=210, s_thr=70, min_a=10, max_a=1500, peak=0.18, width=0.30, cutoff=0.60),
    ]
    for c in combos:
        g_present = r_present = 0
        win_g = win_r = win_n = 0
        fi2 = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi2 % interval == 0:
                h, w = frame.shape[:2]
                y0, y1 = 0, int(h * 0.95)
                roi = frame[y0:y1, :]
                rh = y1 - y0
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                cands = lit_candidates(hsv, c["v_thr"], c["s_thr"], rh, w,
                                       c["min_a"], c["max_a"], c["peak"], c["width"], c["cutoff"])
                g = any(col == "green" for col, *_ in cands)
                r = any(col == "red" for col, *_ in cands)
                ts = fi2 / fps
                if g:
                    g_present += 1
                    if GREEN_WIN[0] <= ts <= GREEN_WIN[1]:
                        win_g += 1
                if r:
                    r_present += 1
                    if GREEN_WIN[0] <= ts <= GREEN_WIN[1]:
                        win_r += 1
            fi2 += 1
        # 重算窗口总帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        wframes = 0
        fi2 = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi2 % interval == 0:
                ts = fi2 / fps
                if GREEN_WIN[0] <= ts <= GREEN_WIN[1]:
                    wframes += 1
            fi2 += 1
        gp = g_present / (fi2_approx := max(1, fi2)) if False else None
        print(f"combo {c}:")
        print(f"  全片 green_frames={g_present} red_frames={r_present}")
        print(f"  窗口21-68: green={win_g}/{wframes} ({win_g/max(1,wframes):.2f}) "
              f"red={win_r}/{wframes} ({win_r/max(1,wframes):.2f})")
    cap.release()


if __name__ == "__main__":
    main()
