"""快速验证 v4 信号灯检测器在 违章02 的逐帧状态 (不跑 YOLO, 秒级完成)。

输出:
  - 信号灯状态直方图
  - 21-68s( GT 绿灯窗口) 内各状态的帧数占比
  - 关键窗口(21-68, 69-104, 0-20) 的状态分布, 判断 v4 是否还原 "绿灯" 语义
"""
import sys
import os

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np

from redlight.infrastructure.config import load_config, project_root
from redlight.models.traffic_light import TrafficLightDetector


VIDEO = os.path.join(ROOT, "input_video", "违章02.mp4")
SAMPLE_FPS = 8  # 与 cli 同 interval


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    det = TrafficLightDetector(cfg, verbose=False)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / SAMPLE_FPS)))
    fi = 0
    hist = {}
    win = {"0-20": {}, "21-68": {}, "69-104": {}, "105+": {}}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % interval == 0:
            ts = fi / fps
            st = det.detect(frame).get("state")
            hist[st] = hist.get(st, 0) + 1
            if ts < 20:
                w = "0-20"
            elif ts <= 68:
                w = "21-68"
            elif ts <= 104:
                w = "69-104"
            else:
                w = "105+"
            win[w][st] = win[w].get(st, 0) + 1
        fi += 1
    cap.release()

    print("===== 02 全片 信号灯状态直方图 =====")
    print(dict(sorted(hist.items(), key=lambda x: -x[1])))
    for w, d in win.items():
        tot = sum(d.values())
        print(f"  窗口 {w}: 总 {tot} 帧 -> {dict(sorted(d.items(), key=lambda x:-x[1]))}")


if __name__ == "__main__":
    main()
