"""v4 细节验证: 02 绿灯窗口内, 实际进入聚合的( ped>0 )红/绿候选几何。

看清为什么 21-68s 仍被大量判 red: 是车灯漏过了 max_area/cutoff, 还是确有红色信号?
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
GREEN_WIN = (21.0, 68.0)
SAMPLE_FPS = 8


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    det = TrafficLightDetector(cfg, verbose=False)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / SAMPLE_FPS)))
    fi = 0
    samples = 0
    green_only = 0
    red_only = 0
    both = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = fi / fps
        if fi % interval == 0 and GREEN_WIN[0] <= ts <= GREEN_WIN[1]:
            cands = det._candidates(frame, None)
            sig = [c for c in cands if c["ped"] > 0]   # 进入聚合的信号级候选
            g = [c for c in sig if c["color"] == "green"]
            r = [c for c in sig if c["color"] == "red"]
            samples += 1
            if g and not r:
                green_only += 1
            elif r and not g:
                red_only += 1
            elif g and r:
                both += 1
            if samples <= 12 or samples % 25 == 0:
                gtxt = [(round(c["cy"], 3), c["area"], round(c["ped"], 2)) for c in g]
                rtxt = [(round(c["cy"], 3), c["area"], round(c["ped"], 2)) for c in r]
                print(f"ts={ts:5.1f} GREEN={gtxt}  RED={rtxt}")
        fi += 1
    cap.release()
    print(f"\n窗口内 {samples} 采样帧: green_only={green_only} red_only={red_only} both={both}")


if __name__ == "__main__":
    main()
