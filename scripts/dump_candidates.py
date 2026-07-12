"""dump 07 若干帧的候选亮斑细节(位置/尺寸/颜色/发射), 定位红绿灯到底在哪、被什么淹没。"""
import sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from redlight.models.traffic_light import TrafficLightDetector
from redlight.infrastructure.config import load_config


def dump(video, ts=(3, 8, 15, 22, 30, 38, 42), cutoff=0.6):
    cfg = load_config("configs/config.yaml")
    det = TrafficLightDetector(cfg, verbose=False)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fi = 0
    want = {int(t * fps) for t in ts}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in want:
            spots = det._candidates(frame)
            t = fi / fps
            print(f"\n===== t={t:.1f}s  (W={W} H={H})  total spots={len(spots)} =====")
            # 按发射强度排序, 打印前 18 个
            sp = sorted(spots, key=lambda s: -s.get("bright", 0))[:18]
            print(f"  {'color':>6} {'cx':>5} {'cy':>5} {'w':>4} {'h':>4} {'area':>5} {'frac_v':>6} {'bright':>7}")
            for s in sp:
                x1, y1, x2, y2 = s["box"]
                w = x2 - x1; h = y2 - y1
                print(f"  {s['color']:>6} {s['cx']:5.2f} {s['cy']:5.2f} {w:4d} {h:4d} {s['area']:5d} {s['frac_v']:6.2f} {s['bright']:7.1f}")
        fi += 1
        if fi > max(want) + 2:
            break
    cap.release()


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "input_video/违章07.mp4"
    dump(vid)
