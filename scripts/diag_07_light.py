"""诊断 07 红绿灯识别: 逐秒打印 最终state + 该帧上部区域 绿/红 发射强度之和。

用于验证 v7 重写的"发射强度选灯"是否真的修复 07(前43s应为绿灯)。
"""
import sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from redlight.models.traffic_light import TrafficLightDetector
from redlight.infrastructure.config import load_config


def diag(video, max_sec=45, cutoff=0.6):
    cfg = load_config("configs/config.yaml")
    det = TrafficLightDetector(cfg, verbose=False)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    fi = 0
    last_print = -10
    print(f"{'t(s)':>6} {'state':>9} {'g_sp':>5} {'r_sp':>5} {'ax':>5} {'ay':>5} {'sel':>6}")
    print("-" * 52)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if t > max_sec:
            break
        res = det.detect(frame)
        cands = res.get("candidates", [])
        g_n = r_n = 0
        for c in cands:
            if c["cy"] >= cutoff:
                continue
            if c["color"] == "green":
                g_n += 1
            elif c["color"] == "red":
                r_n += 1
        anc = res.get("anchor") or {}
        ax = anc.get("cx", -1.0)
        ay = anc.get("cy", -1.0)
        sel = anc.get("last_dom", "-")
        if t - last_print >= 1.0:
            last_print = t
            print(f"{t:6.1f} {res['state']:>9} {g_n:5d} {r_n:5d} {ax:5.2f} {ay:5.2f} {sel:>6}")
        fi += 1
    cap.release()


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "input_video/违章07.mp4"
    sec = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    diag(vid, max_sec=sec)
