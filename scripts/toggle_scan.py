"""信号灯判别: 同一位置既在红段常红、又在绿段转绿 -> 才是真信号灯.
静态红物体(永远红)和偶发反光(闪烁)都将被排除.

用法:
  python scripts/toggle_scan.py 违章04 --stride 2
"""
import sys, os, argparse
import numpy as np
import cv2

FRAMES_ROOT = os.path.join("datasets", "frames")


def robust_imread(path):
    with open(path, "rb") as f:
        buf = f.read()
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def frame_blobs(hsv, hue_ranges, s_min=100, v_min=60, v_max=235, area_min=8, area_max=1500, cy_max=0.7):
    out = []
    for (h0, h1) in hue_ranges:
        if h0 <= h1:
            m = (hsv[:, :, 0] >= h0) & (hsv[:, :, 0] <= h1)
        else:
            m = (hsv[:, :, 0] >= h0) | (hsv[:, :, 0] <= h1)
        m &= (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min) & (hsv[:, :, 2] <= v_max)
        n, _, st, ce = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        H = hsv.shape[0]
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < area_min or a > area_max:
                continue
            if ce[i, 1] / H > cy_max:
                continue
            out.append((round(ce[i, 0] / hsv.shape[1] / 0.04) * 0.04,
                        round(ce[i, 1] / H / 0.04) * 0.04))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--green-from", type=float, default=41.0)
    ap.add_argument("--fps", type=float, default=29.7)
    args = ap.parse_args()

    vdir = os.path.join(FRAMES_ROOT, args.video)
    files = sorted([f for f in os.listdir(vdir) if f.endswith(".jpg")])

    red_frames = {}
    green_frames = {}
    total = 0
    for i, fn in enumerate(files):
        if i % args.stride != 0:
            continue
        idx = int(fn.split("_")[1].split(".")[0])
        t = idx / args.fps
        fr = robust_imread(os.path.join(vdir, fn))
        if fr is None:
            continue
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        reds = frame_blobs(hsv, [(0, 12), (168, 179)])
        grns = frame_blobs(hsv, [(38, 82)])
        for p in reds:
            red_frames[p] = red_frames.get(p, 0) + 1
        for p in grns:
            green_frames[p] = green_frames.get(p, 0) + 1
        total += 1

    print(f"\n{args.video}: 分析 {total} 帧, 绿段起点 t>={args.green_from}s")
    # 候选: 既常红又偶绿 (真信号灯)
    cands = []
    for p, rf in red_frames.items():
        gf = green_frames.get(p, 0)
        if rf >= total * 0.3 and gf >= 2:
            cands.append((p, rf, gf))
    cands.sort(key=lambda x: (-x[2], -x[1]))
    print(f"\n=== 真信号灯候选 (红≥30%帧 且 绿≥2帧) ===")
    for p, rf, gf in cands[:12]:
        print(f"  ({p[0]:.3f},{p[1]:.3f})  red_frames={rf}/{total}  green_frames={gf}")
    # 反例: 永远红(静态物体)
    statics = [(p, rf) for p, rf in red_frames.items() if rf >= total * 0.5 and green_frames.get(p, 0) == 0]
    statics.sort(key=lambda x: -x[1])
    print(f"\n=== 静态红物体(永远红, 非信号) Top6 ===")
    for p, rf in statics[:6]:
        print(f"  ({p[0]:.3f},{p[1]:.3f})  red_frames={rf}/{total}")


if __name__ == "__main__":
    main()
