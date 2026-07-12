"""反射鲁棒的信号灯定位: 真信号灯在红段几乎每帧都出现、且亮斑面积稳定(物理尺寸固定);
反光面积忽大忽小。再验证该位置在绿段出现绿斑 -> 锁定 prior。

用法:
  python scripts/stable_lamp.py 违章04 --stride 1
"""
import sys, os, argparse
import numpy as np
import cv2

FRAMES_ROOT = os.path.join("datasets", "frames")


def robust_imread(path):
    with open(path, "rb") as f:
        buf = f.read()
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def frame_blobs(hsv, hue_ranges, s_min=100, v_min=60, v_max=235, area_min=8, area_max=1200, cy_max=0.7):
    H, W = hsv.shape[:2]
    out = []
    for (h0, h1) in hue_ranges:
        if h0 <= h1:
            m = (hsv[:, :, 0] >= h0) & (hsv[:, :, 0] <= h1)
        else:
            m = (hsv[:, :, 0] >= h0) | (hsv[:, :, 0] <= h1)
        m &= (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min) & (hsv[:, :, 2] <= v_max)
        n, _, st, ce = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < area_min or a > area_max:
                continue
            if ce[i, 1] / H > cy_max:
                continue
            out.append((round(ce[i, 0] / W / 0.03) * 0.03,
                        round(ce[i, 1] / H / 0.03) * 0.03, int(a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--green-from", type=float, default=41.0)
    ap.add_argument("--fps", type=float, default=29.7)
    args = ap.parse_args()

    vdir = os.path.join(FRAMES_ROOT, args.video)
    files = sorted([f for f in os.listdir(vdir) if f.endswith(".jpg")])

    red_zone_areas = {}   # pos -> [area,...] in red zone
    green_zone_green = {} # pos -> count green in green zone
    rz_n = gz_n = 0
    for i, fn in enumerate(files):
        if i % args.stride != 0:
            continue
        idx = int(fn.split("_")[1].split(".")[0])
        t = idx / args.fps
        fr = robust_imread(os.path.join(vdir, fn))
        if fr is None:
            continue
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        if t < args.green_from:
            rz_n += 1
            for p, _, a in frame_blobs(hsv, [(0, 12), (168, 179)]):
                red_zone_areas.setdefault(p, []).append(a)
        else:
            gz_n += 1
            for p, _, _ in frame_blobs(hsv, [(38, 82)]):
                green_zone_green[p] = green_zone_green.get(p, 0) + 1

    print(f"{args.video}: 红段帧={rz_n} 绿段帧={gz_n}")
    # 红段稳定红: 出现率>=0.6 且 面积变异系数小
    cands = []
    for p, areas in red_zone_areas.items():
        occ = len(areas) / rz_n
        if occ < 0.6:
            continue
        med = float(np.median(areas))
        if med < 15 or med > 800:
            continue
        cv = float(np.std(areas) / med) if med else 9
        if cv > 0.6:
            continue
        gz = green_zone_green.get(p, 0)
        cands.append((p, occ, int(med), round(cv, 2), gz))
    cands.sort(key=lambda x: (-x[4], -x[1], x[3]))
    print(f"\n=== 红段稳定红 + 绿段是否转绿 (真信号候选) ===")
    print(f"{'pos':<14}{'occ':>6}{'medArea':>9}{'cv':>6}{'gzGreen':>8}")
    for p, occ, med, cv, gz in cands[:15]:
        print(f"  ({p[0]:.3f},{p[1]:.3f})  {occ:6.2f}  {med:8d}  {cv:6.2f}  {gz:7d}")


if __name__ == "__main__":
    main()
