"""给定多个候选 prior, 用 HSV 直采(不依赖聚类)测每帧绿/红像素数,
挑出"红段常红、绿段转绿"的那个 = 行人信号位置.

用法:
  python scripts/test_prior.py 违章04 --roi 200 \
      --priors "0.18,0.24 0.18,0.27 0.06,0.36 0.21,0.27 0.55,0.10"
"""
import sys, os, argparse
import numpy as np
import cv2

FRAMES_ROOT = os.path.join("datasets", "frames")


def robust_imread(path):
    with open(path, "rb") as f:
        buf = f.read()
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def sample_color(fr, cx, cy, roi):
    H, W = fr.shape[:2]
    px, py = int(cx * W), int(cy * H)
    x0, x1 = max(0, px - roi // 2), min(W, px + roi // 2)
    y0, y1 = max(0, py - roi // 2), min(H, py + roi // 2)
    if x1 <= x0 or y1 <= y0:
        return 0, 0
    patch = fr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168)) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 70) & (hsv[:, :, 2] <= 235)
    grn = (hsv[:, :, 0] >= 38) & (hsv[:, :, 0] <= 82) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 70) & (hsv[:, :, 2] <= 235)
    return int(grn.sum()), int(red.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--priors", required=True, help="空格分隔的 cx,cy 对")
    ap.add_argument("--roi", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    priors = [tuple(map(float, p.split(","))) for p in args.priors.split()]
    vdir = os.path.join(FRAMES_ROOT, args.video)
    files = sorted([f for f in os.listdir(vdir) if f.endswith(".jpg")])

    # 预取每帧的 (t, g[r], r[r])
    timelines = {p: [] for p in priors}
    for i, fn in enumerate(files):
        if i % args.stride != 0:
            continue
        idx = int(fn.split("_")[1].split(".")[0])
        fr = robust_imread(os.path.join(vdir, fn))
        if fr is None:
            continue
        for p in priors:
            g, r = sample_color(fr, p[0], p[1], args.roi)
            timelines[p].append((idx, g, r))

    fps = 29.7
    for p in priors:
        tl = timelines[p]
        if not tl:
            continue
        red_zone = [(idx, g, r) for idx, g, r in tl if idx / fps < 40]
        grn_zone = [(idx, g, r) for idx, g, r in tl if idx / fps >= 41]
        rz_r = max([r for _, _, r in red_zone], default=0)
        rz_g = max([g for _, g, _ in red_zone], default=0)
        gz_g = max([g for _, g, _ in grn_zone], default=0)
        gz_r = max([r for _, _, r in grn_zone], default=0)
        print(f"prior=({p[0]:.2f},{p[1]:.2f}): "
              f"红段 maxR={rz_r:4d} maxG={rz_g:4d} | 绿段 maxG={gz_g:4d} maxR={gz_r:4d}")
        # 打印首尾各 2 帧采样
        for idx, g, r in tl[:2] + tl[-2:]:
            print(f"    t={idx/fps:5.1f}s  g={g:4d} r={r:4d}")


if __name__ == "__main__":
    main()
