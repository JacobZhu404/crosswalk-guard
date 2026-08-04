"""qw 接线护栏③合成漂移探针: 测 v2 running-max 在相机漂移下的行为(拖影失效模式)。

cc gate(2026-08-04)必含**单向 pan**档——running-max 的失效模式是"持续漂移拖影"
(mask 面积单调膨胀); ±1-3% 对称抖动像素几乎不净移动, 测不出拖影。

三档(对同一视频):
  - static : 原帧直喂(detector 基线行为)
  - jitter : 每帧随机仿射 ±1~1.5% 平移(帧宽比例) ±0.5° 旋转 ±1% 缩放(净位移≈0)
  - pan    : 每帧固定增量单向平移(累计净漂移达帧宽 10-15%)+ 缓慢单向缩放, 模拟镜头漂移

口径: detector 层直测(不经 cli.run), 按生产节奏每 interval=round(fps/8) 原始帧调一次
v2.detect(与 dag crosswalk_interval=4 的生产节奏同构)。测 mask 面积/质心时间序列:
  - 拖影 = mask 面积随帧单调膨胀(末面积/首面积 >> 1 且远大于 static 档)
  - 质心漂移 = 累积 mask 质心偏离 static 档质心的像素位移
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.crosswalk_v2 import CrosswalkDetectorV2

# 代表视频: 02(mid IoU)/06(高 IoU)/08(中高), mask-IoU 覆盖高中低(见 before/after 报告)
DEFAULT_VIDEOS = ["违章02", "违章06", "违章08"]


def affine_jitter(w, h, rng, seed_base):
    """对称抖动仿射矩阵(净位移≈0): 平移±1.5%帧宽, 旋转±0.5°, 缩放±1%。"""
    dx = rng.uniform(-0.015 * w, 0.015 * w)
    dy = rng.uniform(-0.015 * h, 0.015 * h)
    ang = rng.uniform(-0.5, 0.5)
    s = rng.uniform(0.99, 1.01)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def affine_pan(frame_idx, w, h, total_steps):
    """单向 pan: 每帧平移增量使累计净漂移达帧宽的 12%(约 0.12*w/total_steps 每步)。
    加 0.25% 每步的单向缩放(镜头缓慢推进), 模拟持续漂移。"""
    step_frac = 0.12 / max(total_steps, 1)
    dx = step_frac * w * frame_idx
    dy = step_frac * 0.4 * h * frame_idx
    s = 1.0 + 0.0025 * frame_idx
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, s)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def mask_stats(mask):
    if mask is None:
        return 0.0, None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, None
    return float(len(xs)), (float(xs.mean()), float(ys.mean()))


def probe_video(video, mode, cfg, seed=42):
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, int(round(fps / cfg.inference.fps)))

    det = CrosswalkDetectorV2(cfg)
    rng = random.Random(seed)

    areas, centers = [], []
    frame_idx = 0
    detect_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            if mode == "jitter":
                M = affine_jitter(W, H, rng, seed)
                fw = cv2.warpAffine(frame, M, (W, H))
            elif mode == "pan":
                total_steps = max(1, int(detect_count + 1))
                M = affine_pan(frame_idx // interval, W, H, total_steps)
                fw = cv2.warpAffine(frame, M, (W, H))
            else:
                fw = frame
            mask = det.detect(fw)
            a, c = mask_stats(mask)
            areas.append(a)
            centers.append(c)
            detect_count += 1
        frame_idx += 1
    cap.release()
    return {"video": video, "mode": mode, "detects": detect_count, "areas": areas, "centers": centers}


def summarize(r, verbose=False):
    areas = r["areas"]
    first = areas[0] if areas else 0.0
    peak = max(areas) if areas else 0.0
    last = areas[-1] if areas else 0.0
    mean = float(np.mean(areas)) if areas else 0.0
    # 拖影系数: 末面积/首面积(持续漂移拖影 -> 单调膨胀 -> >>1)
    trail = (last / first) if first > 0 else float("nan")
    # 质心漂移: 累积过程质心相对首帧质心的最大位移(px)
    c0 = r["centers"][0] if r["centers"] else None
    drift = 0.0
    if c0 is not None:
        for c in r["centers"]:
            if c is None:
                continue
            drift = max(drift, float(np.hypot(c[0] - c0[0], c[1] - c0[1])))
    return {"first": first, "peak": peak, "last": last, "mean": mean,
            "trail": trail, "drift_px": drift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    ap.add_argument("--modes", nargs="+", default=["static", "jitter", "pan"])
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"{'视频':<6} {'档位':<7} {'首面积':>9} {'末面积':>9} {'峰值':>9} {'均值':>9} {'拖影(末/首)':>12} {'质心漂移px':>10}")
    rows = {}
    for v in args.videos:
        rows[v] = {}
        for m in args.modes:
            r = probe_video(v, m, cfg)
            if r is None:
                print(f"{v:<6} {m:<7} 视频缺失")
                continue
            s = summarize(r)
            rows[v][m] = s
            print(f"{v:<6} {m:<7} {s['first']:>9.0f} {s['last']:>9.0f} {s['peak']:>9.0f} "
                  f"{s['mean']:>9.0f} {s['trail']:>12.2f} {s['drift_px']:>10.1f}")
            if args.verbose and m == "pan":
                print(f"    pan 面积序列(每 20 个 detect 采样): "
                      f"{[int(a) for a in r['areas'][::20][:10]]}...")
    # 交叉对照: pan/jitter vs static 的末面积比
    print("\n=== 对照(相对 static) ===")
    for v in args.videos:
        if "static" not in rows.get(v, {}):
            continue
        s0 = rows[v]["static"]
        for m in ("jitter", "pan"):
            if m not in rows.get(v, {}):
                continue
            sm = rows[v][m]
            ratio = sm["last"] / s0["last"] if s0["last"] > 0 else float("nan")
            print(f"{v:<6} {m:<7} 末面积/static末面积={ratio:.2f}  "
                  f"质心漂移(相对static):{sm['drift_px'] - s0['drift_px']:.1f}px")


if __name__ == "__main__":
    main()
