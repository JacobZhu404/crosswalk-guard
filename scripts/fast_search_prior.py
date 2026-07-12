"""快速数据驱动 prior 搜索 (用 _sample_prior_color 直采, 绕过 tracker/blob, ~30x 快).

与 grid_search_prior 的区别: 不重跑完整 detect(), 而是对每帧直接在先验 ROI 上做
HSV 颜色统计(即 prior 模式的 _sample_prior_color 降级路径), 再用与 _state_from_global
一致的"近窗绿红多数投票"得到每帧状态. 这正捕捉"prior 中心是否罩住真灯"这一信号,
足以定位最优中心. 最终 prior 仍需用 eval_light_fast.py(完整 detect) 验证防回退.

用法:
  python scripts/fast_search_prior.py 违章02 --base 0.78 0.15 160
  python scripts/fast_search_prior.py 违章04 --base 0.69 0.27 220 --cx-lo 0.50 --cx-hi 0.75
"""
import os
import sys
import csv
import argparse
import glob

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from redlight.infrastructure.config import load_config
from redlight.models.traffic_light import TrafficLightDetector


def load_gt(gt_path):
    g = {}
    with open(gt_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            # 含 tentative(全部计入, 与 grid_search 一致); confirmed 优先但此处全要
            g.setdefault(r["video"], []).append(
                (float(r["start_s"]), float(r["end_s"]), r["state"]))
    for v in g:
        g[v].sort(key=lambda x: x[0])
    return g


def gt_at(segs, t):
    for s, e, st in segs:
        if s <= t <= e:
            return st
    return segs[-1][2] if segs else "unknown"


_FRAME_CACHE = {}


def robust_imread(p):
    if p not in _FRAME_CACHE:
        with open(p, "rb") as f:
            _FRAME_CACHE[p] = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    return _FRAME_CACHE[p]


def eval_one(video, prior, cfg, gt_segs, frame_list, tmap, det):
    cx, cy, roi = prior
    det.signal_prior = (cx, cy)
    det.prior_roi_px = int(roi)
    det.global_recent = []
    det._last_state = None
    win = []
    ok = tot = 0
    for fp in frame_list:
        idx = int(os.path.basename(fp)[6:-4])
        fr = robust_imread(fp)
        if fr is None:
            continue
        det._last_frame = fr
        st = det._sample_prior_color()
        if st is None:
            st = det._last_state  # 无有色像素 -> 保持上一帧(近似 prior_hold)
        det._last_state = st
        det.global_recent.append(st)
        win.append(st)
        if len(win) > det.window:
            win.pop(0)
        g = sum(1 for c in win if c == "green")
        r = sum(1 for c in win if c == "red")
        out = "green" if g >= r else "red"
        t = tmap.get((video, idx))
        if t is None:
            continue
        gt = gt_at(gt_segs, t)
        tot += 1
        if out == gt:
            ok += 1
    return ok / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--base", nargs=3, type=float, required=True)
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--cx-lo", type=float, default=None)
    ap.add_argument("--cx-hi", type=float, default=None)
    ap.add_argument("--cy-lo", type=float, default=None)
    ap.add_argument("--cy-hi", type=float, default=None)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--roi", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    det = TrafficLightDetector(cfg, verbose=False)
    gt = load_gt(args.gt)
    if args.video not in gt:
        print(f"{args.video} 无 GT, 退出")
        return
    segs = gt[args.video]

    # manifest: (video,idx)->t
    tmap = {}
    mp = os.path.join(args.frames_dir, "manifest.csv")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    tmap[(r["video"], int(r["frame_idx"]))] = float(r["timestamp"])
                except (KeyError, ValueError):
                    pass

    frame_list = sorted(glob.glob(os.path.join(args.frames_dir, args.video, "frame_*.jpg")))

    bx, by, broi = args.base
    roi = args.roi if args.roi else int(broi)
    cx_lo = args.cx_lo if args.cx_lo is not None else bx - 0.14
    cx_hi = args.cx_hi if args.cx_hi is not None else bx + 0.06
    cy_lo = args.cy_lo if args.cy_lo is not None else by - 0.08
    cy_hi = args.cy_hi if args.cy_hi is not None else by + 0.06

    cands = []
    cx = cx_lo
    while cx <= cx_hi + 1e-9:
        cy = cy_lo
        while cy <= cy_hi + 1e-9:
            cands.append((round(cx, 3), round(cy, 3), roi))
            cy += args.step
        cx += args.step

    print(f"快速搜索 {args.video}: {len(cands)} 候选 (cx {cx_lo:.2f}-{cx_hi:.2f}, cy {cy_lo:.2f}-{cy_hi:.2f}, roi={roi}, step={args.step})")
    results = []
    for c in cands:
        acc = eval_one(args.video, c, cfg, segs, frame_list, tmap, det)
        results.append((acc, c))
    results.sort(key=lambda r: -r[0])
    print(f"\n=== {args.video} Top10 (acc, prior) ===")
    for acc, c in results[:10]:
        mark = " <== 当前base" if list(c) == [round(bx, 3), round(by, 3), roi] else ""
        print(f"  acc={acc*100:5.1f}%  prior=[{c[0]},{c[1]},{c[2]}]{mark}")


if __name__ == "__main__":
    main()
