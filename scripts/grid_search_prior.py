"""网格搜索最优 signal_prior (数据驱动, 用 GT 算准确率).

围绕一个基准 prior, 在 cx/cy 网格上扫描, 对每个候选跑检测器, 计算
accuracy(GT 含 tentative, 全量计入) 选最优. 用于修正"搜索区偏"而不靠肉眼猜.

用法:
  python scripts/grid_search_prior.py 违章02 --base 0.78 0.15 160
  python scripts/grid_search_prior.py 违章04 --base 0.69 0.27 220 --cx-lo 0.50 --cx-hi 0.75
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


def robust_imread(p):
    with open(p, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def eval_one(video, prior, cfg, gt_segs, frames_dir):
    files = sorted(glob.glob(os.path.join(frames_dir, video, "frame_*.jpg")))
    if not files:
        return None
    det = TrafficLightDetector(cfg, verbose=False)
    det.signal_prior = (prior[0], prior[1])
    det.prior_roi_px = int(prior[2])
    cx, cy, roi = prior
    ok = tot = 0
    for fp in files:
        idx = int(os.path.basename(fp)[6:-4])
        fr = robust_imread(fp)
        if fr is None:
            continue
        # 时间从文件名近似: 抽帧 step 固定, 用相邻帧序推算秒不可靠, 这里用 manifest 优先
        t = _t_from_idx.get((video, idx))
        if t is None:
            t = 0.0
        res = det.detect(fr)
        st = res.get("state", "unknown")
        gt = gt_at(gt_segs, t)
        tot += 1
        if st == gt:
            ok += 1
    return ok / tot if tot else None


def main():
    global _t_from_idx
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--base", nargs=3, type=float, required=True)
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--cx-lo", type=float, default=None)
    ap.add_argument("--cx-hi", type=float, default=None)
    ap.add_argument("--cy-lo", type=float, default=None)
    ap.add_argument("--cy-hi", type=float, default=None)
    ap.add_argument("--step", type=float, default=0.03)
    ap.add_argument("--roi", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    gt = load_gt(args.gt)
    if args.video not in gt:
        print(f"{args.video} 无 GT, 退出"); return
    segs = gt[args.video]

    # manifest: (video,idx)->t
    _t_from_idx = {}
    mp = os.path.join(args.frames_dir, "manifest.csv")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    _t_from_idx[(r["video"], int(r["frame_idx"]))] = float(r["timestamp"])
                except (KeyError, ValueError):
                    pass

    bx, by, broi = args.base
    roi = args.roi if args.roi else int(broi)
    cx_lo = args.cx_lo if args.cx_lo is not None else bx - 0.12
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

    print(f"网格搜索 {args.video}: {len(cands)} 候选 (cx {cx_lo:.2f}-{cx_hi:.2f}, cy {cy_lo:.2f}-{cy_hi:.2f}, roi={roi})")
    results = []
    for i, c in enumerate(cands):
        acc = eval_one(args.video, c, cfg, segs, args.frames_dir)
        results.append((acc if acc is not None else 0.0, c))
        if (i + 1) % 10 == 0:
            print(f"  进度 {i+1}/{len(cands)}")
    results.sort(key=lambda r: -r[0])
    print(f"\n=== {args.video} Top8 (acc, prior) ===")
    for acc, c in results[:8]:
        mark = " <== 当前base" if list(c) == [round(bx,3),round(by,3),roi] else ""
        print(f"  acc={acc*100:5.1f}%  prior=[{c[0]},{c[1]},{c[2]}]{mark}")


if __name__ == "__main__":
    main()
