#!/usr/bin/env python3
"""09 prior 直采路径精确模拟(复刻 _sample_roi + _sample_prior_color 含扩展)。

目的: 在生产实际消费的 prior 直采路径上, 扫 ROI 尺寸, 找能干净读出
GT[0-10]红 / [11-72]绿 / [72-106.4]unknown 的 (cx,cy,roi_px)。

GT (events.csv, 违章09):
  red   [0, 10]
  green [11, 72]
  unknown [72, 106.4]

用法:
  python scripts/diag_09_prior_sim.py <video.mp4> [--gt events.csv]
"""
import sys, os, json, argparse
import numpy as np
import cv2


def sample_roi(frame, px, py, roi_px):
    """复刻 _sample_roi: 返回 (color, g_n, r_n)。color in green/red/None。"""
    h, w = frame.shape[:2]
    cx_i, cy_i = int(px * w), int(py * h)
    x1 = max(0, cx_i - roi_px // 2)
    y1 = max(0, cy_i - roi_px // 2)
    x2 = min(w, cx_i + roi_px // 2)
    y2 = min(h, cy_i + roi_px // 2)
    if x2 <= x1 or y2 <= y1:
        return None, 0, 0
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0, 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g_mask = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([95, 255, 255]))
    r1 = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([12, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([158, 60, 40]), np.array([180, 255, 255]))
    r_mask = r1 | r2
    g_n = int(cv2.countNonZero(g_mask))
    r_n = int(cv2.countNonZero(r_mask))
    total = (x2 - x1) * (y2 - y1)
    if total == 0:
        return None, 0, 0
    g_frac, r_frac = g_n / total, r_n / total
    min_frac = 0.002
    if g_frac < min_frac and r_frac < min_frac:
        return None, 0, 0
    if g_frac > r_frac * 1.3:
        return "green", g_n, r_n
    if r_frac > g_frac * 1.3:
        return "red", g_n, r_n
    return ("green" if g_frac >= r_frac else "red"), g_n, r_n


def sample_prior(frame, px, py, prior_roi_px, expand_factor=2.0):
    """复刻 _sample_prior_color(含自适应扩展)。"""
    res, gn, rn = sample_roi(frame, px, py, prior_roi_px)
    if res is not None:
        return res
    if prior_roi_px > 0 and expand_factor > 1.0:
        big = int(prior_roi_px * expand_factor)
        res, gn, rn = sample_roi(frame, px, py, big)
        if res is not None:
            return res
    return None


def load_gt(gt_csv, video):
    import csv
    segs = []  # (start, end, state)
    with open(gt_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("video", "").strip() != video:
                continue
            try:
                s = float(row["start_s"]); e = float(row["end_s"])
                st = row.get("state", "").strip().lower()
                segs.append((s, e, st))
            except Exception:
                pass
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gt", default=os.path.join(os.path.dirname(__file__), "..", "datasets", "gt", "events.csv"))
    ap.add_argument("--cx", type=float, default=0.65)
    ap.add_argument("--cy", type=float, default=0.30)
    ap.add_argument("--rois", default="20,30,40,50,60,80,100,160")
    ap.add_argument("--factor", type=float, default=2.0)
    args = ap.parse_args()

    gt_segs = load_gt(args.gt, os.path.basename(args.video).replace(".mp4", ""))
    print("GT segs:", gt_segs)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rois = [int(x) for x in args.rois.split(",")]

    # 逐帧结果缓存
    frames_pred = {r: [] for r in rois}
    t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for r in rois:
            frames_pred[r].append(sample_prior(frame, args.cx, args.cy, r, args.factor))
        t += 1.0 / fps
    cap.release()
    print("总帧数≈%.0f, 时长≈%.1fs" % (len(frames_pred[rois[0]]), t))

    # 评分
    def seg_state_at(segs, tt):
        for s, e, st in segs:
            if s - 1e-6 <= tt <= e + 1e-6:
                return st
        return "unknown"

    print("\n%-5s %10s %10s %10s %12s %14s" % ("roi", "fg_red", "fr_green", "unk_green", "green_run_s", "red_run_s@red"))
    results = {}
    for r in rois:
        preds = frames_pred[r]
        fg_red = 0        # GT red 区间被判 green
        fr_green = 0      # GT green 区间被判 red
        unk_green = 0     # GT unknown 区间被判 green
        n_red = n_green = n_unk = 0
        # 连续绿 run 统计(全视频最长)
        max_green_run = 0.0
        cur = 0.0
        for i, p in enumerate(preds):
            tt = i / fps
            st = seg_state_at(gt_segs, tt)
            if st == "red":
                n_red += 1
                if p == "green": fg_red += 1
            elif st == "green":
                n_green += 1
                if p == "red": fr_green += 1
            else:
                n_unk += 1
                if p == "green": unk_green += 1
            # run
            if p == "green":
                cur += 1.0 / fps
                max_green_run = max(max_green_run, cur)
            else:
                cur = 0.0
        # GT red 区间是否被正确判红(最长连续 red run 覆盖 red 区间)
        red_run_in_red = 0.0
        cur = 0.0
        for i, p in enumerate(preds):
            tt = i / fps
            st = seg_state_at(gt_segs, tt)
            if st == "red":
                if p == "red":
                    cur += 1.0 / fps
                    red_run_in_red = max(red_run_in_red, cur)
                else:
                    cur = 0.0
            else:
                cur = 0.0
        results[r] = dict(fg_red=fg_red, fr_green=fr_green, unk_green=unk_green,
                          n_red=n_red, n_green=n_green, n_unk=n_unk,
                          max_green_run=max_green_run, red_run_in_red=red_run_in_red)
        print("%-5d %10d %10d %10d %12.1f %14.1f" % (
            r, fg_red, fr_green, unk_green, max_green_run, red_run_in_red))

    # 选最优: fg_red==0 且 fr_green 最小 且 max_green_run 尽量覆盖绿窗(>50s), 取最小 roi
    best = None
    for r in rois:
        x = results[r]
        if x["fg_red"] == 0 and x["max_green_run"] >= 50.0:
            if best is None or r < best:
                best = r
    print("\n候选最优 roi(红区零假绿 & 绿run≥50s & 最小roi) =", best)
    if best is not None:
        print("  该 roi 详情:", json.dumps(results[best], ensure_ascii=False))


if __name__ == "__main__":
    main()
