#!/usr/bin/env python3
"""09 prior 直采路径 2D 栅格扫描: 在生产实际消费的 prior 直采路径上,
扫 (cx,cy), 找能干净读出 GT[0-10]red / [11-72]green 的位置。

GT (events.csv light_state 列):
  red   [0, 10]
  green [11, 72]
  unknown [72, 106.4]

评分(越高越好):
  g_in_green = [11,72] 帧判绿比例 (want ~1)
  no_g_in_red = 1 - ([0,10] 帧判绿比例) (want ~1)
  no_r_in_green = 1 - ([11,72] 帧判红比例) (want ~1)
  score = min(g_in_green, no_g_in_red, no_r_in_green)  # 短板
"""
import sys, os, csv, argparse
import numpy as np
import cv2


def sample_roi(frame, px, py, roi_px):
    h, w = frame.shape[:2]
    cx_i, cy_i = int(px * w), int(py * h)
    x1 = max(0, cx_i - roi_px // 2); y1 = max(0, cy_i - roi_px // 2)
    x2 = min(w, cx_i + roi_px // 2); y2 = min(h, cy_i + roi_px // 2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([95, 255, 255]))
    r1 = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([12, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([158, 60, 40]), np.array([180, 255, 255]))
    r = r1 | r2
    gn = int(cv2.countNonZero(g)); rn = int(cv2.countNonZero(r))
    total = (x2 - x1) * (y2 - y1)
    if total == 0:
        return None
    gf, rf = gn / total, rn / total
    if gf < 0.002 and rf < 0.002:
        return None
    if gf > rf * 1.3:
        return "green"
    if rf > gf * 1.3:
        return "red"
    return "green" if gf >= rf else "red"


def load_gt(gt_csv, video):
    segs = []
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            if row.get("video", "").strip() != video:
                continue
            try:
                s = float(row["start_s"]); e = float(row["end_s"])
                st = row.get("light_state", "").strip().lower()
                segs.append((s, e, st))
            except Exception:
                pass
    return segs


def seg_state_at(segs, tt):
    for s, e, st in segs:
        if s - 1e-6 <= tt <= e + 1e-6:
            return st
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gt", default=os.path.join(os.path.dirname(__file__), "..", "datasets", "gt", "events.csv"))
    ap.add_argument("--roi", type=int, default=40)
    ap.add_argument("--cx0", type=float, default=0.30)
    ap.add_argument("--cx1", type=float, default=0.85)
    ap.add_argument("--cy0", type=float, default=0.10)
    ap.add_argument("--cy1", type=float, default=0.60)
    ap.add_argument("--step", type=float, default=0.025)
    args = ap.parse_args()

    video = os.path.basename(args.video).replace(".mp4", "")
    gt = load_gt(args.gt, video)
    print("GT:", gt)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    print("帧数=%d 时长≈%.1fs fps=%.1f" % (n, n / fps, fps))

    cxs = np.arange(args.cx0, args.cx1 + 1e-9, args.step)
    cys = np.arange(args.cy0, args.cy1 + 1e-9, args.step)
    best = []
    for cy in cys:
        for cx in cxs:
            g_in_red = g_in_green = r_in_green = 0.0
            n_red = n_green = 0
            for i, fr in enumerate(frames):
                st = seg_state_at(gt, i / fps)
                p = sample_roi(fr, cx, cy, args.roi)
                if st == "red":
                    n_red += 1
                    if p == "green": g_in_red += 1
                elif st == "green":
                    n_green += 1
                    if p == "green": g_in_green += 1
                    elif p == "red": r_in_green += 1
            g_in_red_f = g_in_red / n_red if n_red else 0
            g_in_green_f = g_in_green / n_green if n_green else 0
            r_in_green_f = r_in_green / n_green if n_green else 0
            score = min(g_in_green_f, 1 - g_in_red_f, 1 - r_in_green_f)
            best.append((score, cx, cy, g_in_green_f, g_in_red_f, r_in_green_f))
    best.sort(reverse=True)
    print("\nTop 12 (score, cx, cy, g_in_green, g_in_red, r_in_green):")
    for b in best[:12]:
        print("  %.3f  cx=%.3f cy=%.3f  g@green=%.2f g@red=%.2f r@green=%.2f" % b)


if __name__ == "__main__":
    main()
