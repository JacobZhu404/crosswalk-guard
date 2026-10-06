#!/usr/bin/env python3
"""09 prior 细网格精修: 围绕候选中心, 按各 GT 窗口连续绿 run 选最优 prior。

目标: [11,72] 绿窗出现尽可能长(整段)的绿 run; [0,10] 红窗 与 [72,106.4] 未知窗
的连续绿 run 都 < 6s(否则 T=6 会 confirmed 成 FP)。
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


def max_run_in_window(preds, fps, segs, w0, w1, want="green"):
    """窗口 [w0,w1] 内连续 want 的最大时长。"""
    best = 0.0
    cur = 0.0
    for i, p in enumerate(preds):
        tt = i / fps
        if w0 - 1e-6 <= tt <= w1 + 1e-6:
            if p == want:
                cur += 1.0 / fps
                best = max(best, cur)
            else:
                cur = 0.0
        else:
            cur = 0.0
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gt", default=os.path.join(os.path.dirname(__file__), "..", "datasets", "gt", "events.csv"))
    ap.add_argument("--roi", type=int, default=40)
    ap.add_argument("--cx0", type=float, default=0.50)
    ap.add_argument("--cx1", type=float, default=0.60)
    ap.add_argument("--cy0", type=float, default=0.06)
    ap.add_argument("--cy1", type=float, default=0.16)
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()

    video = os.path.basename(args.video).replace(".mp4", "")
    gt = load_gt(args.gt, video)
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

    cxs = np.arange(args.cx0, args.cx1 + 1e-9, args.step)
    cys = np.arange(args.cy0, args.cy1 + 1e-9, args.step)
    results = []
    for cy in cys:
        for cx in cxs:
            preds = [sample_roi(fr, cx, cy, args.roi) for fr in frames]
            run_green_win = max_run_in_window(preds, fps, gt, 11.0, 72.0, "green")
            run_red_win = max_run_in_window(preds, fps, gt, 0.0, 10.0, "green")
            run_unk_win = max_run_in_window(preds, fps, gt, 72.0, 106.4, "green")
            # 优选: 绿窗 run 长, 且红窗/未知窗绿 run < 6s
            safe = (run_red_win < 6.0) and (run_unk_win < 6.0)
            results.append((run_green_win, cx, cy, run_red_win, run_unk_win, safe))
    # 排序: 优先 safe, 再绿窗 run 长
    results.sort(key=lambda x: (x[5], x[0]), reverse=True)
    print("Top 12 (green_win_run, cx, cy, red_win_green_run, unk_win_green_run, safe):")
    for r in results[:12]:
        print("  green_win=%.1f cx=%.3f cy=%.3f red_win=%.2f unk_win=%.2f safe=%s" % r)


if __name__ == "__main__":
    main()
