#!/usr/bin/env python3
"""精细诊断违章03首段: 先验ROI内到底是什么颜色。

对比两种阈值:
- 严格(_candidates): S>=130, V>=60  (lit mask)
- 宽松(_sample_prior_color): S>=60, V>=40 (g_mask/r_mask)

输出每帧的 ROI 裁剪图 + 掩码 + 统计数字。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np
import json

from redlight.infrastructure.config import load_config
from redlight.models.traffic_light import TrafficLightDetector
from redlight.evaluation.frame_dataset import FrameDataset


def analyze_prior_roi(frame, prior, frame_idx=0):
    """分析先验ROI内的HSV统计，返回详细数字和可视化图。"""
    h, w = frame.shape[:2]
    px, py, roi_px = prior
    cx_i, cy_i = int(px * w), int(py * h)
    r = roi_px // 2
    x1, y1 = max(0, cx_i - r), max(0, cy_i - r)
    x2, y2 = min(w, cx_i + r), min(h, cy_i + r)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 严格阈值 (_candidates 用的 lit mask)
    lit_strict = cv2.inRange(hsv_roi, np.array([0, 130, 60]), np.array([180, 255, 255]))
    g_strict = cv2.inRange(hsv_roi, np.array([40, 22, 0]), np.array([100, 255, 255]))
    r1_strict = cv2.inRange(hsv_roi, np.array([0, 22, 0]), np.array([35, 255, 255]))
    r2_strict = cv2.inRange(hsv_roi, np.array([150, 22, 0]), np.array([180, 255, 255]))
    r_strict = r1_strict | r2_strict

    # 宽松阈值 (_sample_prior_color 用的)
    g_loose = cv2.inRange(hsv_roi, np.array([35, 60, 40]), np.array([95, 255, 255]))
    r1_loose = cv2.inRange(hsv_roi, np.array([0, 60, 40]), np.array([12, 255, 255]))
    r2_loose = cv2.inRange(hsv_roi, np.array([158, 60, 40]), np.array([180, 255, 255]))
    r_loose = r1_loose | r2_loose

    total_px = roi.shape[0] * roi.shape[1]

    stats = {
        "frame_idx": frame_idx,
        "total_px": total_px,
        "strict": {
            "lit": int(cv2.countNonZero(lit_strict)),
            "green": int(cv2.countNonZero(g_strict & lit_strict)),
            "red": int(cv2.countNonZero(r_strict & lit_strict)),
        },
        "loose": {
            "green": int(cv2.countNonZero(g_loose)),
            "red": int(cv2.countNonZero(r_loose)),
        },
        "roi_coords": (x1, y1, x2, y2),
    }

    # 可视化
    vis = frame.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.circle(vis, (cx_i, cy_i), 3, (255, 0, 0), -1)

    # 拼 ROI 细节图
    roi_vis = roi.copy()
    g_c = cv2.cvtColor(g_loose, cv2.COLOR_GRAY2BGR)
    r_c = cv2.cvtColor(r_loose, cv2.COLOR_GRAY2BGR)
    lit_c = cv2.cvtColor(lit_strict, cv2.COLOR_GRAY2BGR)
    g_strict_c = cv2.cvtColor(g_strict & lit_strict, cv2.COLOR_GRAY2BGR)
    r_strict_c = cv2.cvtColor(r_strict & lit_strict, cv2.COLOR_GRAY2BGR)

    # 统一缩放到相同高度
    target_h = 160
    def resize(img):
        if img.shape[0] == 0:
            return np.zeros((target_h, target_h, 3), dtype=np.uint8)
        scale = target_h / img.shape[0]
        new_w = max(1, int(img.shape[1] * scale))
        return cv2.resize(img, (new_w, target_h))

    row1 = np.hstack([resize(roi_vis), resize(g_c), resize(r_c)])
    row2 = np.hstack([resize(lit_c), resize(g_strict_c), resize(r_strict_c)])

    # 标注
    bar1 = np.zeros((25, row1.shape[1], 3), dtype=np.uint8)
    texts1 = ["ROI", "Loose-G", "Loose-R"]
    offsets1 = [0]
    for i in range(1, 3):
        offsets1.append(offsets1[-1] + resize(roi_vis).shape[1] if i == 1 else offsets1[-1] + resize(g_c).shape[1])
    # 简化：在每张图上直接写标签
    def label(img, text, color=(255,255,255)):
        h, w = img.shape[:2]
        cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return img

    row1 = np.hstack([label(resize(roi_vis), "ROI"),
                      label(resize(g_c), "Loose-G", (0,255,0)),
                      label(resize(r_c), "Loose-R", (0,0,255))])
    row2 = np.hstack([label(resize(lit_c), "Strict-lit"),
                      label(resize(g_strict_c), "Strict-G", (0,255,0)),
                      label(resize(r_strict_c), "Strict-R", (0,0,255))])

    detail = np.vstack([row1, row2])

    # 顶部信息条
    info_h = 35
    info = np.zeros((info_h, detail.shape[1], 3), dtype=np.uint8)
    g_l, r_l = stats["loose"]["green"], stats["loose"]["red"]
    g_s, r_s = stats["strict"]["green"], stats["strict"]["red"]
    text = (f"idx={frame_idx}  loose(g={g_l},r={r_l})  strict(g={g_s},r={r_s})  "
            f"sampled={'green' if g_l > r_l * 1.3 else 'red' if r_l > g_l * 1.3 else 'tie'}")
    cv2.putText(info, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    composite = np.vstack([info, detail])
    return stats, composite, vis


def main():
    video = "违章03"
    frames_dir = os.path.join(ROOT, "datasets", "frames")
    out_dir = os.path.join(ROOT, "data", "output", "hsv_diag", video + "_prior")
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    dataset = FrameDataset(frames_dir)

    with open(os.path.join(ROOT, "configs", "light_priors.json"), encoding="utf-8") as f:
        priors = json.load(f)
    prior = priors.get(video)  # [cx, cy, roi_px]

    det = TrafficLightDetector(cfg, verbose=False)
    if prior:
        det.signal_prior = (float(prior[0]), float(prior[1]))
        det.prior_roi_px = int(prior[2])

    frames = list(dataset.iter_video(video))
    # 诊断前30帧(约0-7s) + 变色点附近
    key_indices = list(range(0, min(30, len(frames)), 2))  # 更密采样
    if len(frames) > 300:
        key_indices += [280, 290, 300, 310, 320]  # ~65-67s 变色点

    print(f"=== 违章03 先验ROI精细诊断 ===")
    print(f"先验={prior}  诊断帧数={len(key_indices)}")
    print(f"{'idx':>5} {'ts':>6} {'obs':>5} {'g_loose':>8} {'r_loose':>8} {'g_strict':>9} {'r_strict':>9} {'sampled':>8}")
    print("-" * 80)

    all_stats = []
    for ki in key_indices:
        idx, ts, frame = frames[ki]
        if frame is None:
            continue

        res = det.observe(frame)
        obs = res["obs"]

        result = analyze_prior_roi(frame, prior, idx)
        if result is None:
            continue
        stats, detail, vis = result
        all_stats.append(stats)

        g_l, r_l = stats["loose"]["green"], stats["loose"]["red"]
        g_s, r_s = stats["strict"]["green"], stats["strict"]["red"]
        sampled = "green" if g_l > r_l * 1.3 else "red" if r_l > g_l * 1.3 else "tie"
        marker = " <<<" if obs != sampled else ""
        print(f"{idx:>5} {ts:>6.1f} {obs:>5} {g_l:>8} {r_l:>8} {g_s:>9} {r_s:>9} {sampled:>8}{marker}")

        out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, detail)

    print(f"\n诊断图已保存: {out_dir}/")

    # 汇总: 首段red(0-4s)的loose green/red趋势
    print("\n=== 首段 0-7s loose green/red 趋势 ===")
    for s in all_stats:
        if s["frame_idx"] < 30:
            gl, rl = s["loose"]["green"], s["loose"]["red"]
            print(f"  idx={s['frame_idx']:>3}  g={gl:>5}  r={rl:>5}  ratio_g/r={gl/max(rl,1):.2f}")


if __name__ == "__main__":
    main()
