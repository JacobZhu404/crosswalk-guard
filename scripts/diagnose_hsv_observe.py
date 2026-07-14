#!/usr/bin/env python3
"""诊断 HSV observe() 在违章04 首段 red 被判 green 的根因。

对比:
- observe(): 无状态, 全局亮斑面积加总 -> 易受环境绿/树叶干扰
- detect(prior): 先验锁定行人信号位置 -> 只统计 ROI 内颜色

输出: 关键帧的诊断图(原图 + HSV掩码 + candidates + 先验ROI)
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


def draw_candidates(frame, spots, prior=None):
    """在图上画出所有 candidates 和先验位置。"""
    vis = frame.copy()
    h, w = vis.shape[:2]
    for s in spots:
        x1, y1, x2, y2 = s["box"]
        color = (0, 255, 0) if s["color"] == "green" else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cx_i, cy_i = int(s["cx"] * w), int(s["cy"] * h)
        cv2.circle(vis, (cx_i, cy_i), 2, color, -1)
        cv2.putText(vis, f"{s['color'][0]}:{s['area']}", (x1, max(y1 - 2, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    if prior:
        px, py, roi_px = prior
        cx_i, cy_i = int(px * w), int(py * h)
        r = roi_px // 2
        cv2.rectangle(vis, (cx_i - r, cy_i - r), (cx_i + r, cy_i + r), (255, 0, 0), 2)
        cv2.circle(vis, (cx_i, cy_i), 3, (255, 0, 0), -1)
        cv2.putText(vis, "PRIOR", (cx_i - r, cy_i - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return vis


def hsv_masks(frame, prior=None):
    """返回 HSV 分割掩码图(绿/红/亮斑)。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lit = cv2.inRange(hsv, np.array([0, 130, 60]), np.array([180, 255, 255]))
    g_mask = cv2.inRange(hsv, np.array([40, 22, 0]), np.array([100, 255, 255]))
    r1 = cv2.inRange(hsv, np.array([0, 22, 0]), np.array([35, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([150, 22, 0]), np.array([180, 255, 255]))
    r_mask = r1 | r2

    # 如果给了先验，也画直采ROI的掩码
    if prior:
        h, w = frame.shape[:2]
        px, py, roi_px = prior
        cx_i, cy_i = int(px * w), int(py * h)
        r = roi_px // 2
        x1, y1 = max(0, cx_i - r), max(0, cy_i - r)
        x2, y2 = min(w, cx_i + r), min(h, cy_i + r)
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            g_roi = cv2.inRange(hsv_roi, np.array([35, 60, 40]), np.array([95, 255, 255]))
            r1_roi = cv2.inRange(hsv_roi, np.array([0, 60, 40]), np.array([12, 255, 255]))
            r2_roi = cv2.inRange(hsv_roi, np.array([158, 60, 40]), np.array([180, 255, 255]))
            r_roi = r1_roi | r2_roi
            return lit, g_mask, r_mask, (g_roi, r_roi, (x1, y1, x2, y2))
    return lit, g_mask, r_mask, None


def main():
    video = "违章04"
    frames_dir = os.path.join(ROOT, "datasets", "frames")
    out_dir = os.path.join(ROOT, "data", "output", "hsv_diag", video)
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    dataset = FrameDataset(frames_dir)

    # 先验
    with open(os.path.join(ROOT, "configs", "light_priors.json"), encoding="utf-8") as f:
        priors = json.load(f)
    prior = priors.get(video)  # [cx, cy, roi_px]

    # 两个检测器: observe(无先验) vs detect(有先验)
    det_observe = TrafficLightDetector(cfg, verbose=False)
    det_prior = TrafficLightDetector(cfg, verbose=False)
    if prior:
        det_prior.signal_prior = (float(prior[0]), float(prior[1]))
        det_prior.prior_roi_px = int(prior[2])

    # 取前60帧(约0-14s, 覆盖首段red) + 中间几帧做对比
    frames = list(dataset.iter_video(video))
    key_indices = list(range(0, min(60, len(frames)), 5))  # 每5帧采样
    if len(frames) > 100:
        key_indices += list(range(80, min(140, len(frames)), 10))

    print(f"=== HSV 诊断: {video} ===")
    print(f"总帧数={len(frames)}  诊断帧数={len(key_indices)}  先验={prior}")
    print(f"{'idx':>5} {'ts':>6} {'obs':>6} {'det':>6} {'g_area':>8} {'r_area':>8} {'n_cand':>7} {'top_cands':>40}")
    print("-" * 100)

    for ki in key_indices:
        idx, ts, frame = frames[ki]
        if frame is None:
            continue

        # observe() 输出
        res_ob = det_observe.observe(frame)
        obs = res_ob["obs"]
        spots = res_ob.get("candidates", [])
        g_area = sum(s["area"] for s in spots if s["color"] == "green")
        r_area = sum(s["area"] for s in spots if s["color"] == "red")

        # detect() 输出(先验模式)
        res_pr = det_prior.detect(frame)
        det_state = res_pr["state"]

        # top candidates 摘要
        top = sorted(spots, key=lambda s: -s["area"])[:4]
        top_s = " ".join(f"{s['color'][0]}({s['area']},{s['cx']:.2f},{s['cy']:.2f})" for s in top)

        marker = " <<<" if obs != "red" and ts < 42 else ""
        print(f"{idx:>5} {ts:>6.1f} {obs:>6} {det_state:>6} {g_area:>8} {r_area:>8} {len(spots):>7} {top_s:>40}{marker}")

        # 可视化落盘
        vis = draw_candidates(frame, spots, prior)
        lit, g_mask, r_mask, roi_masks = hsv_masks(frame, prior)

        # 拼成一张大图: [原图+候选框 | 亮斑掩码 | 绿掩码 | 红掩码]
        h, w = frame.shape[:2]
        lit_c = cv2.cvtColor(lit, cv2.COLOR_GRAY2BGR)
        g_c = cv2.cvtColor(g_mask, cv2.COLOR_GRAY2BGR)
        r_c = cv2.cvtColor(r_mask, cv2.COLOR_GRAY2BGR)

        # 如果 ROI 掩码存在，拼进去
        if roi_masks:
            g_roi, r_roi, (x1, y1, x2, y2) = roi_masks
            g_roi_c = cv2.cvtColor(g_roi, cv2.COLOR_GRAY2BGR)
            r_roi_c = cv2.cvtColor(r_roi, cv2.COLOR_GRAY2BGR)
            # 创建全图大小的 ROI 掩码图(其余黑)
            g_roi_full = np.zeros_like(g_c)
            r_roi_full = np.zeros_like(r_c)
            g_roi_full[y1:y2, x1:x2] = g_roi_c
            r_roi_full[y1:y2, x1:x2] = r_roi_c
            row1 = np.hstack([vis, lit_c, g_c, r_c])
            row2 = np.hstack([
                np.zeros_like(vis),
                np.zeros_like(lit_c),
                g_roi_full,
                r_roi_full,
            ])
            # 标注
            cv2.putText(row2, "PRIOR-G", (w * 2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(row2, "PRIOR-R", (w * 3 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            composite = np.vstack([row1, row2])
        else:
            composite = np.hstack([vis, lit_c, g_c, r_c])

        # 顶部信息条
        bar = np.zeros((30, composite.shape[1], 3), dtype=np.uint8)
        text = (f"idx={idx} t={ts:.1f}s  observe={obs}  detect={det_state}  "
                f"g_area={g_area} r_area={r_area}  n_cand={len(spots)}")
        cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        composite = np.vstack([bar, composite])

        out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, composite)

    print(f"\n诊断图已保存: {out_dir}/")
    print(f"每行格式: idx timestamp observe detect green_area red_area n_candidates top_spots")


if __name__ == "__main__":
    main()
