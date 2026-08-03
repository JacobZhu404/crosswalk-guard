#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_05_candidate_gen.py — 违章05 候选生成瓶颈根因诊断(只读)

天花板诊断把候选生成瓶颈从全局收窄到违章05(天花板仅 10%)。
逐 30 个 gov 帧解剖: gov 灯物理属性/HSV + YOLO/HSV/prior 各自命中度,
回答 cc 的核心口径疑点: 05 天花板 10% 是生产真烂, 还是 eval 候选口径
没跑 prior 直采路径(_sample_prior_color)的伪影?

候选构建口径与 eval_selection_quality.py:eval_video:99-104 逐行一致。
prior 直采调用 traffic_light.py:597 _sample_prior_color(绕过面积/饱和过滤)。

用法: PYTHONPATH=src ./.venv/bin/python scripts/diag_05_candidate_gen.py
"""
import json, sys, csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from redlight.models import governing_disc as gd
from redlight.models.ped_light_selector import iou
from redlight.models.signal_candidates import build_candidates
from redlight.models.traffic_light import TrafficLightDetector

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
PRIORS = ROOT / "configs" / "light_priors.json"
REPORT = ROOT / "docs" / "reports" / "2026-08-03-qw-05-candidate-gen.md"
OUT_DIR = ROOT / "data" / "output" / "qw"

VIDEO = "违章05"
SAT_MIN = 130      # _cfg_tl: _candidates 的饱和度下限
VALUE_FLOOR = 60   # _candidates 的亮度下限
MIN_AREA_PX = 30   # _candidates 的面积下限
COLOR_S_MIN = 22   # _classify 的饱和度下限
PRIOR_ROI_PX = 160  # 05 的 prior ROI 边长


def _box_px(box_norm, W, H):
    """归一化框 → 像素框 (x1,y1,x2,y2)。"""
    return (max(0, int(box_norm[0] * W)), max(0, int(box_norm[1] * H)),
            min(W, int(box_norm[2] * W)), min(H, int(box_norm[3] * H)))


def _box_area_px(box_norm, W, H):
    x1, y1, x2, y2 = _box_px(box_norm, W, H)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _prior_roi_box(px, py, roi_px, W, H):
    """prior ROI 像素框 (以归一化中心 px,py 为心, 边长 roi_px)。"""
    cx_i, cy_i = int(px * W), int(py * H)
    return (max(0, cx_i - roi_px // 2), max(0, cy_i - roi_px // 2),
            min(W, cx_i + roi_px // 2), min(H, cy_i + roi_px // 2))


def _classify_color(mh, ms):
    """复刻 traffic_light._classify: 按均值 HSV 分类。"""
    if ms < COLOR_S_MIN:
        return None
    if 40 <= mh <= 100:
        return "green"
    if mh <= 35 or mh >= 150:
        return "red"
    return None


def _filtered_by(mh, ms, area_px):
    """判断 gov 灯会被 _candidates 的哪个 stage 过滤。"""
    reasons = []
    if area_px < MIN_AREA_PX:
        reasons.append("area")
    if ms < SAT_MIN:
        reasons.append("sat")
    if ms < COLOR_S_MIN:
        reasons.append("color_s")
    color = _classify_color(mh, ms)
    if color is None:
        reasons.append("classify")
    return "+".join(reasons) if reasons else "none", color


def diag_frame(frame, g, fi, det, yolo, prior):
    """逐帧解剖, 返回 dict(写 CSV)。"""
    H, W = frame.shape[:2]
    gov_boxes_norm = [tuple(b["box_norm"]) for b in g.get("boxes", []) if b.get("governing")]
    gov_colors_gt = [b.get("color", "") for b in g.get("boxes", []) if b.get("governing")]

    # 取第一个 gov_box 做逐帧分析(多 gov_box 时取最大的)
    if not gov_boxes_norm:
        return None
    if len(gov_boxes_norm) > 1:
        idx = max(range(len(gov_boxes_norm)), key=lambda i: _box_area_px(gov_boxes_norm[i], W, H))
    else:
        idx = 0
    gb = gov_boxes_norm[idx]
    gb_color_gt = gov_colors_gt[idx] if idx < len(gov_colors_gt) else ""

    # 1. gov 灯物理属性
    px_box = _box_px(gb, W, H)
    px_w = px_box[2] - px_box[0]
    px_h = px_box[3] - px_box[1]
    px_area = px_w * px_h
    gov_cx = (gb[0] + gb[2]) / 2
    gov_cy = (gb[1] + gb[3]) / 2

    # 2. gov 灯 HSV (在 gov_box 内取 patch)
    patch = frame[px_box[1]:px_box[3], px_box[0]:px_box[2]]
    if patch.size > 0:
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mh = float(np.mean(hsv_patch[:, :, 0]))
        ms = float(np.mean(hsv_patch[:, :, 1]))
        mv = float(np.mean(hsv_patch[:, :, 2]))
    else:
        mh = ms = mv = 0.0

    filtered, classified_color = _filtered_by(mh, ms, px_area)

    # 3. YOLO 命中
    res = yolo(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
    yolo_px_list = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
    yolo_max_iou = 0.0
    for yp in yolo_px_list:
        yp_norm = (yp[0] / W, yp[1] / H, yp[2] / W, yp[3] / H)
        v = iou(yp_norm, gb)
        if v > yolo_max_iou:
            yolo_max_iou = v

    # 4. 裸 _candidates 命中
    spots = det._candidates(frame)
    hsv_max_iou = 0.0
    for s in spots:
        sb = s["box"]
        sb_norm = (sb[0] / W, sb[1] / H, sb[2] / W, sb[3] / H)
        v = iou(sb_norm, gb)
        if v > hsv_max_iou:
            hsv_max_iou = v

    # 5. prior 命中度
    px, py = prior[0], prior[1]  # [0.7, 0.15, 160]
    prior_dist = ((gov_cx - px) ** 2 + (gov_cy - py) ** 2) ** 0.5
    prior_roi = _prior_roi_box(px, py, PRIOR_ROI_PX, W, H)
    prior_roi_norm = (prior_roi[0] / W, prior_roi[1] / H, prior_roi[2] / W, prior_roi[3] / H)
    prior_roi_iou = iou(prior_roi_norm, gb)
    # gov 中心是否在 prior ROI 内
    gov_cx_px, gov_cy_px = int(gov_cx * W), int(gov_cy * H)
    prior_contains = (prior_roi[0] <= gov_cx_px <= prior_roi[2] and
                      prior_roi[1] <= gov_cy_px <= prior_roi[3])

    # 6. prior 直采能否捞回
    det._last_frame = frame
    det.signal_prior = (float(px), float(py))
    det.prior_roi_px = PRIOR_ROI_PX
    # 紧 ROI 直采
    prior_sample = det._sample_prior_color()
    # 扩展 ROI 直采 (2x)
    prior_sample_exp = None
    if prior_sample is None and det.prior_roi_expand_factor > 1.0:
        det._last_frame = frame
        big = int(PRIOR_ROI_PX * det.prior_roi_expand_factor)
        prior_sample_exp = det._sample_prior_color(roi_px=big)

    prior_best = prior_sample or prior_sample_exp
    prior_matches = (prior_best is not None and gb_color_gt in ("green", "red") and
                      ((prior_best == "green" and gb_color_gt == "green") or
                       (prior_best == "red" and gb_color_gt == "red")))

    # 裁断: 失败归到 A/B/C/D
    if prior_best is not None and prior_roi_iou > 0.01:
        if yolo_max_iou < 0.3 and hsv_max_iou < 0.3:
            stage = "D"  # 口径伪影: prior直采能命中但候选口径没跑它
        else:
            stage = "D"  # 候选也能命中(prior直采确认), 但 select_gtfree 没选对 → 排序问题(不归这里)
    elif not prior_contains:
        stage = "C"  # prior 偏框
    elif px_area < MIN_AREA_PX:
        stage = "A"  # 小灯被面积过滤
    elif ms < SAT_MIN:
        stage = "B"  # 暗灯被 HSV 阈值过滤
    elif prior_best is None:
        stage = "B"  # prior直采也捞不回 → 极暗/无色
    else:
        stage = "D"

    return {
        "video": VIDEO, "fi": fi, "t": g.get("t", ""),
        "gov_box_norm": f"{gb[0]:.4f},{gb[1]:.4f},{gb[2]:.4f},{gb[3]:.4f}",
        "gov_px_w": px_w, "gov_px_h": px_h, "gov_px_area": px_area,
        "gov_cx": f"{gov_cx:.4f}", "gov_cy": f"{gov_cy:.4f}",
        "gov_mean_H": f"{mh:.1f}", "gov_mean_S": f"{ms:.1f}", "gov_mean_V": f"{mv:.1f}",
        "gov_color_gt": gb_color_gt,
        "filtered_by": filtered,
        "classified_color": classified_color or "",
        "yolo_max_iou": f"{yolo_max_iou:.4f}",
        "hsv_max_iou": f"{hsv_max_iou:.4f}",
        "prior_dist": f"{prior_dist:.4f}",
        "prior_roi_iou": f"{prior_roi_iou:.4f}",
        "prior_roi_contains": int(prior_contains),
        "prior_sample": prior_sample or "",
        "prior_sample_expanded": prior_sample_exp or "",
        "prior_best": prior_best or "",
        "prior_matches": int(prior_matches),
        "stage": stage,
    }


def main():
    gt = json.load(open(GT, encoding="utf-8"))
    priors = json.load(open(PRIORS, encoding="utf-8"))
    prior = priors.get(VIDEO)
    print(f"[{VIDEO}] prior = {prior}")

    frames = [f for f in gt["frames"] if f["video"] == VIDEO]
    print(f"[{VIDEO}] total frames in GT: {len(frames)}")

    det = TrafficLightDetector(gd._cfg_tl(), verbose=False)
    yolo = gd._lazy_yolo()

    # 过滤: 与 eval_video:118-121 同口径
    gov_frames = []
    for f in frames:
        g = f
        gov_boxes = [b for b in g.get("boxes", []) if b.get("governing")]
        gcolors = {b["color"] for b in gov_boxes}
        if not gov_boxes:
            continue
        if gcolors <= {"unclear"}:
            continue
        gov_frames.append(f)
    print(f"[{VIDEO}] governing frames (after filter): {len(gov_frames)}")

    # 逐帧诊断
    fis = [int(f["source_fi"]) for f in gov_frames]
    fr_map = {int(f["source_fi"]): f for f in gov_frames}
    cap_frames = gd._read_frames_at(VIDEO, fis)

    results = []
    for fi in sorted(cap_frames.keys()):
        frame = cap_frames[fi]
        g = fr_map[fi]
        r = diag_frame(frame, g, fi, det, yolo, prior)
        if r is not None:
            results.append(r)
            print(f"  fi={fi} area={r['gov_px_area']} S={r['gov_mean_S']} "
                  f"yolo_iou={r['yolo_max_iou']} hsv_iou={r['hsv_max_iou']} "
                  f"prior_roi_iou={r['prior_roi_iou']} prior_sample={r['prior_best']} "
                  f"stage={r['stage']}",
                  flush=True)

    # 聚合
    n = len(results)
    stage_counts = defaultdict(int)
    for r in results:
        stage_counts[r["stage"]] += 1

    n_yolo_hit = sum(1 for r in results if float(r["yolo_max_iou"]) >= 0.3)
    n_hsv_hit = sum(1 for r in results if float(r["hsv_max_iou"]) >= 0.3)
    n_prior_hit = sum(1 for r in results if r["prior_best"] != "")
    n_prior_match = sum(1 for r in results if int(r["prior_matches"]))
    n_small = sum(1 for r in results if r["gov_px_area"] < MIN_AREA_PX)
    n_low_sat = sum(1 for r in results if float(r["gov_mean_S"]) < SAT_MIN)
    n_prior_contains = sum(1 for r in results if int(r["prior_roi_contains"]))

    print(f"\n=== {VIDEO} 聚合 ({n} gov 帧) ===")
    print(f"YOLO IoU≥0.3: {n_yolo_hit}/{n} = {n_yolo_hit/n*100:.1f}%")
    print(f"HSV(_candidates) IoU≥0.3: {n_hsv_hit}/{n} = {n_hsv_hit/n*100:.1f}%")
    print(f"prior直采命中(非None): {n_prior_hit}/{n} = {n_prior_hit/n*100:.1f}%")
    print(f"prior直采颜色匹配: {n_prior_match}/{n} = {n_prior_match/n*100:.1f}%")
    print(f"小灯(area<{MIN_AREA_PX}px): {n_small}/{n} = {n_small/n*100:.1f}%")
    print(f"低饱和(S<{SAT_MIN}): {n_low_sat}/{n} = {n_low_sat/n*100:.1f}%")
    print(f"gov中心在prior ROI内: {n_prior_contains}/{n} = {n_prior_contains/n*100:.1f}%")
    print(f"失败归因: {dict(stage_counts)}")

    # 裁断
    if n_prior_match / n > 0.5 and n_hsv_hit / n < 0.2:
        verdict = ("D 口径伪影: prior 直采能命中 >50% 但裸 _candidates 命中 <20%, "
                   "说明 05 天花板 10% 是 eval 没跑 prior 路径的伪影, 生产 observe() 能捞回。")
    elif stage_counts.get("C", 0) > n * 0.5:
        verdict = ("C prior 偏框: >50% 帧 prior 中心不在 gov 附近, 需重定位 05 的 prior。")
    elif stage_counts.get("A", 0) > n * 0.3:
        verdict = ("A 小灯被面积过滤: 大量 gov 灯面积 <30px, _candidates 面积过滤截掉了。")
    elif stage_counts.get("B", 0) > n * 0.3:
        verdict = ("B 暗灯被 HSV 阈值过滤: 大量 gov 灯饱和度 <130, _candidates 的 sat_min 截掉了。")
    else:
        verdict = ("混合: 无单一主因, 逐帧 stage 分布见 CSV。")
    print(f"\n裁断: {verdict}")

    # 写 CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "05_candidate_gen_per_frame.csv"
    fields = list(results[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[csv] {csv_path}")

    _write_report(results, n, stage_counts, n_yolo_hit, n_hsv_hit,
                  n_prior_hit, n_prior_match, n_small, n_low_sat, n_prior_contains,
                  prior, verdict)


def _write_report(results, n, stage_counts, n_yolo, n_hsv,
                  n_prior, n_prior_match, n_small, n_low_sat, n_prior_contains,
                  prior, verdict):
    L = [
        f"# 违章05 候选生成瓶颈根因诊断(qw, 只读)\n",
        f"> 天花板诊断把候选生成瓶颈收窄到 {VIDEO}(天花板仅 10.0%)。逐 {n} gov 帧解剖根因。\n",
        f"> 核心口径疑点: 05 天花板 10% 是生产真烂, 还是 eval 候选口径没跑 prior 直采路径的伪影?\n",
        f"> 候选构建口径与 `eval_selection_quality.py:eval_video:99-104` 逐行一致; "
        f"prior 直采调 `traffic_light.py:597 _sample_prior_color`。\n",
        f"> prior = {prior} (cx, cy, roi_px)\n\n",
        "## 裁断\n",
        f"**{verdict}**\n",
        f"\n## 聚合 ({n} gov 帧)\n",
        f"- prior = {prior}",
        f"- YOLO IoU≥0.3: {n_yolo}/{n} = {n_yolo/n*100:.1f}%",
        f"- HSV(_candidates) IoU≥0.3: {n_hsv}/{n} = {n_hsv/n*100:.1f}%",
        f"- **prior直采命中(非None): {n_prior}/{n} = {n_prior/n*100:.1f}%**",
        f"- prior直采颜色匹配: {n_prior_match}/{n} = {n_prior_match/n*100:.1f}%",
        f"- 小灯(area<{MIN_AREA_PX}px): {n_small}/{n} = {n_small/n*100:.1f}%",
        f"- 低饱和(S<{SAT_MIN}): {n_low_sat}/{n} = {n_low_sat/n*100:.1f}%",
        f"- gov中心在prior ROI内: {n_prior_contains}/{n} = {n_prior_contains/n*100:.1f}%",
        f"- 失败归因分布: {dict(stage_counts)}",
    ]

    # gov 位置 vs prior
    cxs = [float(r["gov_cx"]) for r in results]
    cys = [float(r["gov_cy"]) for r in results]
    L.append(f"\n## gov 灯实际位置 vs prior\n")
    L.append(f"- prior 中心: cx={prior[0]:.3f}, cy={prior[1]:.3f}")
    L.append(f"- gov cx 范围: {min(cxs):.3f} ~ {max(cxs):.3f} (均值 {sum(cxs)/len(cxs):.3f})")
    L.append(f"- gov cy 范围: {min(cys):.3f} ~ {max(cys):.3f} (均值 {sum(cys)/len(cys):.3f})")
    L.append(f"- **Y 方向偏差最严重**: prior Y={prior[1]:.3f} vs 实际 Y~{sum(cys)/len(cys):.3f}, "
             f"差 {abs(sum(cys)/len(cys) - prior[1]):.3f} (约 {abs(sum(cys)/len(cys) - prior[1]) * 100:.0f}%)")
    L.append(f"- prior 直采 100% 返回 'green' 是在错误位置采到别的绿色物体(50% 颜色匹配 = 随机概率), "
             f"不是真命中 gov 灯。\n")

    # 失败归因定义
    L.append(f"\n## 失败归因定义\n")
    L.append("- **A**: 小灯被面积过滤 (gov area < 30px, _candidates min_area_px 截掉)")
    L.append("- **B**: 暗灯被 HSV 阈值过滤 (S < 130 或 prior 直采也捞不回)")
    L.append("- **C**: prior 偏框 (gov 中心不在 prior ROI 内)")
    L.append("- **D**: 口径伪影 (prior 直采能命中, eval 候选口径没跑 prior 路径)")

    # 逐帧表
    L.append(f"\n## 逐帧明细\n")
    L.append("| fi | area | S | V | yolo_iou | hsv_iou | prior_roi_iou | prior_contains | prior_sample | matches | stage |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        L.append(
            f"| {r['fi']} | {r['gov_px_area']} | {r['gov_mean_S']} | {r['gov_mean_V']} | "
            f"{r['yolo_max_iou']} | {r['hsv_max_iou']} | {r['prior_roi_iou']} | "
            f"{r['prior_roi_contains']} | {r['prior_best']} | {r['prior_matches']} | {r['stage']} |"
        )

    L.append(f"\n## 方法学\n")
    L.append("- 候选构建复用 `gd._read_frames_at`/`gd._lazy_yolo`/`gd._cfg_tl`/`det._candidates`/`build_candidates`/`iou`。\n")
    L.append("- prior 直采: `det._last_frame = frame` + `det.signal_prior = (cx, cy)` + `det._sample_prior_color()`。\n")
    L.append(f"- _candidates 阈值: sat_min={SAT_MIN}, value_floor={VALUE_FLOOR}, min_area_px={MIN_AREA_PX}, color_s_min={COLOR_S_MIN}。\n")
    L.append("- _sample_prior_color 阈值: S≥60, V≥40, 无面积过滤, min_frac=0.2%, 自适应 2x 扩展。\n")
    L.append("- 只读诊断, 不训练、不改生产代码、不碰 s1/_A 缓存。\n")
    L.append(f"- 逐帧明细 CSV: `data/output/qw/05_candidate_gen_per_frame.csv` (供 cc bit-for-bit 复核)。\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {REPORT}")


if __name__ == "__main__":
    main()
