#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_falsegreen_canonical.py — 用 canonical 逐帧 GT 重算误绿(取代 events 段映射 + 静态 true_center)。

相对 measure_falsegreen.py 的关键变化(cc 2026-07-30):
- 真值源 = datasets/gt/light_canonical_gt.json(Jacob 手工逐帧标, 399帧/全11视频)。
  gt_walk(帧) = 该帧存在 governing 框且其 color==green; no_light → gt_walk=False(无真灯, 任何绿=误绿);
  governing 全 unclear → 该帧 UNKNOWN 排除。彻底去 events 段级粗糙 + 相机运动混淆。
- 只在 399 个 GT 帧上评测(逐帧比该帧 governing 真值), 无时间段映射。
- 成因归因用**该帧 governing 框**(非静态 true_center)→ 解 Diag3 混淆:
  选中框与任一 governing 框 IoU≥0.3=选对灯→"状态判错"; <0.3=选错灯(干扰/反射/背面/车灯/树叶)。

去循环不变: 选灯 select_gtfree(无GT参, PED_PRIOR 形状先验, 禁 seed_with_gt/derive_ped_priors);
           GT 只比对不进推理; 只读不接线。
5 行同 measure_falsegreen: R0 color@prior / R1 color@best(noL3) / R2 color@best(L3.5) / R3 clf@best(noL3) / R4 clf@best(L3.5)。
用法: PYTHONPATH=src ./.venv/bin/python scripts/measure_falsegreen_canonical.py
"""
import json, sys, types
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from PIL import Image

from redlight.models.ped_light_selector import select_gtfree, compute_temporal_scores, iou
from redlight.models.l3_ped_vehicle import L3PedVehicleNet, crop_candidate, score_crop
from redlight.models.traffic_light import TrafficLightDetector
from redlight.models.signal_candidates import build_candidates
from redlight.models.signal_state_classifier import SignalStateClassifier

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
PRIORS = ROOT / "configs" / "light_priors.json"
L3W = ROOT / "models" / "l3_ped_full.pt"
CLF = ROOT / "models" / "ped_signal.pt"
REPORT = ROOT / "docs" / "reports" / "2026-07-30-cc-falsegreen-canonical.md"
PED_PRIOR = {"aspect_mean": 3.0, "aspect_std": 1.0, "area_mean": 0.005, "area_std": 0.003}
ROWS = ["R0", "R1", "R2", "R3", "R4"]
LABELS = ["R0 现役端到端(color@prior)", "R1 color@best(no-L3)", "R2 color@best(L3 w=.5)",
          "R3 clf@best(no-L3)", "R4 clf@best(L3 w=.5)"]


def _cfg():
    tl = types.SimpleNamespace(method="color", smoothing_window=8, sat_min=130, value_floor=60,
                               min_area_px=30, max_area_ratio=0.008, max_aspect_ratio=3.5, color_s_min=22)
    return types.SimpleNamespace(traffic_light=tl)


def color_state(roi):
    if roi is None or roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g = int(cv2.inRange(hsv, np.array([35, 130, 60]), np.array([85, 255, 255])).sum()) // 255
    r = (int(cv2.inRange(hsv, np.array([0, 130, 60]), np.array([10, 255, 255])).sum()) +
         int(cv2.inRange(hsv, np.array([170, 130, 60]), np.array([180, 255, 255])).sum())) // 255
    if g == 0 and r == 0:
        return None
    return "green" if g >= r else "red"


def crop_cv2(frame, b):
    H, W = frame.shape[:2]
    px = (max(0, int(b[0] * W)), max(0, int(b[1] * H)), min(W, int(b[2] * W)), min(H, int(b[3] * H)))
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return frame[px[1]:px[3], px[0]:px[2]]


def box_from_prior(prior, W, H):
    cx, cy, roi_px = prior
    return (max(0, cx - (roi_px / 2) / W), max(0, cy - (roi_px / 2) / H),
            min(1, cx + (roi_px / 2) / W), min(1, cy + (roi_px / 2) / H))


def main():
    gt = json.load(open(GT, encoding="utf-8"))
    priors = json.load(open(PRIORS, encoding="utf-8"))
    det = TrafficLightDetector(_cfg(), verbose=False)
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    clf = SignalStateClassifier(str(CLF), verbose=False)
    l3 = L3PedVehicleNet(); l3.load_state_dict(torch.load(L3W, map_location="cpu", weights_only=True)); l3.eval()

    # 按视频组织 GT 帧
    by_video = defaultdict(dict)
    for fr in gt["frames"]:
        by_video[fr["video"]][int(fr["source_fi"])] = fr

    stats = {r: defaultdict(int) for r in ROWS}
    per_video = {v: {r: defaultdict(int) for r in ROWS} for v in by_video}
    cause = Counter()
    n_unknown = defaultdict(int)
    n_nocand = defaultdict(int)
    miss_green = defaultdict(int)

    for vi, video in enumerate(sorted(by_video)):
        gframes = by_video[video]
        want = sorted(gframes)
        cap = cv2.VideoCapture(str(ROOT / "input_video" / f"{video}.mp4"))
        # 逐帧读, 命中 GT 帧则跑候选(存下供 temporal + 评测)
        recs = []  # {fi, frame, candidates}
        src_fi = 0
        wantset = set(want)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if src_fi in wantset:
                H, W = frame.shape[:2]
                res = model(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
                yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
                hsv_px = [s["box"] for s in det._candidates(frame)]
                cands = build_candidates(yolo_px, hsv_px, W, H)
                crec = [{"box_norm": [c["box"][0] / W, c["box"][1] / H, c["box"][2] / W, c["box"][3] / H],
                         "source": c["source"]} for c in cands]
                recs.append({"fi": src_fi, "frame": frame.copy(), "candidates": crec})
            src_fi += 1
        cap.release()
        ts_list = compute_temporal_scores(recs)  # L2 时序(GT帧间, 稀疏)

        for idx, rec in enumerate(recs):
            fi = rec["fi"]; frame = rec["frame"]; H, W = frame.shape[:2]
            g = gframes[fi]
            gov = [b for b in g.get("boxes", []) if b.get("governing")]
            gcolors = {b["color"] for b in gov}
            no_light = g.get("no_light", False)
            # 真值判定
            if gov and gcolors <= {"unclear"}:
                n_unknown[video] += 1
                continue
            gt_walk = ("green" in gcolors)  # no_light 或 全红/off → False
            cands = rec["candidates"]
            if not cands:
                n_nocand[video] += 1
                if gt_walk:
                    for r in ROWS:
                        stats[r]["eval"] += 1; per_video[video][r]["eval"] += 1
                        stats[r]["missgreen"] += 1
                continue
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            l3_scores = {j: score_crop(l3, crop_candidate(frame_pil, tuple(c["box_norm"])))["ped"]
                         for j, c in enumerate(cands)}
            ts = ts_list[idx]
            sel_no = select_gtfree(cands, PED_PRIOR, ts)
            sel_l3 = select_gtfree(cands, PED_PRIOR, ts, l3_scores=l3_scores, l3_weight=0.5)
            prior_box = box_from_prior(priors[video], W, H)
            col_prior = color_state(crop_cv2(frame, prior_box))

            def state_of(sel):
                if sel is None:
                    return None, None
                roi = crop_cv2(frame, tuple(sel["box_norm"]))
                cc = color_state(roi)
                cl = clf.classify(roi)[0] if roi is not None else "off"
                return cc, cl
            col_no, clf_no = state_of(sel_no)
            col_l3, clf_l3 = state_of(sel_l3)

            outs = {"R0": col_prior == "green", "R1": col_no == "green", "R2": col_l3 == "green",
                    "R3": clf_no == "walk", "R4": clf_l3 == "walk"}
            for r in ROWS:
                s = stats[r]; pv = per_video[video][r]
                s["eval"] += 1; pv["eval"] += 1
                if outs[r]:
                    s["walk"] += 1; pv["walk"] += 1
                    if not gt_walk:
                        s["fg"] += 1; pv["fg"] += 1
                elif gt_walk:
                    s["missgreen"] += 1
            if gt_walk:
                for r in ROWS:
                    if not outs[r]:
                        miss_green[r] += 1
            # 成因(R1 误绿): 用该帧 governing 框
            if outs["R1"] and not gt_walk and sel_no is not None:
                sb = sel_no["box_norm"]
                best = max((iou(sb, b["box_norm"]) for b in gov), default=0.0)
                if not gov:
                    cause["选错灯/干扰(该帧无真行人灯)"] += 1
                elif best >= 0.3:
                    cause["状态判错(选对灯但色判错)"] += 1
                else:
                    cause["选错灯(干扰:反射/背面/车灯/树叶)"] += 1
        print(f"  [{vi+1}/{len(by_video)}] {video}: GT帧{len(recs)}")

    # 汇总
    print("\n=== 误绿(canonical 逐帧GT) ===")
    print(f"{'行':30s} 帧级误绿率   可评帧   误绿帧")
    rows_out = []
    for i, r in enumerate(ROWS):
        s = stats[r]; ev = s["eval"]; fg = s["fg"]
        rate = fg / ev if ev else 0
        rows_out.append((LABELS[i], rate, ev, fg))
        print(f"  {LABELS[i]:30s} {rate*100:6.2f}%   {ev:5d}   {fg:5d}")
    print(f"\n因 unknown 排除: {sum(n_unknown.values())} {dict(n_unknown)}")
    print(f"无候选帧: {sum(n_nocand.values())} {dict(n_nocand)}")
    print(f"成因(R1误绿): {dict(cause)}")
    print(f"漏绿(附注): {dict(miss_green)}")

    # 扣 05 后的 R1
    r1 = per_video
    tot_ev = sum(per_video[v]["R1"]["eval"] for v in per_video)
    tot_fg = sum(per_video[v]["R1"]["fg"] for v in per_video)
    ev05 = per_video.get("违章05", {}).get("R1", {}).get("eval", 0)
    fg05 = per_video.get("违章05", {}).get("R1", {}).get("fg", 0)
    ex_rate = (tot_fg - fg05) / (tot_ev - ev05) if (tot_ev - ev05) else 0
    print(f"\nR1 全量 {tot_fg}/{tot_ev}={tot_fg/tot_ev*100:.2f}%  | 扣05后 {tot_fg-fg05}/{tot_ev-ev05}={ex_rate*100:.2f}%")

    _write_report(rows_out, per_video, cause, n_unknown, n_nocand, miss_green, tot_fg, tot_ev, fg05, ev05, ex_rate)
    print(f"\n[out] {REPORT}")


def _write_report(rows, per_video, cause, unk, nocand, miss, tot_fg, tot_ev, fg05, ev05, ex_rate):
    L = ["# 误绿测量(canonical 逐帧 GT 重算) — cc 2026-07-30\n",
         "> 真值=`datasets/gt/light_canonical_gt.json`(Jacob 手工逐帧标, 399帧/全11视频)。逐帧比该帧 governing 真值, 无 events 段映射/无静态 true_center。只测不接线。\n",
         "## 总体(5 行)\n", "| 行 | 帧级误绿率 | 可评帧 | 误绿帧 |", "|---|---|---|---|"]
    for label, rate, ev, fg in rows:
        L.append(f"| {label} | {rate*100:.2f}% | {ev} | {fg} |")
    L.append("\n- 帧级误绿率 = (out=='walk/green' 且 该帧 governing 真值非绿) / 可评帧。gt_walk=该帧任一 governing 框为绿。")
    L.append("- **主标尺 R1**(最好 GT-free 选灯 + 现役 color 判色)。R0=现役端到端(裸 prior-ROI, 偏框故偏高)。R3/R4=分类器路径。")
    L.append(f"\n## 扣掉病态 05 后(R1)\n- 全量 R1: {tot_fg}/{tot_ev} = **{tot_fg/tot_ev*100:.2f}%**")
    L.append(f"- 扣 05: {tot_fg-fg05}/{tot_ev-ev05} = **{ex_rate*100:.2f}%**")
    L.append("\n## 11 视频分解(帧级误绿率)\n| 视频 | " + " | ".join(ROWS) + " | 可评帧(R1) |")
    for v in sorted(per_video):
        cells = []
        for r in ROWS:
            s = per_video[v][r]; cells.append(f"{(s['fg']/s['eval']*100 if s['eval'] else 0):.1f}%")
        L.append(f"| {v} | " + " | ".join(cells) + f" | {per_video[v]['R1']['eval']} |")
    L.append("\n## 成因归类(R1 误绿帧, 基于该帧 governing 框 IoU)\n| 成因 | 数 |")
    tot = sum(cause.values()) or 1
    for k, c in cause.most_common():
        L.append(f"| {k} | {c} ({c/tot*100:.1f}%) |")
    L.append(f"\n## 边界\n- 因 governing unclear 排除: {sum(unk.values())} {dict(unk)}")
    L.append(f"- 无候选帧: {sum(nocand.values())} {dict(nocand)}")
    L.append(f"- 漏绿(真值绿管线未输出绿, 附注): {dict(miss)}")
    REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
