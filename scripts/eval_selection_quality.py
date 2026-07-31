#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_selection_quality.py — §3.3 全 399 帧选灯质量评分台(LOVO + 弃权门 + R2/R4)。

主指标:
- 选灯精度: governing 存在帧, 选中框与某 governing 框 IoU≥0.3 的比例.
- 正确弃权率: no_light 帧, select 返 None(不输出绿)的比例.
- 误绿率(头条副指标, 扣05): 选中且 color=green 但帧无绿真值, 扣05后.
- 漏绿(硬约束 R4): governing=green 帧但 select 返 None/非绿 → ≤80 判不过.

落实 cc R2/R4:
- R2: 无灯帧分布不均(61在03/18在01/06·11为0). 03 折**单列**不混 mean; 06/11 正确弃权率显式 **N/A**(不充0/100).
- R4: 漏绿 ≤80 硬约束, gate 漏绿>80 直接判不过.

τ 全局(R3): 训练折内推荐单一全局 τ(train 脚本定), 本台在 τ-grid 扫敏感性曲线,
           选满足 漏绿≤80 且 误绿(扣05)最小 的全局 τ; 禁 per-video τ.

LOVO: 每视频 V 用其余视频训判别器, 在 V 帧上评(去循环). GT 只评测不进推理.

用法(冒烟): PYTHONPATH=src ./.venv/bin/python scripts/eval_selection_quality.py --videos 违章03,违章06 --seeds 0
"""
import json, sys, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from redlight.models import governing_disc as gd
from redlight.models.ped_light_selector import select_gtfree, iou
from redlight.models.signal_candidates import build_candidates
from redlight.models.traffic_light import TrafficLightDetector

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
PED_PRIOR = {"aspect_mean": 3.0, "aspect_std": 1.0, "area_mean": 0.005, "area_std": 0.003}
TAU_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
REPORT = ROOT / "docs" / "reports" / "2026-07-31-wb-selection-quality.md"


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


def _video_frames(gt):
    by_video = defaultdict(list)
    for fr in gt["frames"]:
        by_video[fr["video"]].append(fr)
    return by_video


def eval_video(video, frames, model, det, yolo):
    """在单视频 GT 帧上评: 返回逐帧结果(含 best_cand/best_conf 供 τ-grid 复用)。LOVO 模型已训好。"""
    tf = gd._get_transform()
    fr_map = {int(f["source_fi"]): f for f in frames}
    cap_frames = gd._read_frames_at(video, [int(f["source_fi"]) for f in frames])
    rows = []
    for fi, frame in cap_frames.items():
        g = fr_map[fi]
        H, W = frame.shape[:2]
        gov_boxes = [tuple(b["box_norm"]) for b in g.get("boxes", []) if b.get("governing")]
        gcolors = {b["color"] for b in g.get("boxes", []) if b.get("governing")}
        no_light = g.get("no_light", False)
        if gov_boxes and gcolors <= {"unclear"}:
            continue  # UNKNOWN 排除
        gt_green = "green" in gcolors
        res = yolo(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
        yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
        hsv_px = [s["box"] for s in det._candidates(frame)]
        cands_raw = build_candidates(yolo_px, hsv_px, W, H)
        cands = [{"box_norm": (c["box"][0] / W, c["box"][1] / H, c["box"][2] / W, c["box"][3] / H),
                  "source": c["source"]} for c in cands_raw]
        if not cands:
            rows.append({"video": video, "fi": fi, "no_light": no_light, "gt_green": gt_green,
                         "gov_boxes": gov_boxes, "best_cand": None, "best_conf": 0.0, "best_color": None})
            continue
        scores = {j: gd.score_crop(model, tf(gd.crop_candidate(frame, tuple(c["box_norm"]))))
                  for j, c in enumerate(cands)}
        # 选 best_conf 候选(与 select_gtfree 同口径, weight=1.0)
        best_j = max(scores, key=lambda j: scores[j])
        best_cand = cands[best_j]
        best_conf = scores[best_j]
        best_color = color_state(crop_cv2(frame, tuple(best_cand["box_norm"])))
        rows.append({"video": video, "fi": fi, "no_light": no_light, "gt_green": gt_green,
                     "gov_boxes": gov_boxes, "best_cand": best_cand, "best_conf": best_conf,
                     "best_color": best_color})
    return rows


def metrics_for_tau(rows, tau, exclude_video=None):
    """给定全局 τ 算指标。exclude_video 用于扣05主标尺。"""
    sel = [r for r in rows if r["best_cand"] is not None]
    abstain = [r for r in rows if r["best_cand"] is None]
    # 选中 = best_conf>=tau 的帧; 其余(abstain) sel=None
    def _sel_color(r):
        return r["best_color"] if r["best_conf"] >= tau else None
    gov_frames = [r for r in rows if r["gov_boxes"]]
    nol_frames = [r for r in rows if r["no_light"]]
    green_frames = [r for r in rows if r["gt_green"]]
    # 选灯精度
    hit = 0
    for r in gov_frames:
        if r["best_conf"] >= tau and r["best_cand"] is not None:
            if any(iou(r["best_cand"]["box_norm"], gb) >= 0.3 for gb in r["gov_boxes"]):
                hit += 1
    sel_prec = hit / len(gov_frames) if gov_frames else None
    # 正确弃权
    rej = sum(1 for r in nol_frames if r["best_conf"] < tau)  # 无灯帧被弃权
    rej_rate = rej / len(nol_frames) if nol_frames else None
    # 误绿(扣 exclude_video): 选中绿但帧非绿(含无灯帧=弃权门要治的主失败模式)
    fg = sum(1 for r in rows if _sel_color(r) == "green" and not r["gt_green"]
             and (exclude_video is None or r["video"] != exclude_video))
    n_eval = sum(1 for r in rows if not r["gt_green"]
                 and (exclude_video is None or r["video"] != exclude_video))
    fg_rate = fg / n_eval if n_eval else 0.0
    # 漏绿
    miss = sum(1 for r in green_frames if not (_sel_color(r) == "green"))
    return {"sel_prec": sel_prec, "rej_rate": rej_rate, "n_nol": len(nol_frames),
            "fg": fg, "n_eval": n_eval, "fg_rate": fg_rate, "miss": miss, "n_green": len(green_frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None)
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    gt = json.load(open(GT, encoding="utf-8"))
    det = TrafficLightDetector(gd._cfg_tl(), verbose=False)
    yolo = gd._lazy_yolo()
    by_video = _video_frames(gt)
    videos = [v for v in by_video if v in args.videos.split(",")] if args.videos else sorted(by_video)

    all_rows = []
    for V in videos:
        sub = {"schema": gt.get("schema"), "frames": [f for f in gt["frames"] if f["video"] != V]}
        pos, neg_b, neg_a = gd.build_crop_dataset(sub, use_negative_a=False)
        model = gd.train_model(pos, neg_b, seed=seeds[0])
        rows = eval_video(V, by_video[V], model, det, yolo)
        all_rows.extend(rows)
        print(f"  [{V}] GT帧={len(rows)}")

    # τ 敏感性(R3): 扫 grid, 选 漏绿≤80 且 误绿(扣05)最小 的全局 τ
    print("\n=== τ 敏感性(全局, R3) ===")
    print(f"{'τ':>4} {'选灯精度':>8} {'弃权率':>8} {'误绿(扣05)':>11} {'漏绿':>5} {'gate':>6}")
    best = None
    for tau in TAU_GRID:
        m = metrics_for_tau(all_rows, tau, exclude_video="违章05")
        gate = "PASS" if m["miss"] <= 80 else "FAIL"
        print(f"{tau:>4.1f} {(_f(m['sel_prec'])*100):>7.1f}% {(_f(m['rej_rate'])*100):>7.1f}% "
              f"{m['fg_rate']*100:>10.2f}% {m['miss']:>5} {gate:>6}")
        # 选 τ: 漏绿<=80 前提下误绿最小
        if m["miss"] <= 80:
            if best is None or m["fg_rate"] < best[1]["fg_rate"]:
                best = (tau, m)
    tau_global = best[0] if best else 0.5
    print(f"\n[τ] 全局选定 = {tau_global:.1f} (漏绿≤80 约束下误绿最小)")

    # 03 单列 + 06/11 N/A (R2)
    print("\n=== 无灯帧处理(R2) ===")
    v03 = [r for r in all_rows if r["video"] == "违章03"]
    if v03:
        m03 = metrics_for_tau(v03, tau_global)
        print(f"03(单列, 不混 mean): 无灯帧={m03['n_nol']} 正确弃权率={_f(m03['rej_rate'])*100:.1f}%")
    for v in ("违章06", "违章11"):
        nol = [r for r in all_rows if r["video"] == v and r["no_light"]]
        print(f"{v}: 正确弃权率 = N/A (零无灯帧)" if not nol else f"{v}: 无灯帧={len(nol)}")

    mg = metrics_for_tau(all_rows, tau_global, exclude_video="违章05")
    print(f"\n=== 主标尺(τ={tau_global:.1f}, 扣05) ===")
    print(f"误绿(扣05): {mg['fg']}/{mg['n_eval']} = {mg['fg_rate']*100:.2f}%")
    print(f"漏绿: {mg['miss']} (硬约束≤80 → {'PASS' if mg['miss']<=80 else 'FAIL'})")
    print("[注] 冒烟口径(少量视频/单seed); 完整 LOVO(≥5 seed 报 worst)需全量运行。")


def _f(x):
    return x if x is not None else 0.0


if __name__ == "__main__":
    main()
