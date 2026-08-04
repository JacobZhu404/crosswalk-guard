#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# diag_0906_green_failure.py — 09/06 绿帧漏绿五分归因诊断(只读, 不写生产码)。
#
# 背景: selection-quality 报告里 09 漏绿=22(base)/29(neg_a worst)、06 漏绿若干,
# cc plan-gate 裁定: 既有的逐帧 observe() 路径基本已在做(observe 159-194 已用逐帧 YOLO,
# 固定 prior 直采197只兜底), 不能因"重定心"就重造已有行为。放行条件 = 先出失败归因诊断。
#
# 五分(四类真因):
#   ① 候选生成召回洞 — YOLO+HSV 候选池里有没有框 IoU>=0.3 覆盖任一真绿 governing 灯?
#   ② yolo_prior_near=0.12 误留 prior — observe() 走固定 prior 直采(197)时, 真灯是否在 prior 外?
#   ③ 裁剪 IoU vs canonical — observe/selection 实际使用的 crop 框 IoU(canon 绿框) 是否<0.3?
#   ④ HSV 判色(绿抑制) — 真绿灯区域 color_state 是否被压成 red/off(绿/红像素比)?
#   ⑤ 多灯选中哪盏 — ≥2 governing 灯时, observe/selection 选中的是不是绿的那盏?
#
# 两类子系统都查:
#   A) selection 路径: select_gtfree(cands, PED_PRIOR) + color_state(crop) — 即报告漏绿的来源。
#   B) observe 路径: det.observe(frame, yolo_boxes) 带 signal_prior — 即 prior 重定位项目要改的信号态路径。
#
# 用法: PYTHONPATH=src ./.venv/bin/python scripts/diag_0906_green_failure.py [--videos 违章09,违章06]
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from redlight.models import governing_disc as gd
from redlight.models.ped_light_selector import select_gtfree, iou
from redlight.models.signal_candidates import build_candidates
from redlight.models.traffic_light import TrafficLightDetector

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
LIGHT_LOC = ROOT / "datasets" / "light_location_gt.json"
PED_PRIOR = {"aspect_mean": 3.0, "aspect_std": 1.0, "area_mean": 0.005, "area_std": 0.003}
YOLO_PRIOR_NEAR = 0.12
IOU_OK = 0.3


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


def crop(frame, b):
    H, W = frame.shape[:2]
    px = (max(0, int(b[0]*W)), max(0, int(b[1]*H)), min(W, int(b[2]*W)), min(H, int(b[3]*H)))
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return frame[px[1]:px[3], px[0]:px[2]]


def prior_center_for(video):
    ll = json.load(open(LIGHT_LOC, encoding="utf-8"))
    cs = []
    for a in ll["annotations"]:
        if a["video"] == video and a.get("prior_norm"):
            p = a["prior_norm"]
            cs.append(((p[0]+p[2])/2, (p[1]+p[3])/2))
    if not cs:
        return None
    return (float(np.mean([c[0] for c in cs])), float(np.mean([c[1] for c in cs])))


def diagnose(video, frames, det, yolo):
    prior = prior_center_for(video)
    det.signal_prior = prior  # 复现生产 prior 路径
    recs = []
    cap = gd._read_frames_at(video, [int(f["source_fi"]) for f in frames])
    for fr in frames:
        fi = int(fr["source_fi"])
        frame = cap[fi]
        H, W = frame.shape[:2]
        gboxes = [tuple(b["box_norm"]) for b in fr.get("boxes", []) if b.get("governing")]
        gcolors = [b.get("color") for b in fr.get("boxes", []) if b.get("governing")]
        green_boxes = [b for b, c in zip(gboxes, gcolors) if c == "green"]
        if not green_boxes:
            continue
        n_gov = len(gboxes)
        # 候选池(与 canonical 同口径)
        res = yolo(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
        yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
        hsv_px = [s["box"] for s in det._candidates(frame)]
        cands_raw = build_candidates(yolo_px, hsv_px, W, H)
        cands = [{"box_norm": (c["box"][0]/W, c["box"][1]/H, c["box"][2]/W, c["box"][3]/H),
                  "source": c["source"]} for c in cands_raw]
        # ① 候选召回: 任一 candidate 覆盖任一真绿框?
        best_recall_iou = max((iou(c["box_norm"], gb) for c in cands for gb in green_boxes), default=0.0)
        recall_ok = best_recall_iou >= IOU_OK
        # 绿框候选来源分解: 覆盖真绿的候选里, 是 YOLO 还是 HSV?
        green_cands = [c for c in cands if any(iou(c["box_norm"], gb) >= IOU_OK for gb in green_boxes)]
        green_cand_is_yolo = any(c["source"] == "yolo" for c in green_cands)

        # A) selection 路径
        best = select_gtfree(cands, PED_PRIOR, governing_scores=None,
                             governing_weight=0.3, governing_threshold=0.0)
        sel_source = best["source"] if best else None
        sel_box = best["box_norm"] if best else None
        sel_color = color_state(crop(frame, sel_box)) if sel_box else None
        sel_iou = max((iou(sel_box, gb) for gb in green_boxes), default=0.0) if sel_box else 0.0

        # B) observe 路径(带 prior)
        yolo_boxes_for_observe = [b for b in yolo_px if (b[3]-b[1]) > 0 and (b[2]-b[0]) > 0]
        obs = det.observe(frame, yolo_light_boxes=yolo_boxes_for_observe)
        # 复现分支判定(observe 不返回用哪个框, 这里重算)
        use_yolo = None
        obs_box = None
        best_dist = None
        if yolo_boxes_for_observe:
            bb = None
            bdist = None
            bb_box = None
            for b in yolo_boxes_for_observe:
                x1, y1, x2, y2 = [int(v) for v in b]
                bcx = ((x1+x2)/2)/W; bcy = ((y1+y2)/2)/H
                if bcy < det.yolo_cy_min:
                    continue
                # 精确复刻 observe: 必须 _sample_box 出颜色, 否则该框被跳过
                col, gn, rn = det._sample_box(b, W, H)
                if col is None:
                    continue
                if prior is not None:
                    d = ((bcx-prior[0])**2 + (bcy-prior[1])**2)**0.5
                else:
                    d = None
                sc = -d if d is not None else max(gn, rn)
                if bb is None or sc > bb[1]:
                    bb = (b, sc); bdist = d; bb_box = (x1/W, y1/H, x2/W, y2/H)
            use_yolo = bb is not None and (prior is None or bdist is None or bdist > YOLO_PRIOR_NEAR)
            if use_yolo and bb is not None:
                obs_box = bb_box
                best_dist = bdist
            elif prior is not None:
                # prior 直采路径: crop = prior ROI(_sample_prior_color 用 signal_prior 点)
                obs_box = (prior[0]-0.02, prior[1]-0.03, prior[0]+0.02, prior[1]+0.03)
        obs_iou = max((iou(obs_box, gb) for gb in green_boxes), default=0.0) if obs_box else 0.0
        obs_color = obs.get("obs")

        # 真绿区域 HSV 判色(④)
        true_colors = [color_state(crop(frame, gb)) for gb in green_boxes]
        true_green_judged = sum(1 for tc in true_colors if tc == "green")
        # ④ 绿抑制: 真绿框被判定非绿
        hsv_suppress = (true_green_judged < len(green_boxes)) and (len(green_boxes) > 0)

        rec = {
            "video": video, "fi": fi, "n_gov": n_gov,
            "recall_iou": round(best_recall_iou, 3), "recall_ok": recall_ok,
            "green_cand_is_yolo": green_cand_is_yolo, "sel_source": sel_source,
            "sel_color": sel_color, "sel_iou": round(sel_iou, 3),
            "obs_path": ("yolo" if use_yolo else ("prior" if prior is not None else "fallback")),
            "obs_color": obs_color, "obs_iou": round(obs_iou, 3), "obs_dist_to_prior": round(best_dist, 3) if best_dist is not None else None,
            "true_green_judged": f"{true_green_judged}/{len(green_boxes)}",
            "hsv_suppress": hsv_suppress,
        }
        recs.append(rec)
    return recs


def classify(rec):
    """主因归类。关键区分: observe(逐帧YOLO)已工作 vs selector 选错框。

    ① 候选生成召回洞 — 候选池里无任何框覆盖真绿(召回缺失)。
    ② yolo_prior_near=0.12 误留 prior — observe 走固定 prior 直采且 crop 与真绿错位。
    ③ 裁剪 IoU 错位 — observe 实际 crop 与真绿框 IoU<0.3(真 crop 错位, 非选灯)。
    ④ HSV 绿抑制 — 真绿区域 color_state 压成非绿(绿/红像素比)。
    ⑤ selector 选错框 — 候选池有绿框(recall OK), 但 select_gtfree 选了非绿/非重叠候选(排序/多灯)。
    ⑥ 已识别绿(不漏)。
    """
    if not rec["recall_ok"]:
        return "①候选召回洞"
    # observe(逐帧YOLO)是否工作: 本职是出颜色, YOLO 框比 canonical 紧框大→IoU 偏低属正常,
    # 故以 obs_color=='green' 判 observe 是否找到绿(obs_iou 仅作对齐透明度, 不卡 PASS)。
    obs_ok = (rec["obs_color"] == "green")
    if rec["obs_path"] == "prior" and not obs_ok:
        return "②prior误留/③crop错位"
    if rec["hsv_suppress"]:
        return "④HSV绿抑制"
    # 候选池有绿框(recall OK), 但 selector 没选到绿 → 排序/多灯(⑤)
    if rec["sel_color"] != "green" or rec["sel_iou"] < IOU_OK:
        if obs_ok:
            return "⑤selector选错框(绿候选在池, 排序/多灯挑非绿)"
        return "③/④选框非绿(未定位)"
    return "⑥已识别绿(不漏)"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="违章09,违章06")
    args = ap.parse_args()
    videos = args.videos.split(",")
    gt = json.load(open(GT, encoding="utf-8"))
    byv = defaultdict(list)
    for f in gt["frames"]:
        byv[f["video"]].append(f)
    det = TrafficLightDetector(gd._cfg_tl(), verbose=False)
    yolo = gd._lazy_yolo()
    allrecs = []
    for v in videos:
        recs = diagnose(v, byv[v], det, yolo)
        allrecs.extend(recs)
        print(f"\n===== {v}: {len(recs)} gt_green 帧 =====")
        cls = defaultdict(int)
        for r in recs:
            c = classify(r)
            cls[c] += 1
        for r in recs:
            print(f"  fi={r['fi']:<5} gov={r['n_gov']} recall={r['recall_iou']}({'OK' if r['recall_ok'] else 'X'}) "
                  f"greenCandYOLO={r['green_cand_is_yolo']} sel={r['sel_source']}:{r['sel_color']}/{r['sel_iou']} "
                  f"obs={r['obs_path']}:{r['obs_color']}/{r['obs_iou']} dist={r['obs_dist_to_prior']} "
                  f"trueGreen={r['true_green_judged']} -> {classify(r)}")
        print(f"  -- 主因分布: {dict(cls)}")
    # 汇总
    print("\n===== 汇总(全部视频) =====")
    tot = defaultdict(int)
    for r in allrecs:
        tot[classify(r)] += 1
    for k, n in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {k}: {n}")
    print(f"  总 gt_green 帧: {len(allrecs)}")


if __name__ == "__main__":
    main()
