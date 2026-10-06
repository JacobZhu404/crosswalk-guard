#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recompute_cause_perframe.py — 用逐帧 GT 框(true_box_norm)重算误绿成因, 去静态 true_center 混淆(Diag3 同款)。

只重放 R1 主标尺(color@best, no-L3): 不需要 CNN(L3/分类器), 仅几何+时间选灯 + color 判状态。
读 candidates_dense.json(已存候选) + 源视频帧(crop+color) + light_location_gt.json(逐帧真灯框)。
对原 measure 用"每视频静态 true_center"的归因, 换成"该帧若有标注 GT 框则用之, 否则 fallback 静态",
看 06 等非 05 视频的"车灯误选"有多少是灯移位导致的误判。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/recompute_cause_perframe.py
输出: 原/新归因分布对比 + per-video 修正 + 被重分类帧清单(到 stdout + JSON)。
"""
import json, sys, os, time
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np

from redlight.models.ped_light_selector import select_gtfree, compute_temporal_scores

CAND = ROOT / "data" / "output" / "candidates_dense.json"
EVENTS = ROOT / "datasets" / "gt" / "events.csv"
GTLOC = ROOT / "datasets" / "light_location_gt.json"
PRIORS = ROOT / "configs" / "light_priors.json"
FPS = 29.70

# 内联 ped_prior(与 measure 一致: 竖长条形状先验)
PED_PRIOR = {"aspect_mean": 3.0, "aspect_std": 1.0, "area_mean": 0.005, "area_std": 0.003}
DIST_TH = 0.04  # 成因距离阈值(与 measure 一致)


def crop_cv2(frame, box_norm):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box_norm
    px = (max(0, int(x1 * W)), max(0, int(y1 * H)), min(W, int(x2 * W)), min(H, int(y2 * H)))
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return frame[px[1]:px[3], px[0]:px[2]]


def box_from_prior(prior, W, H):
    cx, cy, roi_px = prior
    hw = (roi_px / 2) / W
    hh = (roi_px / 2) / H
    return (max(0, cx - hw), max(0, cy - hh), min(1, cx + hw), min(1, cy + hh))


def _color_state_roi(roi):
    if roi is None or roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_g = cv2.inRange(hsv, np.array([35, 130, 60]), np.array([85, 255, 255]))
    mask_r = cv2.inRange(hsv, np.array([0, 130, 60]), np.array([10, 255, 255])) | \
             cv2.inRange(hsv, np.array([170, 130, 60]), np.array([180, 255, 255]))
    g = int(mask_g.sum()) // 255
    r = int(mask_r.sum()) // 255
    if g == 0 and r == 0:
        return None
    return "green" if g >= r else "red"


def load_events():
    import csv
    segs = defaultdict(list)
    with open(EVENTS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            segs[row["video"]].append((float(row["start_s"]), float(row["end_s"]),
                                       row["light_state"].strip(), row["is_violation"].strip()))
    return segs


def gt_lookup(segs, video, t):
    for s, e, state, viol in segs.get(video, []):
        if s - 1e-6 <= t <= e + 1e-6:
            if state == "green":
                return True
            if state in ("red", "occluded", "none"):
                return False
            return None
    return None


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def main():
    cands = json.load(open(CAND))
    segs = load_events()
    gtloc = json.load(open(GTLOC))
    priors = json.load(open(PRIORS))
    derived = gtloc.get("derived_per_video", {})
    fps_per_video = cands.get("fps_per_video", {})

    # 逐帧 GT 框: (video, fi) -> true_box_norm
    ann_by_vf = {}
    for a in gtloc.get("annotations", []):
        tb = a.get("true_box_norm")
        if tb is not None:
            ann_by_vf[(a["video"], a["fi"])] = tb

    videos = cands["videos"]
    cause_static = Counter()
    cause_perframe = Counter()
    reclassified = []          # 被重分类的帧
    per_video_static = defaultdict(Counter)
    per_video_perframe = defaultdict(Counter)
    n_falsegreen = 0

    t0 = time.time()
    for vi, (video, recs) in enumerate(videos.items()):
        recs = sorted(recs, key=lambda r: r["fi"])
        ts_list = compute_temporal_scores(recs)
        fi_list = [r["fi"] for r in recs]
        cap = cv2.VideoCapture(str(ROOT / "input_video" / f"{video}.mp4"))
        src_fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if src_fi in fi_list:
                _fps = fps_per_video.get(video, FPS)
                t = src_fi / _fps
                gw = gt_lookup(segs, video, t)
                # 仅评误绿: 管线输出绿 且 GT 非绿
                if gw is not False:
                    src_fi += 1
                    continue
                rec = recs[fi_list.index(src_fi)]
                cands_i = rec["candidates"]
                if not cands_i:
                    src_fi += 1
                    continue
                H, W = frame.shape[:2]
                fi_idx = fi_list.index(src_fi)
                temporal_scores = ts_list[fi_idx]
                sel = select_gtfree(cands_i, PED_PRIOR, temporal_scores)
                prior_box = box_from_prior(priors[video], W, H)
                roi_prior = crop_cv2(frame, prior_box)
                out_prior_color = _color_state_roi(roi_prior)
                sel_col = _color_state_roi(crop_cv2(frame, tuple(sel["box_norm"]))) if sel else None
                if sel and sel_col == "green":
                    n_falsegreen += 1
                    sel_c = center(sel["box_norm"])
                    # 原: 静态 true_center
                    tc = derived.get(video, {}).get("true_center")
                    if tc and dist(sel_c, tc) > DIST_TH:
                        c_s = "车灯误选"
                    elif out_prior_color == "green":
                        c_s = "crop偏框"
                    else:
                        c_s = "状态判错"
                    cause_static[c_s] += 1
                    per_video_static[video][c_s] += 1
                    # 新: 逐帧 GT 框(有则用, 否则 fallback 静态)
                    tb = ann_by_vf.get((video, src_fi))
                    ref = center(tb) if tb else tc
                    if ref and dist(sel_c, ref) > DIST_TH:
                        c_p = "车灯误选"
                    elif out_prior_color == "green":
                        c_p = "crop偏框"
                    else:
                        c_p = "状态判错"
                    cause_perframe[c_p] += 1
                    per_video_perframe[video][c_p] += 1
                    if c_p != c_s:
                        reclassified.append({
                            "video": video, "fi": src_fi, "has_gt_frame": tb is not None,
                            "sel_center": [round(x, 3) for x in sel_c],
                            "static_center": [round(x, 3) for x in tc] if tc else None,
                            "gt_frame_center": [round(x, 3) for x in center(tb)] if tb else None,
                            "prior_color": out_prior_color, "from": c_s, "to": c_p,
                        })
            src_fi += 1
        cap.release()
        print(f"  [{vi+1}/{len(videos)}] {video} 用时 {time.time()-t0:.0f}s")

    # 汇总
    print(f"\n=== R1 误绿帧总数(重放)= {n_falsegreen} (measure 报告 R1 falsegreen=71? 注: 此处含所有 gw=False 且 sel绿, 与 measure 口径应一致) ===")
    print(f"\n=== 成因归因对比(去混淆) ===")
    print(f"{'成因':10} {'原(静态true_center)':>20} {'新(逐帧GT框)':>20}")
    allk = ["车灯误选", "crop偏框", "状态判错"]
    for k in allk:
        print(f"{k:10} {cause_static.get(k,0):>20} {cause_perframe.get(k,0):>20}")
    tot_s = sum(cause_static.values()); tot_p = sum(cause_perframe.values())
    print(f"{'合计':10} {tot_s:>20} {tot_p:>20}")

    print(f"\n=== per-video 新归因(逐帧GT) ===")
    for v in sorted(per_video_perframe):
        print(f"  {v}: {dict(per_video_perframe[v])}")

    print(f"\n=== 被重分类帧(共 {len(reclassified)}) ===")
    for r in reclassified:
        tag = "逐帧GT" if r["has_gt_frame"] else "fallback静态"
        print(f"  {r['video']} fi={r['fi']} [{tag}] {r['from']} -> {r['to']}  sel={r['sel_center']} gtFrame={r['gt_frame_center']} static={r['static_center']}")

    out = {
        "schema": "recompute_cause_perframe_v1",
        "n_falsegreen": n_falsegreen,
        "cause_static": dict(cause_static),
        "cause_perframe": dict(cause_perframe),
        "per_video_perframe": {v: dict(c) for v, c in per_video_perframe.items()},
        "reclassified": reclassified,
    }
    op = ROOT / "data" / "output" / "recompute_cause_perframe.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {op}")


if __name__ == "__main__":
    main()
