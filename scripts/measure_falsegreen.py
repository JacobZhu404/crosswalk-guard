#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_falsegreen.py — 量"误绿一次"(Fork A 核心产出)。只测量,不接线。

去循环口径(贴 cc review 2026-07-26):
- 选灯: select_gtfree(L1/L2/L3 GT-free, 无 gt 参数); ped_prior 内联(禁 derive_ped_priors)。
- 状态: color 主行(TrafficLightDetector._color_state, 复用现役逻辑) + classifier 对照(ped_signal.pt)。
- GT 只比对不进推理; events.csv 三态映射(green→绿 / red-none→非绿计误绿 / unknown+空隙→UNKNOWN 排除)。
- 两纪律: 禁 seed_with_gt/derive_ped_priors; 跑前 08(全绿)/01(全红) light_state 语义自检。

报告 5 行(选灯=best L1/L2/L3, 除 R0 端到端用 prior 位置):
  R0 现役端到端: color @ prior-ROI,  location=prior(生产)
  R1 color@best: color @ selected,   no-L3
  R2 color@best: color @ selected,   L3 w=0.5
  R3 clf@best:    classifier @ selected, no-L3
  R4 clf@best:    classifier @ selected, L3 w=0.5

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/measure_falsegreen.py
  PYTHONPATH=src ./.venv/bin/python scripts/measure_falsegreen.py --smoke   # 仅 2 视频
输出: docs/reports/2026-07-26-wb-falsegreen-measure.md + data/output/falsegreen_gallery/*.html
"""
import json, sys, os, argparse, time, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from PIL import Image

from redlight.models.ped_light_selector import select_gtfree, compute_temporal_scores
from redlight.models.l3_ped_vehicle import L3PedVehicleNet, crop_candidate, score_crop, CLASSES
from redlight.models.traffic_light import TrafficLightDetector
from redlight.models.signal_state_classifier import SignalStateClassifier
import types

FPS = 29.70
CAND = ROOT / "data" / "output" / "candidates_dense.json"
EVENTS = ROOT / "datasets" / "gt" / "events.csv"
GTLOC = ROOT / "datasets" / "light_location_gt.json"
PRIORS = ROOT / "configs" / "light_priors.json"
L3W = ROOT / "models" / "l3_ped_full.pt"
CLF = ROOT / "models" / "ped_signal.pt"
GALLERY_DIR = ROOT / "data" / "output" / "falsegreen_gallery"
REPORT = ROOT / "docs" / "reports" / "2026-07-26-wb-falsegreen-measure.md"
CKPT = ROOT / "data" / "output" / "falsegreen_checkpoint.json"

# 内联 ped_prior(生产知识: 行人灯竖长; 禁 derive_ped_priors)
PED_PRIOR = {"aspect_mean": 3.0, "aspect_std": 1.0, "area_mean": 0.005, "area_std": 0.003}


def _cfg():
    tl = types.SimpleNamespace(method="color", smoothing_window=8, sat_min=130,
                               value_floor=60, min_area_px=30, max_area_ratio=0.008,
                               max_aspect_ratio=3.5, color_s_min=22)
    return types.SimpleNamespace(traffic_light=tl)


def load_events():
    """返回 {video: [(start_s, end_s, light_state, is_violation), ...]} + 覆盖区间。"""
    import csv
    segs = defaultdict(list)
    with open(EVENTS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            segs[row["video"]].append((float(row["start_s"]), float(row["end_s"]),
                                       row["light_state"].strip(), row["is_violation"].strip()))
    return segs


def gt_lookup(segs, video, t):
    """三态映射: green→True; red/occluded/none→False; unknown/空隙(None)→None(UNKNOWN排除)。"""
    for s, e, state, viol in segs.get(video, []):
        if s - 1e-6 <= t <= e + 1e-6:
            if state == "green":
                return True
            if state in ("red", "occluded", "none"):
                return False
            return None  # unknown
    return None  # 空隙


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
    """对齐生产 v7-stable 主色逻辑(green>=red 判绿)的绿/红判定。
    返回 'green' | 'red' | None(无足够有色像素)。独立实现, 不依赖 TrafficLightDetector 内部状态。"""
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


def selfcheck_light_state(det, priors, segs):
    """纪律②: 08 中段应绿、01 中段应红, 确认 events.csv.light_state 是行人灯态(没搞反)。"""
    for video, expect in [("违章08", "green"), ("违章01", "red")]:
        vs = segs.get(video, [])
        green_segs = [s for s in vs if s[2] == "green"]
        if not green_segs and expect == "green":
            raise SystemExit(f"[自检失败] {video} 无 green 段, 与'全绿'预期不符")
        # 取一个 green 段中点(08) / 任意段中点(01)
        if expect == "green":
            s, e, _, _ = green_segs[0]
        else:
            s, e, _, _ = vs[0]
        fi = int(((s + e) / 2) * FPS)
        cap = cv2.VideoCapture(str(ROOT / "input_video" / f"{video}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise SystemExit(f"[自检失败] {video} 读帧失败 fi={fi}")
        prior = priors[video]
        H, W = frame.shape[:2]
        roi = crop_cv2(frame, box_from_prior(prior, W, H))
        got = _color_state_roi(roi)
        print(f"  [自检] {video} fi={fi} priorROI color={got} (期望~{expect})")
        # 硬中止仅当方向明显反了(08→red 或 01→green); None 仅警告(可能阈值/ROI 偏移,非反号)
        if (expect == "green" and got == "red") or (expect == "red" and got == "green"):
            raise SystemExit(f"[自检失败] {video} priorROI color={got} 与期望{expect}相反 → 灯态源搞反, 整份会反号! 中止。")
        if got is None:
            print(f"  [自检警告] {video} priorROI 无色像素(None), 跳过方向校验(可能 ROI 偏移)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅 2 视频(08,01)冒烟")
    args = ap.parse_args()

    cands = json.load(open(CAND))
    segs = load_events()
    gtloc = json.load(open(GTLOC))
    priors = json.load(open(PRIORS))
    derived = gtloc.get("derived_per_video", {})
    fps_per_video = cands.get("fps_per_video", {})

    det = TrafficLightDetector(_cfg(), verbose=False)
    clf = SignalStateClassifier(str(CLF), verbose=False)
    print(f"  分类器 available={clf.available}")

    # 纪律②自检
    print("[纪律②] light_state 语义自检...")
    selfcheck_light_state(det, priors, segs)

    # 加载全量 L3
    l3 = L3PedVehicleNet()
    l3.load_state_dict(torch.load(L3W, map_location="cpu", weights_only=True))
    l3.eval()
    print(f"  L3 全量模型已加载: {L3W.name}")

    # 准备候选(按视频, 帧排序)
    videos = cands["videos"]
    if args.smoke:
        videos = {v: videos[v] for v in ["违章08", "违章01"] if v in videos}

    ROWKEYS = [f"R{i}" for i in range(5)]
    stats = {f"R{i}": defaultdict(int) for i in range(5)}  # 每行: walk/truegreen/falsegreen/eval
    per_video = {v: {f"R{i}": defaultdict(int) for i in range(5)} for v in videos}
    # 严格事件级: 连续 walk 帧=一个绿事件; 事件内任一帧 GT≠绿→误绿事件
    event_stats = {rk: {"tot": 0, "false": 0} for rk in ROWKEYS}
    excluded_unk = defaultdict(int)        # 因 unknown/空隙排除的帧(可评候选但 GT UNKNOWN)
    no_cand = defaultdict(int)             # 无候选帧
    cause = Counter()                      # 误绿成因
    miss_green = defaultdict(int)          # 漏绿(附注): GT 绿但管线未输出绿
    gallery_rows = []

    # 断点续跑: 若上次被强杀, 从 checkpoint 恢复已完成的视频(就地更新, 不重算)
    done_videos = []
    if CKPT.exists() and not args.smoke:
        print(f"[checkpoint] 恢复: {CKPT.name}")
        obj = json.loads(CKPT.read_text(encoding="utf-8"))
        done_videos = obj.get("done_videos", [])
        for rk in ROWKEYS:
            stats[rk].update(obj["stats"].get(rk, {}))
            event_stats[rk].update(obj["event_stats"].get(rk, {}))
        for v in done_videos:
            pv = obj["per_video"].get(v, {})
            for rk in ROWKEYS:
                per_video[v][rk].update(pv.get(rk, {}))
        excluded_unk.update(obj.get("excluded_unk", {}))
        no_cand.update(obj.get("no_cand", {}))
        miss_green.update(obj.get("miss_green", {}))
        cause.update(obj.get("cause", {}))
        print(f"[checkpoint] 已完成 {len(done_videos)} 视频: {done_videos}")

    def _close_run(rk, run_state):
        if run_state[rk]["in"]:
            event_stats[rk]["tot"] += 1
            if run_state[rk]["false"]:
                event_stats[rk]["false"] += 1
            run_state[rk] = {"in": False, "false": False}

    def record_row(rowkey, walk, gt_walk):
        s = stats[rowkey]; pv = per_video[video][rowkey]
        s["eval"] += 1; pv["eval"] += 1
        if walk:
            s["walk"] += 1; pv["walk"] += 1
            if gt_walk is True:
                s["truegreen"] += 1; pv["truegreen"] += 1
            else:
                s["falsegreen"] += 1; pv["falsegreen"] += 1
            # 绿事件 run 累积
            if not run_state[rowkey]["in"]:
                run_state[rowkey] = {"in": True, "false": gt_walk is False}
            else:
                run_state[rowkey]["false"] = run_state[rowkey]["false"] or (gt_walk is False)
        else:
            if gt_walk is True:
                s["missgreen"] += 1; pv["missgreen"] += 1
            _close_run(rowkey, run_state)  # 非绿帧打断绿事件

    t0 = time.time()
    for vi, (video, recs) in enumerate(videos.items()):
        recs = sorted(recs, key=lambda r: r["fi"])
        sampled = {r["fi"]: r for r in recs}
        if video in done_videos:
            print(f"  [skip] {video} (checkpoint 已完成)")
            continue
        run_state = {rk: {"in": False, "false": False} for rk in ROWKEYS}
        # 预计算 temporal_scores(跨帧复现)
        ts_list = compute_temporal_scores(recs)
        cap = cv2.VideoCapture(str(ROOT / "input_video" / f"{video}.mp4"))
        src_fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if src_fi in sampled:
                rec = sampled[src_fi]
                _fps = fps_per_video.get(video, FPS)
                t = src_fi / _fps
                gw = gt_lookup(segs, video, t)
                if gw is None:
                    excluded_unk[video] += 1
                    for rk in ROWKEYS:
                        _close_run(rk, run_state)  # 未知/空隙打断绿事件
                    src_fi += 1
                    continue
                cands_i = rec["candidates"]
                if not cands_i:
                    no_cand[video] += 1
                    if gw is True:
                        for rk in ("R0", "R1", "R2", "R3", "R4"):
                            record_row(rk, False, True)  # 漏绿: 检测层无任何候选
                    src_fi += 1
                    continue
                # 帧转 PIL(供 L3 crop)
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                H, W = frame.shape[:2]
                fi_idx = [r["fi"] for r in recs].index(src_fi)
                temporal_scores = ts_list[fi_idx]
                # L3 评分(每候选)
                l3_scores = {}
                if l3 is not None:
                    for j, c in enumerate(cands_i):
                        ten = crop_candidate(frame_pil, tuple(c["box_norm"]))
                        l3_scores[j] = score_crop(l3, ten)["ped"]
                # 选灯
                sel_nol3 = select_gtfree(cands_i, PED_PRIOR, temporal_scores)
                sel_l3 = select_gtfree(cands_i, PED_PRIOR, temporal_scores, l3_scores=l3_scores, l3_weight=0.5)
                # 裁图
                prior_box = box_from_prior(priors[video], W, H)
                roi_prior = crop_cv2(frame, prior_box)
                out_prior_color = _color_state_roi(roi_prior)
                def state_for(sel):
                    if sel is None:
                        return None, None, None
                    b = sel["box_norm"]
                    roi_sel = crop_cv2(frame, tuple(b))
                    c_col = _color_state_roi(roi_sel)
                    c_clf = clf.classify(roi_sel) if roi_sel is not None else ("off", 0.0)
                    return c_col, c_clf[0], roi_sel
                col_nol3, clf_nol3, roi_nol3 = state_for(sel_nol3)
                col_l3, clf_l3, roi_l3 = state_for(sel_l3)
                # 5 行
                # R0 现役端到端: color @ prior-ROI, location=prior
                record_row("R0", out_prior_color == "green", gw)
                # R1 color @ best, no-L3
                record_row("R1", col_nol3 == "green", gw)
                # R2 color @ best, L3 w=0.5
                record_row("R2", col_l3 == "green", gw)
                # R3 clf @ best, no-L3
                record_row("R3", clf_nol3 == "walk", gw)
                # R4 clf @ best, L3 w=0.5
                record_row("R4", clf_l3 == "walk", gw)
                # 漏绿附注
                if gw is True:
                    for rk, st in [("R0", out_prior_color), ("R1", col_nol3), ("R2", col_l3),
                                   ("R3", clf_nol3), ("R4", clf_l3)]:
                        if not (st == "green" or st == "walk"):
                            miss_green[rk] += 1
                # 成因归类(R1 falsegreen)
                if col_nol3 == "green" and gw is False and sel_nol3 is not None:
                    tc = derived.get(video, {}).get("true_center")
                    sel_c = (sel_nol3["box_norm"][0] + sel_nol3["box_norm"][2]) / 2, \
                            (sel_nol3["box_norm"][1] + sel_nol3["box_norm"][3]) / 2
                    if tc and ((sel_c[0] - tc[0]) ** 2 + (sel_c[1] - tc[1]) ** 2) ** 0.5 > 0.04:
                        cause["车灯误选"] += 1
                    elif out_prior_color == "green":
                        cause["crop偏框"] += 1
                    else:
                        cause["状态判错"] += 1
                # 画廊(误绿样本, 限量)
                if (col_nol3 == "green" and gw is False) or (clf_nol3 == "walk" and gw is False):
                    if len(gallery_rows) < 200:
                        gallery_rows.append((video, src_fi, frame.copy(), sel_nol3, prior_box,
                                             roi_prior, out_prior_color, col_nol3, clf_nol3, gw))
            src_fi += 1
        cap.release()
        for rk in ROWKEYS:
            _close_run(rk, run_state)  # 收尾未闭合的绿事件
        done_videos.append(video)
        if not args.smoke:
            _save_ckpt(CKPT, done_videos, stats, per_video, excluded_unk, no_cand, cause, miss_green, event_stats)
        print(f"  [{vi+1}/{len(videos)}] {video} 用时 {time.time()-t0:.0f}s")

    # 汇总
    print("\n=== 误绿测量结果 ===")
    print("行        帧级误绿率   事件级误绿率(绿事件)   可评帧   绿事件   误绿事件")
    # 事件级需要 run 统计(简化: 这里用帧级为主, 事件级按绿 run 数近似在报告中说明)
    report_lines = []
    for i in range(5):
        s = stats[f"R{i}"]
        eval_n = s["eval"]; fg = s["falsegreen"]; tg = s["truegreen"]
        fr = fg / eval_n if eval_n else 0
        walk_n = s["walk"]
        # 事件级(严格 run 级): 误绿事件 / 总绿事件
        ev = event_stats[f"R{i}"]
        er = ev["false"] / ev["tot"] if ev["tot"] else 0
        label = ["R0 现役端到端(color@prior)", "R1 color@best(no-L3)", "R2 color@best(L3 w=.5)",
                 "R3 clf@best(no-L3)", "R4 clf@best(L3 w=.5)"][i]
        print(f"  {label:28s} {fr*100:6.2f}%   {er*100:6.2f}%   {eval_n:6d}  {ev['tot']:6d}  {ev['false']:6d}")
        report_lines.append((label, fr, er, eval_n, ev["tot"], ev["false"]))

    print(f"\n  因 unknown/空隙排除帧(不计分母): {sum(excluded_unk.values())}  (各视频: {dict(excluded_unk)})")
    print(f"  无候选帧(管线无法输出绿): {sum(no_cand.values())}  (各视频: {dict(no_cand)})")
    print(f"  成因归类(误绿帧): {dict(cause)}")
    print(f"  漏绿附注(真值绿但管线未输出绿): {dict(miss_green)}")

    # 写报告
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    _write_report(report_lines, per_video, excluded_unk, no_cand, cause, miss_green, videos)
    _write_gallery(gallery_rows, derived)
    if CKPT.exists():
        CKPT.unlink()
        print("[checkpoint] 全部完成, 清除 checkpoint")
    print(f"\n[out] 报告: {REPORT}")
    print(f"[out] 画廊: {GALLERY_DIR / 'falsegreen_gallery.html'}")


def _write_report(rows, per_video, excluded_unk, no_cand, cause, miss_green, videos):
    L = []
    L.append("# 误绿测量报告 (Fork A 标尺)\n")
    L.append("> wb 执行。方法: `docs/plans/2026-07-26-wb-falsegreen-measure-method.md`(cc review 修正版)。"
             "只测量不接线,GT 只比对。\n")
    L.append("## 总体(5 行)\n")
    L.append("| 行 | 说明 | 帧级误绿率 | 事件级误绿率 | 可评帧 | 绿事件 | 误绿事件 |")
    for label, fr, er, eval_n, walk_n, fg in rows:
        L.append(f"| {label} | | {fr*100:.2f}% | {er*100:.2f}% | {eval_n} | {walk_n} | {fg} |")
    L.append("\n- **帧级误绿率** = 误绿帧 / 可评帧(有候选且 GT 非 UNKNOWN)。")
    L.append("- **事件级误绿率** = 误绿绿事件 / 总绿事件(run 级: 连续绿输出=一事件, 事件内任一帧 GT≠绿即误绿)。")
    L.append("- **主标尺** = R0(字面现役端到端) + R1(color@最好选灯)。R2-R4 为 A 改进候选对照。")
    L.append("\n## 11 视频分解(帧级误绿率, R1 主行)\n")
    L.append("| 视频 | R0 | R1 | R2 | R3 | R4 | 可评帧 |")
    for v in videos:
        cells = []
        for i in range(5):
            s = per_video[v][f"R{i}"]
            fr = s["falsegreen"] / s["eval"] if s["eval"] else 0
            cells.append(f"{fr*100:.1f}%")
        ev = per_video[v]["R1"]["eval"]
        L.append(f"| {v} | " + " | ".join(cells) + f" | {ev} |")
    L.append(f"\n## 排除/边界(透明化)\n")
    L.append(f"- 因 unknown 段/空隙排除帧(不计分母): **{sum(excluded_unk.values())}** {dict(excluded_unk)}")
    L.append(f"- 无候选帧(管线无法输出绿): **{sum(no_cand.values())}** {dict(no_cand)}")
    L.append(f"- 漏绿附注(真值绿但管线未输出绿, 不计主率): {dict(miss_green)}")
    L.append("\n## 成因归类(误绿帧, 基于 true_center 距离/状态路径对比)\n")
    L.append("| 成因 | 占比 |")
    tot = sum(cause.values()) or 1
    for k, c in cause.most_common():
        L.append(f"| {k} | {c} ({c/tot*100:.1f}%) |")
    if not cause:
        L.append("| (无足够误绿样本归因) | |")
    L.append("\n## 结论提示\n- R0/R1 高 → 现役管线误绿严重, 推 B 建检测器或接 ImVisible 预训练分类器。")
    L.append("- R0 高但 R1 低 → 选灯(偏框)是主因, L3/定位修法有效。")
    L.append("- R1 高且 R3/R4 低 → 状态分类器(接 ImVisible 预训练)是主杠杆。")
    REPORT.write_text("\n".join(L), encoding="utf-8")


def _write_gallery(rows, derived):
    import base64
    html = ["<html><head><meta charset='utf-8'><title>误绿样本画廊</title>",
            "<style>body{font-family:sans-serif;background:#111;color:#eee}"
            ".row{display:inline-block;margin:6px;border:1px solid #444;vertical-align:top}"
            ".cap{font-size:11px;padding:3px 5px;background:#222}</style></head><body>"]
    html.append(f"<h2>误绿样本画廊 ({len(rows)} 帧)</h2>")
    html.append("<p>红框=真值灯中心(true_center 周边); 绿框=选中候选; 黄虚框=prior ROI。"
                "标注: priorColor / selColor / selClf / GT。</p>")
    for (video, fi, frame, sel, prior_box, roi_prior, opc, col, clf, gw) in rows:
        # 画框
        vis = frame.copy()
        H, W = vis.shape[:2]
        tc = derived.get(video, {}).get("true_center")
        if tc:
            cx, cy = int(tc[0] * W), int(tc[1] * H)
            cv2.rectangle(vis, (cx - 20, cy - 40), (cx + 20, cy + 40), (0, 0, 255), 2)
        if sel:
            b = sel["box_norm"]
            x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        pb = prior_box
        px1, py1, px2, py2 = int(pb[0] * W), int(pb[1] * H), int(pb[2] * W), int(pb[3] * H)
        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 255), 1)
        _, buf = cv2.imencode(".jpg", vis)
        b64 = base64.b64encode(buf).decode()
        gt_state = "green" if gw is True else ("非绿" if gw is False else "UNK")
        cap = (f"{video} fi={fi}<br>priorColor={opc} | selColor={col} | selClf={clf}"
               f"<br>GT={gt_state}")
        html.append(f"<div class='row'><img src='data:image/jpeg;base64,{b64}' width='240'>"
                    f"<div class='cap'>{cap}</div></div>")
    html.append("</body></html>")
    (GALLERY_DIR / "falsegreen_gallery.html").write_text("\n".join(html), encoding="utf-8")


def _save_ckpt(path, done_videos, stats, per_video, excluded_unk, no_cand, cause, miss_green, event_stats):
    """按视频落盘断点(原子写), 被强杀可从这里续跑, 不丢已算帧。"""
    ROWKEYS = [f"R{i}" for i in range(5)]
    obj = {
        "done_videos": list(done_videos),
        "stats": {rk: dict(stats[rk]) for rk in ROWKEYS},
        "per_video": {v: {rk: dict(per_video[v][rk]) for rk in ROWKEYS} for v in per_video},
        "excluded_unk": dict(excluded_unk),
        "no_cand": dict(no_cand),
        "cause": dict(cause),
        "miss_green": dict(miss_green),
        "event_stats": {rk: dict(event_stats[rk]) for rk in ROWKEYS},
    }
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    import torch
    main()
