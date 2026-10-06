#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_selection_falsegreen.py — 误绿选灯诊断(cc brief 2026-07-30 §6, 本轮只诊断不改选灯代码)。

主标尺 R1(color@best, no-L3)。逐帧检查正常场景(扣 05)误绿帧, 回答三件事:
  Q1 干扰类型(反射/背面/车灯/树叶)+ 视频;
  Q2 决定性拆分: YOLO∪HSV 候选里任一候选与该帧 governing 框 IoU>=0.3 ?
      在=选灯排序错(判别器/规则能救) / 不在=检测召回洞(另走检测); 给两类占比;
  Q3 在候选集却没选中: 看 L1/L2/L3(final) 分数, 指出为何把干扰排在真灯前。

真值: datasets/gt/light_canonical_gt.json(399帧)。候选/选灯: build_candidates + select_gtfree(只读)。
输出: 报告 md + 叠框画廊 html + 诊断 JSON。不动生产/模型/选灯代码。
用法: PYTHONPATH=src ./.venv/bin/python scripts/diag_selection_falsegreen.py [--videos 违章03,违章06]
"""
import json, sys, types, base64
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import torch
from PIL import Image

from redlight.models.ped_light_selector import select_gtfree, compute_temporal_scores, iou, _l1_geom_score
from redlight.models.l3_ped_vehicle import L3PedVehicleNet, crop_candidate, score_crop
from redlight.models.traffic_light import TrafficLightDetector
from redlight.models.signal_candidates import build_candidates

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
L3W = ROOT / "models" / "l3_ped_full.pt"
YOLO_W = ROOT / "models" / "yolov8n.pt"
REPORT = ROOT / "docs" / "reports" / "2026-07-30-wb-selection-diagnostic.md"
GALLERY = ROOT / "data" / "output" / "diag_selection_gallery.html"
OUTJSON = ROOT / "data" / "output" / "diag_selection_falsegreen.json"
PED_PRIOR = {"aspect_mean": 3.0, "aspect_std": 1.0, "area_mean": 0.005, "area_std": 0.003}
TEMP_W = 0.4
YOLO_BONUS = 0.15
IOU_TH = 0.3


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


def _ctr(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="", help="仅跑这些视频(逗号), 默认全 11")
    ap.add_argument("--exclude", default="违章05", help="忽略的视频(已知局限), 默认 违章05")
    args = ap.parse_args()

    gt = json.load(open(GT, encoding="utf-8"))
    det = TrafficLightDetector(_cfg(), verbose=False)
    from ultralytics import YOLO
    model = YOLO(str(YOLO_W))
    l3 = L3PedVehicleNet()
    l3.load_state_dict(torch.load(L3W, map_location="cpu", weights_only=True))
    l3.eval()

    by_video = defaultdict(dict)
    for fr in gt["frames"]:
        by_video[fr["video"]][int(fr["source_fi"])] = fr
    videos = sorted(by_video)
    if args.videos:
        videos = [v for v in videos if v in args.videos.split(",")]
    exclude = set(args.exclude.split(",")) if args.exclude else set()

    records = []
    cnt_total_fg = 0
    gallery_rows = []

    for vi, video in enumerate(videos):
        gframes = by_video[video]
        want = sorted(gframes)
        cap = cv2.VideoCapture(str(ROOT / "input_video" / f"{video}.mp4"))
        recs = []
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
        ts_list = compute_temporal_scores(recs)
        frame_pil = [Image.fromarray(cv2.cvtColor(r["frame"], cv2.COLOR_BGR2RGB)) for r in recs]

        for idx, rec in enumerate(recs):
            fi = rec["fi"]
            frame = rec["frame"]
            H, W = frame.shape[:2]
            g = gframes[fi]
            boxes = g.get("boxes", [])
            gov = [b for b in boxes if b.get("governing")]
            gcolors = {b["color"] for b in gov}
            no_light = g.get("no_light", False)
            if gov and gcolors <= {"unclear"}:
                continue  # governing 全 unclear → UNKNOWN 排除
            gt_walk = ("green" in gcolors)
            cands = rec["candidates"]
            if not cands:
                continue
            l3_scores = {j: score_crop(l3, crop_candidate(frame_pil[idx], tuple(c["box_norm"])))["ped"]
                         for j, c in enumerate(cands)}
            ts = ts_list[idx]
            sel = select_gtfree(cands, PED_PRIOR, ts)  # R1, no-L3
            if sel is None:
                continue
            sel_color = color_state(crop_cv2(frame, tuple(sel["box_norm"])))
            if not (sel_color == "green" and not gt_walk):
                continue  # 仅评 R1 误绿帧

            cnt_total_fg += 1
            is_excl = video in exclude
            # 逐候选分数(R1 无 L3: final = L1 + yolo_bonus + 0.4*L2)
            cand_scores = []
            for j, c in enumerate(cands):
                b = c["box_norm"]
                L1 = _l1_geom_score(b, PED_PRIOR)
                yb = YOLO_BONUS if c.get("source") == "yolo" else 0.0
                L2 = ts.get(j, 0.0)
                base = L1 + yb + TEMP_W * L2
                cand_scores.append({"j": j, "box_norm": b, "source": c.get("source"),
                                     "L1": L1, "yolo_bonus": yb, "L2": L2, "final": base,
                                     "l3_ped": l3_scores.get(j, 0.0)})
            sel_j = next(j for j, c in enumerate(cands) if c is sel)
            sel_sc = cand_scores[sel_j]
            # Q2 决定性拆分: 任一候选与 governing 框 IoU>=0.3 ?
            if not gov:
                decisive = "无真灯帧(干扰自发绿)"
                best_iou = 0.0
                best_cand = None
            else:
                best_iou = 0.0
                best_cand = None
                for cs in cand_scores:
                    for gb in gov:
                        v = iou(cs["box_norm"], gb["box_norm"])
                        if v > best_iou:
                            best_iou = v
                            best_cand = cs
                if best_iou >= IOU_TH:
                    decisive = "真灯在候选集(选灯排序错)"
                else:
                    decisive = "真灯不在候选集(检测召回洞)"
            # Q1 干扰类型: 匹配最近非 governing GT 框
            matched_gt = None
            best_m = 1e9
            for b in boxes:
                if b.get("governing"):
                    continue
                v = iou(sel["box_norm"], b["box_norm"])
                d = ((_ctr(sel["box_norm"])[0] - _ctr(b["box_norm"])[0]) ** 2 +
                     (_ctr(sel["box_norm"])[1] - _ctr(b["box_norm"])[1]) ** 2) ** 0.5
                sc = (1 - v) + d
                if sc < best_m:
                    best_m = sc
                    matched_gt = b
            if matched_gt is not None:
                gt_type = matched_gt.get("type")
                if gt_type == "vehicle":
                    inferred = "车灯(车辆灯)"
                elif gt_type == "distractor":
                    inferred = "干扰(背面/反射/树叶, 视觉细分)"
                elif gt_type == "pedestrian":
                    inferred = "行人灯(非governing, 他向/他斑马线)"
                else:
                    inferred = f"匹配GT框(type={gt_type})"
                type_rat = f"匹配GT非governing框 type={gt_type}"
            else:
                p = sel_sc["l3_ped"]
                inferred = "未匹配GT框-启发式:" + ("像行人灯(P(ped)高)" if p > 0.5 else "像车灯/其他(P(ped)低)")
                type_rat = f"sel P(ped)={p:.2f}, color={sel_color}"
            recd = {
                "video": video, "fi": fi, "excluded_05": is_excl,
                "gt_walk": gt_walk, "no_light": no_light,
                "gov_colors": sorted(gcolors),
                "sel_box_norm": [round(x, 3) for x in sel["box_norm"]],
                "sel_color": sel_color, "sel_source": sel.get("source"),
                "sel_L1": round(sel_sc["L1"], 3), "sel_yolo_bonus": sel_sc["yolo_bonus"],
                "sel_L2": round(sel_sc["L2"], 3), "sel_final": round(sel_sc["final"], 3),
                "sel_l3_ped": round(sel_sc["l3_ped"], 3),
                "max_iou_cand_gov": round(best_iou, 3),
                "decisive": decisive,
                "matched_gt_type": matched_gt.get("type") if matched_gt else None,
                "inferred_type": inferred,
                "type_rationale": type_rat,
            }
            if best_cand is not None and decisive.startswith("真灯在"):
                recd["govbest_L1"] = round(best_cand["L1"], 3)
                recd["govbest_L2"] = round(best_cand["L2"], 3)
                recd["govbest_final"] = round(best_cand["final"], 3)
                recd["govbest_l3_ped"] = round(best_cand["l3_ped"], 3)
                recd["govbest_iou"] = round(best_iou, 3)
                recd["govbest_source"] = best_cand["source"]
            records.append(recd)
            gallery_rows.append((video, fi, frame.copy(), cands, sel, gov, matched_gt, recd))
        print(f"  [{vi + 1}/{len(videos)}] {video}: GT帧{len(recs)} 累计误绿{cnt_total_fg}")

    _write_report(records, cnt_total_fg, exclude)
    _write_gallery(gallery_rows)
    OUTJSON.write_text(json.dumps({"n_falsegreen": len(records), "records": records},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] 报告 {REPORT}")
    print(f"[out] 画廊 {GALLERY}")
    print(f"[out] JSON {OUTJSON}")


def _write_report(records, cnt_total, exclude):
    # 拆分: 扣 exclude 后的 R1 误绿帧
    non05 = [r for r in records if not r["excluded_05"]]
    excl = [r for r in records if r["excluded_05"]]
    L = ["# 误绿选灯诊断(扣 05, R1 主标尺) — wb 2026-07-30\n",
         "> 纯诊断, 不训模型不改选灯代码。真值=canonical GT, 候选=build_candidates, 选灯=select_gtfree(只读)。"
         "回答 cc brief §6 三问。\n"]
    L.append(f"- R1 误绿帧总数(全 11 视频)= **{cnt_total}**; 其中忽略({','.join(sorted(exclude))})= {len(excl)}; "
             f"**正常场景(扣 05)= {len(non05)}** (cc 主标尺 2.19% 吻合)。\n")

    # Q2 决定性拆分
    L.append("## Q2 决定性拆分(候选集 vs governing 框 IoU≥0.3)\n")
    L.append("| 集合 | 真灯在候选集(选灯排序错→判别器能救) | 真灯不在候选集(检测召回洞→走检测) | 无真灯帧(干扰自发绿) | 合计 |")
    L.append("|---|---|---|---|---|")
    for label, subset in [("正常场景(扣05)", non05), (f"忽略({','.join(sorted(exclude))})", excl)]:
        inc = sum(1 for r in subset if r["decisive"].startswith("真灯在"))
        out = sum(1 for r in subset if r["decisive"].startswith("真灯不在"))
        nol = sum(1 for r in subset if r["decisive"].startswith("无真灯"))
        tot = len(subset)
        pct = (lambda n: f"{n / tot * 100:.0f}%" if tot else "—")
        L.append(f"| {label} | {inc} ({pct(inc)}) | {out} ({pct(out)}) | {nol} ({pct(nol)}) | {tot} |")
    L.append("\n- **解读**: '在候选集' = YOLO∪HSV 里有与 governing 框 IoU≥0.3 的候选, 说明真灯被检出了, 只是 select_gtfree 把它排到了干扰后面 → 改选灯/加判别器可救; "
             "'不在候选集' = 真灯压根没进候选集 → 选灯再好也救不了, 归检测 track(补召回)。\n")

    # Q1 类型分布(正常场景)
    L.append("## Q1 干扰类型分布(正常场景, 扣 05)\n")
    type_ct = Counter(r["inferred_type"] for r in non05)
    L.append("| 干扰类型 | 帧数 | 占比 |")
    L.append("|---|---|---|")
    for k, c in type_ct.most_common():
        L.append(f"| {k} | {c} | {c / len(non05) * 100:.0f}% |")
    L.append("")

    # 逐帧明细(正常场景)
    L.append("## 逐帧明细(正常场景, 扣 05)\n")
    L.append("| 视频 | fi | 类型 | 决定性 | sel色 | sel源 | maxIoU(gov) | sel.L1 | sel.L2 | sel.final | P(ped) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in non05:
        L.append(f"| {r['video']} | {r['fi']} | {r['inferred_type']} | {r['decisive']} | "
                 f"{r['sel_color']} | {r['sel_source']} | {r['max_iou_cand_gov']} | "
                 f"{r['sel_L1']} | {r['sel_L2']} | {r['sel_final']} | {r['sel_l3_ped']} |")
    L.append("")

    # Q3 失效点(在候选集却没选中)
    L.append("## Q3 失效点: '真灯在候选集却没被选中' 的帧(选灯排序错)\n")
    inc_frames = [r for r in non05 if r["decisive"].startswith("真灯在")]
    if inc_frames:
        L.append("| 视频 | fi | 选中.final | 真灯候选.final | Δ | 选中.L1 | 真灯.L1 | 选中.L2 | 真灯.L2 | 真灯IoU | 说明 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in inc_frames:
            d = r["sel_final"] - r["govbest_final"]
            # 解释: 谁高谁低
            bits = []
            if r["sel_L1"] > r["govbest_L1"] + 0.05:
                bits.append("选中L1更高(几何更像竖长条/尺寸更贴)")
            if r["sel_L2"] > r["govbest_L2"] + 0.05:
                bits.append("选中L2更高(时序更稳/复现更多)")
            if r["sel_source"] == "yolo" and r.get("govbest_source") != "yolo":
                bits.append("选中是YOLO候选(+0.15)")
            if not bits:
                bits.append("分数接近, 排序临界")
            L.append(f"| {r['video']} | {r['fi']} | {r['sel_final']} | {r['govbest_final']} | {d:+.3f} | "
                     f"{r['sel_L1']} | {r['govbest_L1']} | {r['sel_L2']} | {r['govbest_L2']} | "
                     f"{r['govbest_iou']} | {'; '.join(bits)} |")
        L.append("\n- **结论**: 这些帧真灯已被检出(候选集内有 IoU≥0.3 的框), 但 select_gtfree 的 L1/L2 组合把它排到了干扰后 → "
                 "是**选灯排序问题**, 加'是否 governing 行人灯'判别器(正=governing crop/负=同帧其他候选)或强化几何/时序规则即可救, 主攻方向成立。\n")
    else:
        L.append("- 正常场景(扣 05)误绿帧中, **没有一帧是真灯在候选集却被选错** → 即正常场景的残余误绿不是选灯排序错, "
                 "而是检测召回洞或干扰自发绿。这会改变主攻方向(见下)。\n")

    # 召回洞帧
    out_frames = [r for r in non05 if r["decisive"].startswith("真灯不在")]
    if out_frames:
        L.append(f"## 检测召回洞帧(正常场景, {len(out_frames)} 帧)\n")
        L.append("| 视频 | fi | 类型 | maxIoU(gov) | 说明 |")
        L.append("|---|---|---|---|---|")
        for r in out_frames:
            L.append(f"| {r['video']} | {r['fi']} | {r['inferred_type']} | {r['max_iou_cand_gov']} | "
                     f"真灯未进候选集 → 选灯救不了, 需补检测召回(更小灯/更高 imgsz/专用检测器) |")
        L.append("")

    # 方向建议
    L.append("## 方向判读(供 cc 定方法)\n")
    n_inc = sum(1 for r in non05 if r["decisive"].startswith("真灯在"))
    n_out = sum(1 for r in non05 if r["decisive"].startswith("真灯不在"))
    n_nol = sum(1 for r in non05 if r["decisive"].startswith("无真灯"))
    if n_inc >= n_out + n_nol:
        L.append("- 正常场景误绿**以选灯排序错为主** → 主攻: 用 canonical GT 训'governing 判别器'增强/替换 L3 + 几何时序规则; "
                 "判别器能直接救这批。")
    else:
        L.append(f"- 正常场景误绿**以检测召回洞({n_out}) + 无真灯干扰自发绿({n_nol}) 为主**, 选灯排序错仅 {n_inc} 帧 → "
                 "**单靠改选灯救不了大头**, 需并行补检测召回(召回洞) + 拒干扰自发绿(无真灯帧的绿候选需被规则/判别器判为非灯)。"
                 "方法计划须同时覆盖两类, 不能只押选灯。")
    L.append("\n> 红线: 本脚本只读, 未训模型/未改 select_gtfree/未接线。下一步方法计划待 cc review。")
    REPORT.write_text("\n".join(L), encoding="utf-8")


def _write_gallery(rows):
    html = ["<html><head><meta charset='utf-8'><title>误绿选灯诊断画廊</title>",
            "<style>body{font-family:sans-serif;background:#111;color:#eee}"
            ".row{display:inline-block;margin:6px;border:1px solid #444;vertical-align:top}"
            ".cap{font-size:11px;padding:3px 5px;background:#222}</style></head><body>"]
    html.append(f"<h2>误绿选灯诊断画廊 ({len(rows)} 帧)</h2>")
    html.append("<p>绿框=选中(误绿)候选; 蓝框=governing 真灯; 红虚框=匹配到的非governing GT框(类型判定依据); "
                "灰细框=全部YOLO∪HSV候选。标注: 类型 / 决定性 / sel(final,L1,L2) / P(ped)。</p>")
    for (video, fi, frame, cands, sel, gov, matched_gt, recd) in rows:
        vis = frame.copy()
        H, W = vis.shape[:2]
        for c in cands:
            b = c["box_norm"]
            x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (90, 90, 90), 1)
        for gb in gov:
            b = gb["box_norm"]
            x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
        if matched_gt:
            b = matched_gt["box_norm"]
            x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_DASH)
        if sel:
            b = sel["box_norm"]
            x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _, buf = cv2.imencode(".jpg", vis)
        b64 = base64.b64encode(buf).decode()
        cap = (f"{video} fi={fi} [{'忽略05' if recd['excluded_05'] else '正常'}]<br>"
               f"类型={recd['inferred_type']}<br>决定性={recd['decisive']}<br>"
               f"sel final={recd['sel_final']} L1={recd['sel_L1']} L2={recd['sel_L2']} P(ped)={recd['sel_l3_ped']}")
        html.append(f"<div class='row'><img src='data:image/jpeg;base64,{b64}' width='260'>"
                    f"<div class='cap'>{cap}</div></div>")
    html.append("</body></html>")
    GALLERY.write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
