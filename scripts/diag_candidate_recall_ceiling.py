#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_candidate_recall_ceiling.py — 选灯精度 40.8% 天花板诊断(只读)

回答一个卡方向的问题: sel_prec=40.8% 低, 是
  (a) 排序错: 候选池里有 IoU≥0.3 的好框但没被 select_gtfree 选中 → 判别器/A消融有 headroom
  (b) 候选生成天花板: 候选池根本没好框 → 判别器救不了, 得修 prior 偏框

方法: 逐 governing 帧重建全候选池(YOLO+HSV), 算 max IoU vs gov_boxes,
      报 IoU≥0.3 召回上限(= sel_prec 的理论上限)。

候选构建口径与 eval_selection_quality.py:eval_video:99-104 逐行一致:
  YOLO conf=0.05, classes=[9], imgsz=1280 → yolo_px
  det._candidates(frame) → hsv_px
  build_candidates(yolo_px, hsv_px, W, H) → 去重合并
  归一化: (box[0]/W, box[1]/H, box[2]/W, box[3]/H)

过滤口径与 eval_video:117-124 一致:
  gov_boxes 非空才参与; gcolors 全 "unclear" → 跳过(UNKNOWN 排除)
  无候选帧计入分母, 天花板贡献 0

用法: PYTHONPATH=src ./.venv/bin/python scripts/diag_candidate_recall_ceiling.py
"""
import json, sys, csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from redlight.models import governing_disc as gd
from redlight.models.ped_light_selector import iou
from redlight.models.signal_candidates import build_candidates
from redlight.models.traffic_light import TrafficLightDetector

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
REPORT = ROOT / "docs" / "reports" / "2026-08-03-qw-candidate-recall-ceiling.md"
OUT_DIR = ROOT / "data" / "output" / "qw"

# 实测 sel_prec (5-seed mean, from 2026-07-31-wb-selection-quality.md)
SEL_PREC = {
    "违章01": 20.0, "违章02": 53.7, "违章03": 44.7, "违章04": 33.7,
    "违章05": 1.3, "违章06": 64.0, "违章07": 73.5, "违章08": 28.6,
    "违章09": 21.2, "违章10": 6.7, "违章11": 65.3,
}
SEL_PREC_OVERALL = 40.8


def diag_video(video, frames, det, yolo):
    """逐 governing 帧重建候选池, 算 ceil_iou。

    返回 list[dict], 每个 dict: {video, fi, n_cands, ceil_iou, n_gov_boxes}
    非 governing 帧(无 gov_boxes)和 UNKNOWN 帧(gcolors 全 unclear)被排除。
    无候选帧 ceil_iou=0.0, 计入分母。
    """
    fr_map = {int(f["source_fi"]): f for f in frames}
    cap_frames = gd._read_frames_at(video, [int(f["source_fi"]) for f in frames])
    fi_order = sorted(cap_frames.keys())

    results = []
    for fi in fi_order:
        frame = cap_frames[fi]
        g = fr_map[fi]

        # --- 过滤: 与 eval_video:118-121 同口径 ---
        gov_boxes = [tuple(b["box_norm"]) for b in g.get("boxes", []) if b.get("governing")]
        gcolors = {b["color"] for b in g.get("boxes", []) if b.get("governing")}
        if not gov_boxes:
            continue  # 非 governing 帧, 不参与
        if gcolors <= {"unclear"}:
            continue  # UNKNOWN 排除(canonical 同口径)

        # --- 重建候选: 与 eval_video:99-104 逐行一致 ---
        H, W = frame.shape[:2]
        res = yolo(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
        yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
        hsv_px = [s["box"] for s in det._candidates(frame)]
        cands_raw = build_candidates(yolo_px, hsv_px, W, H)
        cands = [
            {"box_norm": (c["box"][0] / W, c["box"][1] / H, c["box"][2] / W, c["box"][3] / H),
             "source": c["source"]}
            for c in cands_raw
        ]

        # --- 算 ceil_iou = max over cands ( max over gov_boxes iou(cand_norm, gb) ) ---
        best_iou = 0.0
        for c in cands:
            for gb in gov_boxes:
                v = iou(c["box_norm"], gb)
                if v > best_iou:
                    best_iou = v

        results.append({
            "video": video, "fi": fi, "n_cands": len(cands),
            "ceil_iou": best_iou, "n_gov_boxes": len(gov_boxes),
        })

    return results


def main():
    gt = json.load(open(GT, encoding="utf-8"))
    det = TrafficLightDetector(gd._cfg_tl(), verbose=False)
    yolo = gd._lazy_yolo()

    by_video = defaultdict(list)
    for fr in gt["frames"]:
        by_video[fr["video"]].append(fr)

    all_results = []
    per_video = {}
    for video in sorted(by_video):
        results = diag_video(video, by_video[video], det, yolo)
        all_results.extend(results)

        n_gov = len(results)
        n_has_cand = sum(1 for r in results if r["n_cands"] > 0)
        n_recall = sum(1 for r in results if r["ceil_iou"] >= 0.3)
        ceiling = n_recall / n_gov if n_gov > 0 else 0.0
        zero_cand_frac = (n_gov - n_has_cand) / n_gov if n_gov > 0 else 0.0
        mean_iou = sum(r["ceil_iou"] for r in results) / n_gov if n_gov > 0 else 0.0

        per_video[video] = {
            "n_gov": n_gov, "n_has_cand": n_has_cand, "n_recall_03": n_recall,
            "ceiling_03": ceiling, "zero_cand_frac": zero_cand_frac, "mean_ceil_iou": mean_iou,
        }
        print(f"[{video}] gov帧={n_gov} 有候选={n_has_cand} IoU≥0.3={n_recall} "
              f"天花板={ceiling*100:.1f}% 零候选={zero_cand_frac*100:.1f}% 平均iou={mean_iou:.3f}",
              flush=True)

    # 全局
    n_gov_all = len(all_results)
    n_recall_all = sum(1 for r in all_results if r["ceil_iou"] >= 0.3)
    n_zero_all = sum(1 for r in all_results if r["n_cands"] == 0)
    ceiling_all = n_recall_all / n_gov_all if n_gov_all > 0 else 0.0
    zero_frac_all = n_zero_all / n_gov_all if n_gov_all > 0 else 0.0
    mean_iou_all = sum(r["ceil_iou"] for r in all_results) / n_gov_all if n_gov_all > 0 else 0.0

    print(f"\n=== 全局 ===")
    print(f"governing帧={n_gov_all} IoU≥0.3召回={n_recall_all} 天花板={ceiling_all*100:.1f}%")
    print(f"零候选帧={n_zero_all} ({zero_frac_all*100:.1f}%) 平均ceil_iou={mean_iou_all:.3f}")
    print(f"实测sel_prec={SEL_PREC_OVERALL}% headroom={ceiling_all*100 - SEL_PREC_OVERALL:+.1f}pp")

    # 裁断 (ceil≥70% → 排序问题; ceil≤46% → 候选生成天花板; 中间 → 混合)
    if ceiling_all >= 0.70:
        verdict = (f"天花板 {ceiling_all*100:.1f}% ≫ 实测 {SEL_PREC_OVERALL}% → "
                   f"排序问题: 候选池里有好框(IoU≥0.3)但没被选中, "
                   f"判别器/A消融有 headroom ({ceiling_all*100 - SEL_PREC_OVERALL:+.1f}pp)")
    elif ceiling_all <= SEL_PREC_OVERALL / 100 + 0.05:
        verdict = (f"天花板 {ceiling_all*100:.1f}% ≈ 实测 {SEL_PREC_OVERALL}% → "
                   f"候选生成天花板: 候选池根本没好框, 判别器救不了, 得修 prior 偏框")
    else:
        verdict = (f"天花板 {ceiling_all*100:.1f}% > 实测 {SEL_PREC_OVERALL}% 但差距有限 "
                  f"(headroom {ceiling_all*100 - SEL_PREC_OVERALL:+.1f}pp); "
                  f"排序有一定 headroom, 候选生成也是瓶颈")
    print(f"裁断: {verdict}")

    # 写逐帧 CSV (供 cc bit-for-bit 复核)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "candidate_recall_ceiling_per_frame.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "fi", "n_cands", "ceil_iou", "n_gov_boxes", "recall_03"])
        for r in all_results:
            w.writerow([r["video"], r["fi"], r["n_cands"],
                        f"{r['ceil_iou']:.4f}", r["n_gov_boxes"],
                        1 if r["ceil_iou"] >= 0.3 else 0])
    print(f"[csv] {csv_path}")

    _write_report(per_video, all_results, n_gov_all, n_recall_all, ceiling_all,
                  n_zero_all, zero_frac_all, mean_iou_all, verdict)


def _write_report(per_video, all_results, n_gov, n_recall, ceiling,
                  n_zero, zero_frac, mean_iou, verdict):
    L = [
        "# 选灯精度天花板诊断(qw, 只读)\n",
        "> 回答 sel_prec=40.8% 低是排序错(候选池有好框没选中→判别器有救)还是候选生成天花板(候选池没好框→得修 prior 偏框)。\n",
        "> 候选构建口径与 `eval_selection_quality.py:eval_video:99-104` 逐行一致: "
        "YOLO conf=0.05/classes=[9]/imgsz=1280 + `det._candidates` + `build_candidates` + 归一化。\n",
        "> 过滤口径与 `eval_video:117-124` 一致: gov_boxes 非空; gcolors 全 unclear 跳过; 无候选帧计入分母贡献 0。\n",
        "> per-video sel_prec 基准来自 `docs/reports/2026-07-31-wb-selection-quality.md`(5-seed mean)。\n\n",
        "## 裁断\n",
        f"**{verdict}**\n",
        f"\n## 全局\n",
        f"- governing 帧数(过滤后): {n_gov}",
        f"- IoU≥0.3 召回帧数: {n_recall}",
        f"- **候选召回天花板 @ IoU≥0.3: {ceiling*100:.1f}%**",
        f"- 实测 sel_prec(5-seed mean): {SEL_PREC_OVERALL}%",
        f"- headroom: {(ceiling*100 - SEL_PREC_OVERALL):+.1f}pp",
        f"- 零候选 governing 帧: {n_zero} ({zero_frac*100:.1f}%)",
        f"- 平均 ceil_iou: {mean_iou:.3f}",
        f"\n## 逐视频分解\n",
        "| 视频 | gov帧 | 有候选 | IoU≥0.3 | 天花板 | sel_prec(实测) | headroom | 零候选 | 平均ceil_iou |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for v in sorted(per_video):
        pv = per_video[v]
        sp = SEL_PREC.get(v, "N/A")
        sp_str = f"{sp}%" if isinstance(sp, (int, float)) else sp
        headroom = f"{pv['ceiling_03']*100 - sp:+.1f}pp" if isinstance(sp, (int, float)) else "N/A"
        L.append(
            f"| {v} | {pv['n_gov']} | {pv['n_has_cand']} | {pv['n_recall_03']} | "
            f"{pv['ceiling_03']*100:.1f}% | {sp_str} | {headroom} | "
            f"{pv['zero_cand_frac']*100:.1f}% | {pv['mean_ceil_iou']:.3f} |"
        )

    # 重点视频
    L.append(f"\n## 重点视频分析(05/01/09/10 — sel_prec 最低的四个)\n")
    for v in ["违章05", "违章01", "违章09", "违章10"]:
        pv = per_video[v]
        sp = SEL_PREC.get(v, 0)
        L.append(f"\n### {v}")
        L.append(f"- sel_prec(实测): {sp}%")
        L.append(f"- 天花板: {pv['ceiling_03']*100:.1f}%")
        if pv['n_gov'] > 0:
            if pv['ceiling_03'] * 100 < 30:
                L.append(f"- 判定: **候选生成天花板** — 天花板本身 <30%, 候选池缺好框, 需修 prior 偏框或提 YOLO 分辨率")
            elif pv['ceiling_03'] * 100 <= sp + 5:
                L.append(f"- 判定: **候选生成天花板** — 候选池缺好框, 需修 prior 偏框或提 YOLO 分辨率")
            else:
                L.append(f"- 判定: **排序问题** — 候选池有好框但没选中, 判别器/A消融有 headroom")
            L.append(f"- 零候选帧: {pv['n_gov'] - pv['n_has_cand']}/{pv['n_gov']} ({pv['zero_cand_frac']*100:.1f}%)")
            L.append(f"- 平均 ceil_iou: {pv['mean_ceil_iou']:.3f}")

    L.append(f"\n## 方法学\n")
    L.append("- 复用 `gd._read_frames_at`/`gd._lazy_yolo`/`gd._cfg_tl`/`build_candidates`/`iou`, "
             "不引入新口径。\n")
    L.append("- 只读诊断, 不训练、不改生产代码、不碰 s1 缓存。\n")
    L.append(f"- 逐帧明细 CSV: `data/output/qw/candidate_recall_ceiling_per_frame.csv` (供 cc bit-for-bit 复核)。\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {REPORT}")


if __name__ == "__main__":
    main()
