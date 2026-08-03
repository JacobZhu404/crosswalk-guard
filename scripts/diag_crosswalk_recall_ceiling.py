#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_crosswalk_recall_ceiling.py — 斑马线检测召回天花板(只读)

量 v11(全宽横带)和 v2(透视梯形)相对 GT poly 的差距, 拆成召回 vs 几何过宽。
答三问: ①IoU 天花板 ②召回不足 vs 形状过宽 ③v11 vs v2 头对头。

口径:
  v11 = CrosswalkDetector(cfg).detect(frame), 不传 vehicle_boxes(本征掩膜质量)
  v2  = CrosswalkDetectorV2(cfg).detect(frame), 每帧独立实例(与 eval_crosswalk_mask.py:52 一致)
  GT  = datasets/gt/crosswalk/违章*.json, poly 像素坐标, poly_to_mask 栅格化
  recall    = |pred ∩ GT| / |GT|
  precision = |pred ∩ GT| / |pred|

用法: PYTHONPATH=src ./.venv/bin/python scripts/diag_crosswalk_recall_ceiling.py
"""
import os, sys, csv, json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from redlight.infrastructure.config import load_config
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.crosswalk_v2 import CrosswalkDetectorV2
from redlight.evaluation.module_metrics import poly_to_mask, mask_iou, mask_band

GT_DIR = ROOT / "datasets" / "gt" / "crosswalk"
REPORT = ROOT / "docs" / "reports" / "2026-08-03-qw-crosswalk-recall-ceiling.md"
OUT_DIR = ROOT / "data" / "output" / "qw"
CFG = ROOT / "configs" / "config.yaml"


def _mask_stats(pred_mask, gt_mask):
    """算 IoU / recall / precision / 面积。

    recall = |pred ∩ GT| / |GT|  (v11 带子覆盖了多少真斑马线)
    precision = |pred ∩ GT| / |pred|  (v11 带子里多少是真斑马线)
    """
    p = pred_mask > 0
    g = gt_mask > 0
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    gt_area = int(g.sum())
    pred_area = int(p.sum())
    iou = inter / union if union > 0 else 0.0
    recall = inter / gt_area if gt_area > 0 else 0.0
    precision = inter / pred_area if pred_area > 0 else 0.0
    return {
        "iou": iou, "recall": recall, "precision": precision,
        "gt_area": gt_area, "pred_area": pred_area,
    }


def eval_frame(video, ts, poly, cfg, cap):
    """单帧 v11+v2 检测 + GT 对比。返回 dict 或 None(取帧失败/poly 空)。"""
    if not poly or len(poly) < 3:
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, int(ts * 1000))
    ret, frame = cap.read()
    if not ret:
        return None
    H, W = frame.shape[:2]
    gt_mask = poly_to_mask(poly, H, W)

    # v11 (每帧新实例, 无状态)
    det11 = CrosswalkDetector(cfg)
    mask11 = det11.detect(frame)  # 不传 vehicle_boxes
    s11 = _mask_stats(mask11, gt_mask)
    s11["band"] = mask_band(mask11)

    # v2 (每帧新实例, 与 eval_crosswalk_mask.py:52 一致, 量单帧掩膜质量)
    det2 = CrosswalkDetectorV2(cfg)
    mask2 = det2.detect(frame)
    s2 = _mask_stats(mask2, gt_mask)
    s2["band"] = mask_band(mask2)

    return {
        "ts": ts, "H": H, "W": W,
        "v11_iou": s11["iou"], "v11_recall": s11["recall"], "v11_precision": s11["precision"],
        "v11_area": s11["pred_area"], "v11_band": s11["band"],
        "v2_iou": s2["iou"], "v2_recall": s2["recall"], "v2_precision": s2["precision"],
        "v2_area": s2["pred_area"], "v2_band": s2["band"],
        "gt_area": s11["gt_area"],
    }


def main():
    cfg = load_config(str(CFG))
    gt_files = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".json"))
    print(f"GT 文件: {[f[:-5] for f in gt_files]}")
    print(f"v11 = 全宽横带 (crosswalk.py:162-167)")
    print(f"v2  = 透视梯形 (crosswalk_v2.py:82-110)")
    print()

    all_rows = []
    per_video = {}
    for gf in gt_files:
        video = gf[:-5]
        gt = json.load(open(GT_DIR / gf, encoding="utf-8"))
        frames = gt.get("frames", [])
        video_path = str(ROOT / "input_video" / f"{video}.mp4")
        if not os.path.isfile(video_path):
            print(f"[跳过] {video}: 视频文件不存在")
            continue
        cap = cv2.VideoCapture(video_path)

        v_rows = []
        for fr in frames:
            ts = fr["ts"]
            poly = fr.get("poly")
            note = fr.get("note", "")
            if not poly or len(poly) < 3:
                print(f"  [{video}@{ts}s] poly 空, 跳过")
                continue
            r = eval_frame(video, ts, poly, cfg, cap)
            if r is None:
                print(f"  [{video}@{ts}s] 取帧失败")
                continue
            r["video"] = video
            r["note"] = note
            v_rows.append(r)
            all_rows.append(r)
            print(f"  [{video}@{ts:.1f}s] v11: IoU={r['v11_iou']:.3f} rec={r['v11_recall']:.3f} "
                  f"prec={r['v11_precision']:.3f} area={r['v11_area']} band={r['v11_band']} | "
                  f"v2: IoU={r['v2_iou']:.3f} rec={r['v2_recall']:.3f} "
                  f"prec={r['v2_precision']:.3f} area={r['v2_area']}",
                  flush=True)
        cap.release()

        if v_rows:
            per_video[video] = {
                "n": len(v_rows),
                "v11_iou": np.mean([r["v11_iou"] for r in v_rows]),
                "v11_recall": np.mean([r["v11_recall"] for r in v_rows]),
                "v11_precision": np.mean([r["v11_precision"] for r in v_rows]),
                "v2_iou": np.mean([r["v2_iou"] for r in v_rows]),
                "v2_recall": np.mean([r["v2_recall"] for r in v_rows]),
                "v2_precision": np.mean([r["v2_precision"] for r in v_rows]),
            }
            v = per_video[video]
            print(f"  [{video}] 均值 v11_IoU={v['v11_iou']:.3f} rec={v['v11_recall']:.3f} "
                  f"prec={v['v11_precision']:.3f} | v2_IoU={v['v2_iou']:.3f}\n")

    # 聚合
    n = len(all_rows)
    if n == 0:
        print("无有效帧")
        return

    v11_ious = [r["v11_iou"] for r in all_rows]
    v11_recs = [r["v11_recall"] for r in all_rows]
    v11_pres = [r["v11_precision"] for r in all_rows]
    v2_ious = [r["v2_iou"] for r in all_rows]

    print(f"\n=== 聚合 ({n} frames, {len(per_video)} videos) ===")
    print(f"v11: IoU 均值={np.mean(v11_ious):.3f} recall={np.mean(v11_recs):.3f} precision={np.mean(v11_pres):.3f}")
    print(f"v2:  IoU 均值={np.mean(v2_ious):.3f}")
    low_iou = [r for r in all_rows if r["v11_iou"] < 0.5]
    print(f"v11 IoU<0.5 的帧: {len(low_iou)}/{n}")
    # 分类: recall<0.5=召回不足, precision<0.5=几何过宽
    rec_low = [r for r in all_rows if r["v11_recall"] < 0.5]
    prec_low = [r for r in all_rows if r["v11_precision"] < 0.5]
    both_low = [r for r in all_rows if r["v11_recall"] < 0.5 and r["v11_precision"] < 0.5]
    print(f"v11 recall<0.5: {len(rec_low)}/{n} (真没检到)")
    print(f"v11 precision<0.5: {len(prec_low)}/{n} (几何过宽/全宽带溢出)")
    print(f"v11 both<0.5: {len(both_low)}/{n}")

    # 写 CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "crosswalk_recall_ceiling_per_frame.csv"
    fields = ["video", "ts", "note", "v11_iou", "v11_recall", "v11_precision",
              "v2_iou", "v2_recall", "v2_precision", "gt_area", "v11_area", "v2_area",
              "v11_band", "v2_band", "H", "W"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in fields})
    print(f"[csv] {csv_path}")

    _write_report(all_rows, per_video, n, v11_ious, v11_recs, v11_pres, v2_ious)


def _write_report(all_rows, per_video, n, v11_ious, v11_recs, v11_pres, v2_ious):
    L = [
        "# 斑马线检测召回天花板诊断(qw, 只读)\n",
        "> 量 v11(全宽横带)和 v2(透视梯形)相对 GT poly 的差距, 拆成召回 vs 几何过宽。\n",
        "> v11 = `CrosswalkDetector.detect(frame)` 不传 vehicle_boxes(本征掩膜质量, 与 eval_crosswalk_mask.py:52 一致)。\n",
        "> v2 = `CrosswalkDetectorV2.detect(frame)`, 每帧独立实例(量单帧质量, 不走时序聚合)。\n",
        "> GT = `datasets/gt/crosswalk/`, poly 像素坐标, `poly_to_mask` 栅格化。\n",
        f"> recall = |pred∩GT|/|GT|, precision = |pred∩GT|/|pred|, 单次跑无 seed。\n\n",
        "## 三问裁断\n",
    ]

    # Q1
    low_videos = [v for v, d in per_video.items() if d["v11_iou"] < 0.5]
    L.append(f"**Q1 IoU 天花板**: v11 均值 IoU={np.mean(v11_ious):.3f}, "
              f"范围 [{min(v11_ious):.3f}, {max(v11_ious):.3f}]; "
              f"IoU<0.5 的视频: {low_videos if low_videos else '无'}\n")

    # Q2
    rec_low = sum(1 for r in all_rows if r["v11_recall"] < 0.5)
    prec_low = sum(1 for r in all_rows if r["v11_precision"] < 0.5)
    both_low = sum(1 for r in all_rows if r["v11_recall"] < 0.5 and r["v11_precision"] < 0.5)
    if prec_low > rec_low:
        q2 = (f"**Q2 召回 vs 几何过宽**: v11 recall 均值={np.mean(v11_recs):.3f}, "
              f"precision 均值={np.mean(v11_pres):.3f} → "
              f"**几何过宽为主**({prec_low}/{n} 帧 precision<0.5 vs {rec_low}/{n} recall<0.5)。"
              f"全宽横带纵向盖住条纹但横向溢出全宽 → 瓶颈是形状, 修匹配/上梯形而非提检测灵敏度。")
    elif rec_low > prec_low:
        q2 = (f"**Q2 召回 vs 几何过宽**: v11 recall 均值={np.mean(v11_recs):.3f}, "
              f"precision 均值={np.mean(v11_pres):.3f} → "
              f"**召回不足为主**({rec_low}/{n} 帧 recall<0.5)。"
              f"斑马线条纹没检到 → 需提检测灵敏度/降阈值, 非形状问题。")
    else:
        q2 = (f"**Q2 召回 vs 几何过宽**: v11 recall={np.mean(v11_recs):.3f}, "
              f"precision={np.mean(v11_pres):.3f}, 两者都低 → 混合问题。")
    L.append(q2 + "\n")

    # Q3
    v2_better = sum(1 for r in all_rows if r["v2_iou"] > r["v11_iou"])
    v2_worse = sum(1 for r in all_rows if r["v2_iou"] < r["v11_iou"])
    if v2_better > v2_worse:
        q3 = (f"**Q3 v11 vs v2**: v2 IoU 均值={np.mean(v2_ious):.3f} vs v11={np.mean(v11_ious):.3f}, "
              f"v2 优 {v2_better}/{n} 帧 → 梯形方向可行, 值得立项完善。")
    else:
        q3 = (f"**Q3 v11 vs v2**: v2 IoU 均值={np.mean(v2_ious):.3f} vs v11={np.mean(v11_ious):.3f}, "
              f"v2 仅优 {v2_better}/{n} 帧 → 单帧质量 v2 不优于 v11, 梯形需大改或需时序聚合才有效。")
    L.append(q3 + "\n")

    # denom=mask 稀释效应
    L.append(f"\n## denom=mask 稀释效应(分析讨论)\n")
    L.append(f"v11 全宽横带 → mask_area 巨大 → `compute_overlap_ratio(denom='mask')` 的 "
             f"`inside/mask_area` 被稀释。历史实验 denom=box(F1=0.941) > denom=mask(F1=0.875) "
             f"与此一致: 全宽带不只压 IoU, 还压 denom=mask 下的占道比。"
             f"修形状(约束带宽/上梯形)可同时改善 IoU 和 denom=mask 占比。\n")

    # 聚合
    L.append(f"\n## 聚合 ({n} frames)\n")
    L.append(f"- v11: IoU={np.mean(v11_ious):.3f}, recall={np.mean(v11_recs):.3f}, precision={np.mean(v11_pres):.3f}")
    L.append(f"- v2: IoU={np.mean(v2_ious):.3f}")
    L.append(f"- v11 IoU<0.5: {sum(1 for r in all_rows if r['v11_iou'] < 0.5)}/{n} 帧")
    L.append(f"- v11 recall<0.5: {rec_low}/{n}, precision<0.5: {prec_low}/{n}, both<0.5: {both_low}/{n}")

    # 逐视频
    L.append(f"\n## 逐视频分解\n")
    L.append("| 视频 | n | v11 IoU | v11 recall | v11 precision | v2 IoU | 判定 |")
    L.append("|---|---|---|---|---|---|---|")
    for v in sorted(per_video):
        d = per_video[v]
        if d["v11_recall"] >= 0.5 and d["v11_precision"] < 0.5:
            verdict = "几何过宽(recall 高 precision 低)"
        elif d["v11_recall"] < 0.5 and d["v11_precision"] >= 0.5:
            verdict = "召回不足(recall 低 precision 高)"
        elif d["v11_recall"] < 0.5 and d["v11_precision"] < 0.5:
            verdict = "混合(都低)"
        else:
            verdict = "良好(都高)"
        L.append(f"| {v} | {d['n']} | {d['v11_iou']:.3f} | {d['v11_recall']:.3f} | "
                 f"{d['v11_precision']:.3f} | {d['v2_iou']:.3f} | {verdict} |")

    # 逐帧
    L.append(f"\n## 逐 anchor 帧明细\n")
    L.append("| video | ts | note | v11 IoU | v11 rec | v11 prec | v11 area | v11 band | v2 IoU | v2 rec | v2 prec | v2 area | gt area |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in all_rows:
        L.append(f"| {r['video']} | {r['ts']:.1f} | {r.get('note','')} | "
                 f"{r['v11_iou']:.3f} | {r['v11_recall']:.3f} | {r['v11_precision']:.3f} | "
                 f"{r['v11_area']} | {r['v11_band']} | "
                 f"{r['v2_iou']:.3f} | {r['v2_recall']:.3f} | {r['v2_precision']:.3f} | "
                 f"{r['v2_area']} | {r['gt_area']} |")

    L.append(f"\n## 方法学\n")
    L.append("- 复用 `eval_crosswalk_mask.py` 的骨架, 加 recall/precision 分解。\n")
    L.append("- v11/v2 每帧独立实例(量单帧掩膜质量, v2 不走时序聚合)。\n")
    L.append("- `poly_to_mask(poly, H, W)` 用实际 `frame.shape`, 不硬编码分辨率。\n")
    L.append("- v11 不传 vehicle_boxes(纯检测天花板, 与 eval_crosswalk_mask.py:52 一致)。\n")
    L.append("- 经典 CV 确定性, 单次跑无 seed。\n")
    L.append(f"- 逐帧 CSV: `data/output/qw/crosswalk_recall_ceiling_per_frame.csv`\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {REPORT}")


if __name__ == "__main__":
    main()
