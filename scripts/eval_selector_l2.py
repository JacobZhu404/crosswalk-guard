#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_selector_l2.py — 阶段1 门控(命门) L1+L2 leave-some-out 评测。

在密集窗口候选数据(candidates_temporal.json)上:
  同视频内 compute_temporal_scores (L2, 零标注, 跨帧复现固定设施) →
  跨视频 leave-some-out 派生 ped 几何先验 (L1) →
  每帧 select_gtfree(L1 + L2) 选灯, 度量选灯准确率。

同时报 L1-only(同密集数据, temporal_scores=None) 作公平消融, 隔离 L2 贡献。
gate 只看 leave-some-out (护栏1); 检测缺口视频(03/05 类 YOLO 漏检)不计入选灯准确率(归 M4/M3)。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/eval_selector_l2.py [--cands data/output/candidates_temporal.json]
"""
import json, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from redlight.models.ped_light_selector import (
    derive_ped_priors, select_gtfree, compute_temporal_scores, _center_dist,
)

CENTER_HIT = 0.06


def _group_by_video(records):
    g = {}
    for r in records:
        g.setdefault(r["video"], []).append(r)
    return g


def _run(records, use_l2, center_hit=CENTER_HIT):
    vids = sorted({r["video"] for r in records})
    g = _group_by_video(records)
    det = sel = 0
    per = {}
    for V in vids:
        train_wh = [r["gt_wh"] for r in records if r["video"] != V]
        prior = derive_ped_priors(train_wh)
        frames = g[V]
        tscore = compute_temporal_scores(frames) if use_l2 else [None] * len(frames)
        dv = sv = 0
        for k, r in enumerate(frames):
            cands, gt = r["candidates"], r["gt_box_norm"]
            ts = tscore[k] if use_l2 else None
            sel_box = select_gtfree(cands, prior, temporal_scores=ts)
            has_det = any(_center_dist(c.get("box_norm"), gt) < center_hit
                          for c in cands if c.get("box_norm"))
            if has_det:
                dv += 1
                if sel_box is not None and _center_dist(sel_box.get("box_norm"), gt) < center_hit:
                    sv += 1
        per[V] = {"detected_frames": dv, "selection_correct": sv,
                  "selection_acc": (sv / dv) if dv else None}
        det += dv
        sel += sv
    n = len(records)
    return {"n_frames": n,
            "detection_coverage": (det / n) if n else 0.0,
            "selection_accuracy": (sel / det) if det else 0.0,
            "per_video": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=str(ROOT / "data" / "output" / "candidates_temporal.json"))
    args = ap.parse_args()
    with open(args.cands, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"[eval L2] {len(records)} 密集窗口帧, imgsz={data.get('imgsz')} "
          f"window=±{data.get('window')}/step{data.get('step')}")

    l1 = _run(records, use_l2=False)
    l2 = _run(records, use_l2=True)

    print("\n=== L1-only (同密集数据, 无时序) ===")
    print(f"  检测覆盖 {l1['detection_coverage']*100:.1f}%  选灯准确率 {l1['selection_accuracy']*100:.1f}%")
    print("=== L1+L2 (时序复现, 零标注) ===")
    print(f"  检测覆盖 {l2['detection_coverage']*100:.1f}%  选灯准确率 {l2['selection_accuracy']*100:.1f}%")

    print("\n=== 逐视频 L1+L2 ===")
    print(f"{'video':7} {'det':>4} {'selOK':>5} {'acc':>5}")
    for V, d in sorted(l2["per_video"].items()):
        acc = d["selection_acc"]
        print(f"{V:7} {d['detected_frames']:>4} {d['selection_correct']:>5} "
              f"{('%.2f' % acc) if acc is not None else '  -':>5}")

    delta = l2["selection_accuracy"] - l1["selection_accuracy"]
    print(f"\n[L2 贡献] 选灯准确率 +{delta*100:.1f}pp (leave-some-out)")
    print("[校准] LOVO 是新视频泛化代理(非生产保证); L2 用同视频跨帧复现(固定设施)奖励 ped, 零标注。")


if __name__ == "__main__":
    main()
