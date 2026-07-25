#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_selector_lso.py — 阶段1 门控: PedLightSelector(M2b) leave-some-out 评测。

leave-one-video-out: 每视频 V 用其余视频 GT 派生 ped 几何先验, 在 V 帧上跑 M2b 选灯,
度量(1) 检测覆盖: V 帧是否有候选中心距<0.06 于 GT(ped 被检测到);
     (2) 选灯准确率: ped 被检测到的帧里, M2b 是否选了 ped 候选(不读 GT, 护栏1)。
同时报 GT-in(同视频先验) 作对照哨兵 —— **gate 只看 leave-some-out, GT-in 仅诊断**。

这是"新视频泛化"的代理(cc 校准#2), 非生产保证。05 类漏检属检测缺口(归 M4/M3), 不计入选灯准确率。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/eval_selector_lso.py [--cands data/output/candidates_28.json]
"""
import json, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from redlight.models.ped_light_selector import (
    derive_ped_priors, select_gtfree, leave_some_out_eval, _center_dist,
)

CENTER_HIT = 0.06


def _gt_in_eval(records, center_hit=CENTER_HIT):
    """对照: 同视频先验(GT-in), 仅诊断哨兵, 不用于 gate。"""
    videos = sorted({r["video"] for r in records})
    det = sel = 0
    for V in videos:
        prior = derive_ped_priors([r["gt_wh"] for r in records if r["video"] == V])
        for r in records:
            if r["video"] != V:
                continue
            cands, gt = r["candidates"], r["gt_box_norm"]
            if any(_center_dist(c.get("box_norm"), gt) < center_hit for c in cands if c.get("box_norm")):
                det += 1
                if _center_dist(select_gtfree(cands, prior).get("box_norm"), gt) < center_hit:
                    sel += 1
    n = len(records)
    return {"detection_coverage": det / n, "selection_accuracy": sel / det if det else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=str(ROOT / "data" / "output" / "candidates_28.json"))
    args = ap.parse_args()
    with open(args.cands, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"[eval] {len(records)} 标注帧, imgsz={data.get('imgsz')}")

    lso = leave_some_out_eval(records, center_hit=CENTER_HIT)
    gtin = _gt_in_eval(records, center_hit=CENTER_HIT)

    print("\n=== 阶段1 门控: M2b leave-some-out (生产态, 无 GT 泄露) ===")
    print(f"  检测覆盖 (ped 被检测到): {lso['detection_coverage']*100:.1f}%")
    print(f"  选灯准确率 (检测到帧里 M2b 选对 ped): {lso['selection_accuracy']*100:.1f}%")
    print(f"  mean selected IoU vs GT: {lso['mean_selected_iou']:.3f}")
    print("\n=== 逐视频 (LOVO) ===")
    print(f"{'video':7} {'det':>4} {'selOK':>5} {'acc':>5}")
    for V, d in sorted(lso["per_video"].items()):
        acc = d["selection_acc"]
        print(f"{V:7} {d['detected_frames']:>4} {d['selection_correct']:>5} "
              f"{('%.2f' % acc) if acc is not None else '  -':>5}")

    print("\n=== 对照哨兵: GT-in (同视频先验, 仅诊断, 不用于 gate) ===")
    print(f"  检测覆盖: {gtin['detection_coverage']*100:.1f}%  选灯准确率: {gtin['selection_accuracy']*100:.1f}%")

    print("\n[判定] gate 看 leave-some-out 选灯准确率; 05 类漏检算检测缺口(归 M4/M3), 不计入。")
    print("[校准] LOVO 是'新视频泛化'代理, 非生产保证(真生产=新路口/手机)。")


if __name__ == "__main__":
    main()
