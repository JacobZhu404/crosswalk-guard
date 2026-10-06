#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_selector_failures.py — 阶段1 失败模式诊断: M2b 在检测到 ped 的帧里为何选错?

对每帧(LOVO): 标出 ped 候选(中心距<0.06 于 GT), 打印 M2b 选了谁、L1 分多少、ped 候选 L1 分多少、
top-3 候选的 aspect。判断 L1 是"差一点(可修)"还是"根本不够(需 L3)"。
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from redlight.models.ped_light_selector import (
    derive_ped_priors, select_gtfree, _l1_geom_score, _center_dist, _box_wh,
)

CENTER_HIT = 0.06


def _aspect(box):
    w, h = _box_wh(box)
    return h / w if w > 0 else 99.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=str(ROOT / "data" / "output" / "candidates_28.json"))
    args = ap.parse_args()
    data = json.load(open(args.cands, encoding="utf-8"))
    records = data["records"]
    videos = sorted({r["video"] for r in records})

    for V in videos:
        prior = derive_ped_priors([r["gt_wh"] for r in records if r["video"] != V])
        for r in records:
            if r["video"] != V:
                continue
            cands, gt = r["candidates"], r["gt_box_norm"]
            # ped 候选
            ped = [c for c in cands if _center_dist(c["box_norm"], gt) < CENTER_HIT]
            if not ped:
                continue  # 未检测到 ped, 跳过(检测缺口归 M4/M3)
            ped_box = ped[0]["box_norm"]
            sel = select_gtfree(cands, prior)
            sel_box = sel["box_norm"] if sel else None
            picked_ped = sel_box is not None and _center_dist(sel_box, gt) < CENTER_HIT
            # top-3 by L1
            scored = sorted(cands, key=lambda c: -_l1_geom_score(c["box_norm"], prior))[:3]
            print(f"\n{V} fi={r['fi']}  {'✅选对' if picked_ped else '❌选错'}")
            print(f"  ped 框 aspect={_aspect(ped_box):.2f} L1={_l1_geom_score(ped_box, prior):.2f} "
                  f"src={ped[0]['source']}")
            if sel_box is not None:
                print(f"  M2b选 aspect={_aspect(sel_box):.2f} L1={_l1_geom_score(sel_box, prior):.2f} "
                      f"src={sel['source']} ctrD={_center_dist(sel_box, gt):.3f}")
            print(f"  prior aspect范围=[{prior['aspect_min']:.2f},{prior['aspect_max']:.2f}] "
                  f"area范围=[{prior['area_min']:.4f},{prior['area_max']:.4f}]")
            print(f"  top3候选 aspect/L1/src: " +
                  ", ".join(f"{_aspect(c['box_norm']):.2f}/{_l1_geom_score(c['box_norm'], prior):.2f}/{c['source']}"
                            for c in scored))


if __name__ == "__main__":
    main()
