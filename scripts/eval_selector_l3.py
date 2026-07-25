#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_selector_l3.py — 阶段1 门控(命门) L1+L2+L3 leave-some-out 评测。

在密集窗口候选(candidates_temporal.json)上, 镜像 eval_selector_l2.py, 但 L3 学习式判别头
按 **leave-one-video-out** 训练(每视频 V 用其余 10 视频的 Jacob 标样本 crop 训, 绝不在 V 自身上训,
否则 gate 数字造假, 护栏1 同义延伸)。对 V 帧候选实时裁图打分, select_gtfree(L1+L2+l3) 选灯。

报告:
  - L1+L2 leave-some-out 选灯准确率(基线, 同 eval_selector_l2)
  - L1+L2+L3 leave-some-out 选灯准确率, 并对 l3_weight ∈ {0.2,0.3,0.5,0.7,1.0} 做扫参取最优
  - 逐视频对照, 标出 L3 翻盘的帧

注意: 这是新视频泛化代理(cc 校准#2), 非生产保证。生产接线(dag.py)仍待 cc 放行, 本脚本只测门控。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/eval_selector_l3.py [--smoke] [--cands ...] [--manifest ...] [--labels ...]
"""
import json, sys, os, argparse, random, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from PIL import Image

from redlight.models.ped_light_selector import (
    derive_ped_priors, select_gtfree, compute_temporal_scores, _center_dist,
)
from redlight.models.l3_ped_vehicle import (
    build_labeled_crops, group_by_video, train_model, crop_candidate, score_crop,
    CLASSES, _IDX, _eval_acc, _CropDataset,
)

CENTER_HIT = 0.06
DEVICE = "cpu"
L3_WEIGHTS = [0.2, 0.3, 0.5, 0.7, 1.0]
EPOCHS = 40


def _frame_path(video, fi):
    return str(ROOT / "datasets" / "frames" / video / f"frame_{fi:06d}.jpg")


def _stratified_split(pool, val_frac=0.2, seed=0):
    random.seed(seed)
    by_lab = defaultdict(list)
    for c in pool:
        by_lab[c["label"]].append(c)
    train, val = [], []
    for lab, items in by_lab.items():
        random.shuffle(items)
        nval = max(1, int(len(items) * val_frac))
        val += items[:nval]
        train += items[nval:]
    to_items = lambda lst: [((c["video"], c["fi"], tuple(c["box_norm"])), _IDX[c["label"]])
                            for c in lst]
    return to_items(train), to_items(val)


def _build_l3_scores(model, frame_img, candidates):
    scores = {}
    for i, c in enumerate(candidates):
        bn = c.get("box_norm")
        if not bn:
            continue
        t = crop_candidate(frame_img, tuple(bn))
        p = score_crop(model, t)
        scores[i] = p["ped"]
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=str(ROOT / "data" / "output" / "candidates_temporal.json"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "output" / "l3_labels" / "manifest.json"))
    ap.add_argument("--labels", default=str(ROOT / "data" / "output" / "l3_labels" / "labels.json"))
    ap.add_argument("--smoke", action="store_true", help="只跑 1 个视频, 验证流水线+计时")
    args = ap.parse_args()

    with open(args.cands, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    raw_labels = json.load(open(args.labels, encoding="utf-8"))
    labels = {x["id"]: x["label"] for x in raw_labels}

    # 标注裁图, 按视频分组(用于 LOVO L3 训练)
    all_crops = build_labeled_crops(manifest, labels)
    crops_by_video = group_by_video(all_crops)
    print(f"[L3 数据] 标注裁图 {len(all_crops)} 簇, 覆盖视频 {sorted(crops_by_video.keys())}")
    from collections import Counter
    lab_cnt = Counter(c["label"] for c in all_crops)
    print(f"        类别分布 ped={lab_cnt['ped']} vehicle={lab_cnt['vehicle']} other={lab_cnt['other']}")

    videos = sorted({r["video"] for r in records})
    group = defaultdict(list)
    for r in records:
        group[r["video"]].append(r)
    if args.smoke:
        videos = videos[:1]

    per_video = {}
    det_total = 0
    sel_l2_total = 0
    sel_l3_total = {w: 0 for w in L3_WEIGHTS}

    t0 = time.time()
    for vi, V in enumerate(videos):
        # ---- LOVO: L3 只在"其余视频" crop 上训 ----
        pool = [c for vv, cs in crops_by_video.items() if vv != V for c in cs]
        train_items, val_items = _stratified_split(pool)
        model = train_model(train_items, val_items, epochs=EPOCHS, seed=0, device=DEVICE)
        # 诊断: 该 L3 模型在自身 val 上的精度
        val_acc = _eval_acc(model, torch.utils.data.DataLoader(
            _CropDataset(val_items, aug=False), batch_size=32), DEVICE)

        prior = derive_ped_priors([r["gt_wh"] for r in records if r["video"] != V])
        frames = group[V]
        tscore = compute_temporal_scores(frames)

        dv = 0
        sv_l2 = 0
        sv_l3 = {w: 0 for w in L3_WEIGHTS}
        # 预先对每帧算好 l3_scores(权重无关, 复用)
        per_frame_l3 = []
        for k, r in enumerate(frames):
            cands, gt = r["candidates"], r["gt_box_norm"]
            ts = tscore[k]
            sel2 = select_gtfree(cands, prior, temporal_scores=ts)
            fp = _frame_path(r["video"], r["fi"])
            try:
                frame_img = Image.open(fp).convert("RGB")
            except Exception:
                frame_img = None
            l3 = _build_l3_scores(model, frame_img, cands) if frame_img else {}
            per_frame_l3.append((cands, gt, ts, sel2, l3))
            has_det = any(_center_dist(c.get("box_norm"), gt) < CENTER_HIT
                          for c in cands if c.get("box_norm"))
            if has_det:
                dv += 1
                if sel2 is not None and _center_dist(sel2.get("box_norm"), gt) < CENTER_HIT:
                    sv_l2 += 1
                for w in L3_WEIGHTS:
                    sel3 = select_gtfree(cands, prior, temporal_scores=ts, l3_scores=l3, l3_weight=w)
                    if sel3 is not None and _center_dist(sel3.get("box_norm"), gt) < CENTER_HIT:
                        sv_l3[w] += 1

        acc_l2 = sv_l2 / dv if dv else None
        acc_l3 = {w: (sv_l3[w] / dv if dv else None) for w in L3_WEIGHTS}
        best_w = max(L3_WEIGHTS, key=lambda w: sv_l3[w])
        per_video[V] = {"detected": dv, "sel_l2": sv_l2, "sel_l3": sv_l3,
                        "acc_l2": acc_l2, "acc_l3": acc_l3, "best_w": best_w, "l3_val_acc": val_acc}
        det_total += dv
        sel_l2_total += sv_l2
        for w in L3_WEIGHTS:
            sel_l3_total[w] += sv_l3[w]
        print(f"  [{vi+1}/{len(videos)}] {V}: det={dv} L2_acc={acc_l2:.2f} "
              f"L3_best(w={best_w})={acc_l3[best_w]:.2f} l3_val_acc={val_acc:.2f} "
              f"({time.time()-t0:.1f}s)")

    n = len(records)
    det_cov = det_total / n if n else 0.0
    base_l2 = sel_l2_total / det_total if det_total else 0.0
    print(f"\n=== 检测覆盖(与 L3 无关): {det_cov*100:.1f}% ===")
    print(f"\n=== L1+L2 leave-some-out 选灯准确率: {base_l2*100:.1f}% (基线) ===")
    print("\n=== L1+L2+L3 leave-some-out (按 l3_weight 扫参) ===")
    for w in L3_WEIGHTS:
        a = sel_l3_total[w] / det_total if det_total else 0.0
        print(f"  l3_weight={w}: {a*100:.1f}%  (Δ { (a-base_l2)*100:+.1f}pp)")
    best_w = max(L3_WEIGHTS, key=lambda w: sel_l3_total[w])
    best_a = sel_l3_total[best_w] / det_total if det_total else 0.0
    print(f"\n[L3 门控结论] 最优 l3_weight={best_w}: {best_a*100:.1f}%  vs 基线 {base_l2*100:.1f}%  "
          f"(Δ {(best_a-base_l2)*100:+.1f}pp, leave-some-out)")
    print("[校准] LOVO 是新视频泛化代理(非生产保证); L3 每视频独立训练, 不在评测视频自身 crop 上训(护栏1)。")
    if args.smoke:
        print("[smoke] 仅 1 视频, 管线/计时验证通过。")


if __name__ == "__main__":
    main()
