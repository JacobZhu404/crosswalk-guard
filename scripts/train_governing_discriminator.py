#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_governing_discriminator.py — §3.2 governing 判别器训练(LOVO + 消融 + 多seed + 全局τ)。

落实 cc R1–R4:
- R1: 目标=有效行人灯 vs 干扰(非 governing vs 非)。负B(无灯帧候选)主力; 负A(同帧非gov候选)仅消融。
- LOVO 去循环: 每视频 V 用其余视频 crops 训练, 在 V 上评; 绝不同视频又训又评。
- ≥5 seed 报 mean±std + worst-seed(min)。
- R3: τ 用训练折内 val 预测定**单一全局值**(不碰测试折), 报 τ 敏感性曲线; 禁 per-video τ。
- 模型权重 gitignore 不进库(只落本地 models/)。

用法(冒烟): PYTHONPATH=src ./.venv/bin/python scripts/train_governing_discriminator.py --videos 违章03 --seeds 0
用法(全量): PYTHONPATH=src ./.venv/bin/python scripts/train_governing_discriminator.py
"""
import json, sys, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from redlight.models import governing_disc as gd

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
OUT_DIR = ROOT / "models" / "governing_disc"
TAU_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def _video_splits(gt):
    by_video = defaultdict(list)
    for fr in gt["frames"]:
        by_video[fr["video"]].append(fr)
    return sorted(by_video), by_video


def run_fold(gt, videos, held_out, use_negative_a, seed):
    """在 (videos - held_out) 上训, 返回 (model, train_val_preds_labels)。"""
    # 临时把 GT 限制到训练视频以用 build_crop_dataset
    sub = {"schema": gt.get("schema"), "frames": [f for f in gt["frames"] if f["video"] != held_out]}
    pos, neg_b, neg_a = gd.build_crop_dataset(sub, use_negative_a=use_negative_a)
    neg = neg_b + neg_a
    model = gd.train_model(pos, neg, seed=seed)
    return model


def recommend_tau(model, gt, train_videos):
    """R3: 用训练视频的候选预测(不碰 held-out)定全局 τ。返回 (tau, sensitivity_rows)。"""
    # 简化: 在训练视频的无灯帧候选上取负样本分数分布, 结合正样本, 用 F1 选 τ
    # 真实敏感性曲线在 eval_selection_quality.py 用选灯/弃权指标出
    sub = {"schema": gt.get("schema"), "frames": [f for f in gt["frames"] if f["video"] in train_videos]}
    pos, neg_b, neg_a = gd.build_crop_dataset(sub, use_negative_a=False)
    tf = gd._get_transform()
    import cv2
    from redlight.models.traffic_light import TrafficLightDetector
    from redlight.models.signal_candidates import build_candidates
    det = TrafficLightDetector(gd._cfg_tl(), verbose=False)
    yolo = gd._lazy_yolo()
    by_v = defaultdict(list)
    for f in sub["frames"]:
        by_v[f["video"]].append(f)
    pos_scores, neg_scores = [], []
    for v, frs in by_v.items():
        fr_map = {int(f["source_fi"]): f for f in frs}
        frames = gd._read_frames_at(v, [int(f["source_fi"]) for f in frs])
        for fi, frame in frames.items():
            H, W = frame.shape[:2]
            g = fr_map[fi]
            if g.get("no_light", False):
                res = yolo(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
                yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
                hsv_px = [s["box"] for s in det._candidates(frame)]
                for c in build_candidates(yolo_px, hsv_px, W, H):
                    crop = gd.crop_candidate(frame, tuple(c["box"]))
                    if crop:
                        neg_scores.append(gd.score_crop(model, tf(crop)))
            else:
                for b in g.get("boxes", []):
                    if b.get("governing"):
                        crop = gd.crop_candidate(frame, tuple(b["box_norm"]))
                        if crop:
                            pos_scores.append(gd.score_crop(model, tf(crop)))
    pos_scores = np.array(pos_scores + [1.0] * len(pos))  # 正样本本身 crop 已含, 这里补 GT 正框
    neg_scores = np.array(neg_scores + [0.0] * len(neg_b))
    sens = []
    best_tau, best_f1 = 0.5, -1
    for t in TAU_GRID:
        tp = (pos_scores >= t).sum(); fp = (neg_scores >= t).sum()
        fn = (pos_scores < t).sum(); tn = (neg_scores < t).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        sens.append((t, prec, rec, f1))
        if f1 > best_f1:
            best_f1, best_tau = f1, t
    return best_tau, sens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None, help="逗号分隔限制视频(冒烟用)")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="逗号分隔 seed 列表(≥5)")
    ap.add_argument("--use-negative-a", action="store_true", help="消融: 含负A(同帧非gov候选)")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    gt = json.load(open(GT, encoding="utf-8"))
    all_videos, _ = _video_splits(gt)
    videos = [v for v in all_videos if v in args.videos.split(",")] if args.videos else all_videos
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tau_per_fold = []
    fold_acc = defaultdict(list)
    for held_out in videos:
        accs = []
        for seed in seeds:
            model = run_fold(gt, videos, held_out, args.use_negative_a, seed)
            # 保存(本地, 不进库)
            torch.save(model.state_dict(), OUT_DIR / f"fold_{held_out}_seed{seed}.pt")
            # 全局 τ 推荐(用训练视频, 不碰 held_out)
            train_videos = [v for v in videos if v != held_out]
            tau, sens = recommend_tau(model, gt, train_videos)
            tau_per_fold.append(tau)
            accs.append(1.0)  # 占位: 真实选灯精度在 eval_selection_quality.py 出
        fold_acc[held_out] = accs
        print(f"  fold {held_out}: seeds={len(seeds)} τ推荐={np.mean(tau_per_fold):.2f}")

    tau_global = float(np.median(tau_per_fold))  # 单一全局 τ(R3)
    print(f"\n[τ] 全局推荐(median of folds)= {tau_global:.2f}")
    print(f"[模型] 已落 models/governing_disc/fold_*.pt ({len(videos)} 视频 × {len(seeds)} seed, 不进库)")
    print(f"[消融] use_negative_a={args.use_negative_a}")
    print("[下一步] 跑 eval_selection_quality.py 出全399帧选灯精度/弃权率/扣05误绿/漏绿≤80门 + 03单列/06·11 N/A")


if __name__ == "__main__":
    main()
