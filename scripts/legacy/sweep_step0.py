"""Step0 正则化序列消融: ≥5 seed 跑正则化模型, 复用 diag 四关纯函数, 聚合 mean±std+min。

与 cc ruling 62aeaf1 对齐: 不凭单 seed 判版本, 必须 ≥5 seed 的 mean±std **与 worst-seed(min)**。
本脚本在进程内复用 train_net(正则化插管) + diag 的 domain_confusion_v2 / scan_video / gate_* 纯函数,
避免多次子进程 + JSON 解析导致口径漂移。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/sweep_step0.py \
      --seeds 0 1 2 3 4 --dropout 0.3 --weight-decay 1e-4 --epochs 30 \
      --out-json data/output/sweep_step0.json
"""
import os
import sys
import json
import time
import argparse
import types
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch

from train_classifier_retrain import (
    load_labeled_crops, exclude_deleted, split_rows, load_manifest_split, _rows_to_dataset,
)
from train_ped_signal import train_net, export_torch
from diag_classifier_retrain import (
    domain_confusion_v2, scan_video, gate_neg_off_ratio, gate_true_green_recall,
    scan_window, gate_probe_window, parse_probe_window, NEG_VIDEOS, VIDEOS,
)
from redlight.models.signal_state_classifier import SignalStateClassifier
from redlight.models.traffic_light import TrafficLightDetector

LABELS_CSV = os.path.join(ROOT, "datasets", "classifier_retrain", "labels.csv")
MANIFEST = os.path.join(ROOT, "datasets", "classifier_retrain", "manifest.json")
MODELS_DIR = os.path.join(ROOT, "models")


def train_one(seed, dropout, wd, epochs, out_pt, supplemental_csv=None):
    rows = exclude_deleted(load_labeled_crops(LABELS_CSV, verified_only=False))
    if supplemental_csv:
        sup = exclude_deleted(load_labeled_crops(supplemental_csv, verified_only=False))
        print(f"[data] 补充负例质量样本: {len(sup)} 行")
        rows.extend(sup)
    train_v, val_v = load_manifest_split(MANIFEST)
    tr_rows, _ = split_rows(rows, train_v, val_v)
    Xtr, ytr = _rows_to_dataset(tr_rows)
    if Xtr is None:
        raise RuntimeError("无训练数据")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    net = train_net(Xtr, ytr, epochs=epochs, balanced=True, seed=seed,
                    dropout=dropout, weight_decay=wd)
    export_torch(net, out_pt)
    return out_pt


def gate_one(model_path, dropout, probe):
    clf = SignalStateClassifier(model_path, verbose=False, dropout=dropout)
    if not clf.available:
        raise RuntimeError(f"模型加载失败: {model_path}")
    cfg = types.SimpleNamespace(
        traffic_light=None,
        models=types.SimpleNamespace(ped_signal_model=model_path),
    )
    det = TrafficLightDetector(cfg, verbose=False)

    d = domain_confusion_v2(LABELS_CSV, clf)
    train_v, val_v = (json.load(open(MANIFEST))["split"]["train"],
                      json.load(open(MANIFEST))["split"]["val"])
    videos = []
    for v in VIDEOS:
        rec = scan_video(det, clf, v)
        if rec is not None:
            videos.append(rec)

    g1p, g1 = gate_neg_off_ratio(videos, NEG_VIDEOS)
    g2_06p, g2_06 = gate_true_green_recall(videos, {"违章06"})
    g2_07p, g2_07 = gate_true_green_recall(videos, {"违章07"})
    g4_01p, g4_01 = gate_neg_off_ratio(videos, {"违章01"})
    g4_07p, g4_07 = gate_true_green_recall(videos, {"违章07"})
    g4_11p, g4_11 = gate_neg_off_ratio(videos, {"违章11"})

    probe_val = None
    probe_pass = None
    if probe:
        pv, t0, t1 = parse_probe_window(probe)
        prec = scan_window(det, clf, pv, t0, t1)
        if prec:
            probe_pass, probe_val = gate_probe_window(prec["walk"], prec["off"], prec["other"])

    return {
        "gate1_neg_off": g1,
        "gate2_06_train": g2_06,
        "gate2_07_val": g2_07,
        "gate4_01_neg_off": g4_01,
        "gate4_07_pos_recall": g4_07,
        "gate4_11_neg_off": g4_11,
        "gate3_probe_04": probe_val,
        "walk_to_off": d["walk_to_off"],
    }


def _agg(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return None
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "n": len(arr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--probe", default="违章04:42.0:43.2")
    ap.add_argument("--out-json", default=os.path.join(ROOT, "data", "output", "sweep_step0.json"))
    ap.add_argument("--supplemental-csv", default=None,
                    help="负例质量杠杆: 补充负视频假绿裁剪 labels.csv")
    ap.add_argument("--model-prefix", default="ped_signal_v2_s0reg",
                    help="模型文件名前缀(默认 ped_signal_v2_s0reg; 负例杠杆用 ped_signal_v2_negqual)")
    args = ap.parse_args()

    per_seed = {}
    for seed in args.seeds:
        t0 = time.time()
        out_pt = os.path.join(MODELS_DIR, f"{args.model_prefix}_s{seed}.pt")
        train_one(seed, args.dropout, args.weight_decay, args.epochs, out_pt, supplemental_csv=args.supplemental_csv)
        g = gate_one(out_pt, args.dropout, args.probe)
        per_seed[seed] = g
        dt = time.time() - t0
        print(f"[seed {seed}] 用时 {dt:.1f}s  关1={g['gate1_neg_off']:.3f} "
              f"关2_07val={g['gate2_07_val']:.3f} 关4_07={g['gate4_07_pos_recall']:.3f} "
              f"关4_01={g['gate4_01_neg_off']:.3f} 关4_11={g['gate4_11_neg_off']:.3f} "
              f"关3={g['gate3_probe_04']}", flush=True)

    metrics = list(next(iter(per_seed.values())).keys())
    agg = {m: _agg([per_seed[s].get(m) for s in args.seeds]) for m in metrics}

    out = {
        "config": {
            "seeds": args.seeds, "dropout": args.dropout,
            "weight_decay": args.weight_decay, "epochs": args.epochs,
            "probe": args.probe,
            "supplemental_csv": args.supplemental_csv,
            "note": "Step0 正则化(非加容量); 负例杠杆(补充假绿 off) 复用 diag 四关纯函数",
        },
        "per_seed": per_seed,
        "agg_mean_std_min": agg,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nJSON -> {args.out_json}")
    print("\n=== 聚合 (mean±std, worst-seed min) ===")
    for m in metrics:
        a = agg[m]
        if a:
            print(f"  {m:20s} mean={a['mean']:.3f}  std={a['std']:.3f}  min={a['min']:.3f}  max={a['max']:.3f}")


if __name__ == "__main__":
    main()
