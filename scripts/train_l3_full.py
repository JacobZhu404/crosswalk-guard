#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_l3_full.py — 训一个全量 L3 单模型(797 裁图), 作"量误绿"测量的"当前最好选灯"夹具。

数据: data/output/l3_labels/manifest.json + labels.json(203 簇人工标) → build_labeled_crops
      (manual203 + auto_ped48 + auto_other546 = 797)。
训于 Jacob 标注(非 GT 播种, 符合护栏1)。输出 models/l3_ped_full.pt。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/train_l3_full.py
"""
import json, sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from redlight.models.l3_ped_vehicle import (
    build_labeled_crops, train_model, _IDX, CLASSES, _eval_acc, _CropDataset,
    L3PedVehicleNet,
)
import torch

MANIFEST = ROOT / "data" / "output" / "l3_labels" / "manifest.json"
LABELS = ROOT / "data" / "output" / "l3_labels" / "labels.json"
OUT = ROOT / "models" / "l3_ped_full.pt"


def main():
    manifest = json.load(open(MANIFEST))
    raw = json.load(open(LABELS))
    # labels.json 可能是 [{id,label}] 列表或 {id:label} 字典
    labels = {int(x["id"]): x["label"] for x in raw} if isinstance(raw, list) else \
             {int(k): v for k, v in raw.items()}
    crops = build_labeled_crops(manifest, labels)
    print(f"总标注裁图: {len(crops)}  (manual+auto_ped+auto_other)")
    from collections import Counter
    print("  类别分布:", dict(Counter(c["label"] for c in crops)))

    # 转 train_items = [(video, fi, box_norm), label_idx]; 确定性 90/10 切分做诊断 val
    random.seed(0)
    items = [((c["video"], c["fi"], tuple(c["box_norm"])), _IDX[c["label"]]) for c in crops]
    random.shuffle(items)
    n_val = max(1, int(0.1 * len(items)))
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"  train={len(train_items)} val={len(val_items)}")

    model = train_model(train_items, val_items, epochs=40, lr=1e-3,
                        weight_decay=1e-4, batch_size=32, seed=0, device="cpu", verbose=False)
    val_dl = _CropDataset(val_items, aug=False)
    from torch.utils.data import DataLoader
    vac = _eval_acc(model, DataLoader(val_dl, batch_size=32, shuffle=False, num_workers=0), "cpu")
    print(f"  val_acc={vac:.3f}")

    torch.save(model.state_dict(), str(OUT))
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
