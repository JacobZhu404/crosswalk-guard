"""Phase B 训练:SignalStateClassifier 重训 -> models/ped_signal_v2.pt。

复用:
  - train_ped_signal.train_net / export_torch / _rows_to_dataset / _imgs_to_X
  - signal_state_classifier.LABELS / _INPUT / _build_net (网络结构单一真相源)
  - ped_signal_dataset.load_labeled_crops (crop_path 解析)

与旧脚本的关键差异(cc 硬要求):
  - 不覆盖基线: 默认输出 models/ped_signal_v2.pt。
  - 不用纯 --verified-only: 用全量(尊重 Jacob 修正, 排除已删行)。
  - 按 manifest 固定视频 split 评测(取代 LOVO, 禁泄漏切分)。
  - 源下采样(impostor_outside)在进 dataset 前按行做, 不动 loss
    (类级 CrossEntropyLoss(weight=...) 压不到 off 子源, 且 _rows_to_dataset 不带 source)。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/train_classifier_retrain.py
  PYTHONPATH=src ./.venv/bin/python scripts/train_classifier_retrain.py --drop-source impostor_outside   # 消融版
  PYTHONPATH=src ./.venv/bin/python scripts/train_classifier_retrain.py --downweight-source impostor_outside --downweight-ratio 0.5
"""
import os
import sys
import argparse
import json
import random

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from redlight.data_pipeline.ped_signal_dataset import load_labeled_crops
from redlight.models.signal_state_classifier import LABELS
from train_ped_signal import train_net, export_torch, _rows_to_dataset


# ------------------------------------------------------------------ 纯函数(可单测)
def load_manifest_split(manifest_path):
    """返回 (train_videos, val_videos) 列表, 来自 manifest.json 固化 split。"""
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    return m["split"]["train"], m["split"]["val"]


def exclude_deleted(rows):
    """去掉 label=='delete' 的行(apply 删图残留)。"""
    return [r for r in rows if r.get("label") not in (None, "", "delete")]


def downsample_source(rows, source, keep_ratio, seed=0):
    """进 dataset 前按行下采样某 source: 保留该源 keep_ratio 比例, 其余源不动。
    seeded 可复现。复用现有'先采样行再建 dataset'流程, 不动 loss。"""
    rng = random.Random(seed)
    kept, target = [], []
    for r in rows:
        (target if r.get("source") == source else kept).append(r)
    n = len(target)
    k = int(round(n * keep_ratio))
    rng.shuffle(target)
    return kept + target[:k]


def split_rows(rows, train_videos, val_videos):
    """按视频集合分 train/val(严禁按帧切)。"""
    tr = set(train_videos)
    va = set(val_videos)
    return ([r for r in rows if r["video"] in tr],
            [r for r in rows if r["video"] in va])


def main():
    ap = argparse.ArgumentParser(description="Phase B 重训行人信号灯分类器 -> ped_signal_v2.pt")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "classifier_retrain", "labels.csv"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "datasets", "classifier_retrain", "manifest.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "ped_signal_v2.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--balanced", dest="balanced", action="store_true", default=True)
    ap.add_argument("--no-balanced", dest="balanced", action="store_false")
    ap.add_argument("--downweight-source", default=None, help="按行下采样该 source 到 --downweight-ratio")
    ap.add_argument("--downweight-ratio", type=float, default=0.5)
    ap.add_argument("--drop-source", default=None, help="消融: 直接去除该 source 全部行")
    ap.add_argument("--no-eval", dest="eval_split", action="store_false", default=True,
                    help="跳过按 manifest split 的 train/val 评测")
    ap.add_argument("--seed", type=int, default=0, help="可复现种子(torch/np/random 全锁, 序列消融要求 gate 数字可比)")
    args = ap.parse_args()

    # ---- 可复现: _build_net 权重初始化走 torch 默认生成器, 不锁种子则每次训练 init 不同 ->
    #     gate 数字不可比, 序列消融失效。train_net 的 balanced 分支已用 np RandomState(seed),
    #     此处再锁 torch 默认生成器 + np/random 全局, 三处对齐。 ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- 数据装载 ----
    rows = load_labeled_crops(args.labels, verified_only=False)
    rows = exclude_deleted(rows)
    if args.drop_source:
        rows = [r for r in rows if r.get("source") != args.drop_source]
        print(f"[data] 去除 source={args.drop_source} -> {len(rows)} 行")
    if args.downweight_source:
        rows = downsample_source(rows, args.downweight_source, args.downweight_ratio)
        print(f"[data] 下采样 source={args.downweight_source} 到 {args.downweight_ratio} -> {len(rows)} 行")

    # ---- split ----
    train_v, val_v = load_manifest_split(args.manifest)
    tr_rows, va_rows = split_rows(rows, train_v, val_v)
    print(f"[split] train={len(tr_rows)} (视频 {len(train_v)}) / val={len(va_rows)} (视频 {len(val_v)})")

    Xtr, ytr = _rows_to_dataset(tr_rows)
    if Xtr is None or ytr is None:
        print("无训练数据, 退出")
        return
    net = train_net(Xtr, ytr, epochs=args.epochs, balanced=args.balanced)

    # ---- 评测(固定 split, 取代 LOVO) ----
    if args.eval_split and va_rows:
        Xva, yva = _rows_to_dataset(va_rows)
        if Xva is not None:
            from train_ped_signal import accuracy
            tr_acc = accuracy(net, Xtr, ytr)
            va_acc = accuracy(net, Xva, yva)
            print(f"[eval] train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  (固定视频 split, 非 LOVO)")

    # ---- 导出(原子替换, 沿用 apply 事故教训) ----
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tmp = args.out + ".tmp"
    export_torch(net, tmp)
    os.replace(tmp, args.out)   # 原子替换, 中途异常不会清空目标
    print(f"[export] 原子写入 -> {args.out}")


if __name__ == "__main__":
    main()
