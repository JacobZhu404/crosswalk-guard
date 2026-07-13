"""训练 M1 行人信号灯状态分类器 (walk/stand/off) 并导出 ONNX (Phase2 spec §6)。

- 数据: datasets/ped_signal/labels.csv(build_ped_signal_crops.py 产出, 人工校验 verified=1)。
- 模型: tiny CNN(3x48x48 -> 3), 与 SignalStateClassifier 的输入约定一致(/255, NCHW, 48x48)。
- 评测: leave-one-video-out 交叉验证(spec §6, 代理"新手机视频"泛化)。
- 导出: torch.onnx -> models/ped_signal.onnx, 运行时由 cv2.dnn 加载(不依赖 torch)。

用法:
    python scripts/train_ped_signal.py                 # LOVO 评测 + 全量训练导出 ONNX
    python scripts/train_ped_signal.py --smoke         # 合成数据自检(train->onnx->cv2.dnn 契约)

契约自检(--smoke)不需真实数据, 验证导出的 ONNX 能被 Phase1 的 SignalStateClassifier 消费。
"""
import os
import sys
import argparse
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.data_pipeline.ped_signal_dataset import (
    load_labeled_crops, lovo_folds,
)
from redlight.models.signal_state_classifier import LABELS, _INPUT

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _build_net():
    import torch.nn as nn
    # 固定尺寸(无 adaptive pool, 便于 cv2.dnn): 48->24->12, 16*12*12=2304
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 12 * 12, 3),
    )


def _imgs_to_X(imgs):
    """[HxWx3 BGR uint8] -> torch (N,3,48,48) float[0,1] (与 SignalStateClassifier 一致)。"""
    import cv2
    arr = np.stack([cv2.resize(im, (_INPUT, _INPUT)).astype(np.float32) / 255.0 for im in imgs])
    return torch.from_numpy(arr.transpose(0, 3, 1, 2))


def _rows_to_dataset(rows):
    from redlight.infrastructure.image_utils import robust_imread
    imgs, ys = [], []
    for r in rows:
        im = robust_imread(r["crop_path"])
        if im is None:
            continue
        imgs.append(im)
        ys.append(LABELS.index(r["label"]))
    if not imgs:
        return None, None
    return _imgs_to_X(imgs), torch.tensor(ys, dtype=torch.long)


def train_net(X, y, epochs=30, lr=1e-3):
    net = _build_net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(X), y)
        loss.backward()
        opt.step()
    net.eval()
    return net


def accuracy(net, X, y):
    with torch.no_grad():
        pred = net(X).argmax(1)
    return float((pred == y).float().mean())


def export_onnx(net, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dummy = torch.zeros(1, 3, _INPUT, _INPUT)
    # dynamo=False 用传统 TorchScript 导出器: 不依赖 onnxscript(新导出器需要), 跨机 torch 版本更稳,
    # 且产出的 ONNX 更契合 cv2.dnn。
    try:
        torch.onnx.export(net, dummy, path, input_names=["input"],
                          output_names=["logits"], opset_version=12, dynamo=False)
    except TypeError:
        # 老版本 torch 无 dynamo 参数
        torch.onnx.export(net, dummy, path, input_names=["input"],
                          output_names=["logits"], opset_version=12)
    return path


def run_smoke():
    """合成数据: walk=偏绿, stand=偏红, off=灰。train->onnx->cv2.dnn 契约验证。"""
    import cv2
    from redlight.models.signal_state_classifier import SignalStateClassifier
    rng = np.random.RandomState(0)

    def synth(label, n=20):
        out = []
        for _ in range(n):
            im = (rng.rand(48, 48, 3) * 40).astype(np.uint8)
            if label == "walk":
                im[:, :, 1] = np.clip(im[:, :, 1] + 180, 0, 255)   # G high (BGR)
            elif label == "stand":
                im[:, :, 2] = np.clip(im[:, :, 2] + 180, 0, 255)   # R high
            else:
                im[:] = np.clip(im + 100, 0, 255)                   # gray
            out.append(im)
        return out

    imgs, ys = [], []
    for i, lb in enumerate(LABELS):
        for im in synth(lb):
            imgs.append(im); ys.append(i)
    X = _imgs_to_X(imgs); y = torch.tensor(ys)
    net = train_net(X, y, epochs=40)
    acc = accuracy(net, X, y)
    print(f"[smoke] 合成训练 accuracy={acc:.2f} (应 > 0.9)")

    tmp = os.path.join(tempfile.mkdtemp(), "ped_signal.onnx")
    export_onnx(net, tmp)
    print(f"[smoke] 导出 ONNX: {tmp} ({os.path.getsize(tmp)} bytes)")

    clf = SignalStateClassifier(tmp, verbose=False)
    assert clf.available, "ONNX 未能被 cv2.dnn 加载"
    # 用 Phase1 wrapper 分类三个合成样本
    got = [clf.classify(synth(lb, 1)[0])[0] for lb in LABELS]
    print(f"[smoke] cv2.dnn 分类结果: {got}")
    assert acc > 0.9, "合成数据未学会"
    print("SMOKE OK: train -> ONNX -> cv2.dnn(SignalStateClassifier) 契约通过")


def main():
    ap = argparse.ArgumentParser(description="训练行人信号灯状态分类器 + 导出 ONNX")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "ped_signal", "labels.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "ped_signal.onnx"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="合成数据契约自检(不需真实数据)")
    ap.add_argument("--verified-only", action="store_true", default=True)
    args = ap.parse_args()

    if not _HAS_TORCH:
        print("需要 torch: pip install torch")
        return
    if args.smoke:
        run_smoke()
        return

    rows = load_labeled_crops(args.labels, verified_only=args.verified_only)
    if not rows:
        print(f"无已校验 crop: {args.labels}。先跑 build_ped_signal_crops.py 并在画廊人工校验 verified=1。")
        return

    # LOVO 交叉验证
    folds = lovo_folds(rows, verified_only=args.verified_only)
    accs = []
    for tv, train, test in folds:
        Xtr, ytr = _rows_to_dataset(train)
        Xte, yte = _rows_to_dataset(test)
        if Xtr is None or Xte is None:
            print(f"[LOVO {tv}] 跳过(空)"); continue
        net = train_net(Xtr, ytr, epochs=args.epochs)
        a = accuracy(net, Xte, yte)
        accs.append(a)
        print(f"[LOVO 留出 {tv}] test_acc={a:.3f} (train={len(train)} test={len(test)})")
    if accs:
        print(f"\nLOVO 平均 test_acc = {sum(accs)/len(accs):.3f}  ({len(accs)} 折)")

    # 全量训练 + 导出
    X, y = _rows_to_dataset(rows)
    net = train_net(X, y, epochs=args.epochs)
    export_onnx(net, args.out)
    print(f"\n导出 ONNX -> {args.out} (config: models.ped_signal_onnx; method 设 ped_classifier 即启用)")


if __name__ == "__main__":
    main()
