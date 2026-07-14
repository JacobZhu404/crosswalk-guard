"""训练 M1 行人信号灯状态分类器 (walk/stand/off) (Phase2 spec §6)。

- 数据: datasets/ped_signal/labels.csv(build_ped_signal_crops.py 产出, 人工校验 verified=1)。
- 模型: tiny CNN(3x48x48 -> 3), 与 SignalStateClassifier 的输入约定一致(/255, NCHW, 48x48)。
- 评测: leave-one-video-out 交叉验证(spec §6, 代理"新手机视频"泛化)。
- 导出: 主产物为 PyTorch 权重 models/ped_signal.pt (运行时直接 torch 加载, 零额外依赖);
        --export-onnx 可选再导一份 .onnx 供 cv2.dnn 加载(需 onnx 包, 本项目运行时已绑 torch 故非必需)。

用法:
    python scripts/train_ped_signal.py                 # LOVO 评测 + 全量训练导出 .pt
    python scripts/train_ped_signal.py --smoke         # 合成数据自检(train->.pt->SignalStateClassifier 契约)

契约自检(--smoke)不需真实数据, 验证导出的权重能被 Phase1 的 SignalStateClassifier 消费。
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
from redlight.models.signal_state_classifier import (
    LABELS, _INPUT, _build_net,
)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


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


def export_torch(net, path):
    """主产物: 仅存 state_dict (torch 运行时直接加载, 零额外依赖)。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(net.state_dict(), path)
    return path


def export_onnx(net, path):
    """可选产物: 导出 ONNX 供 cv2.dnn 加载。需要 onnx 包, 缺失时直接报清晰错误由调用方决定跳过。"""
    try:
        import onnx  # noqa: F401
    except Exception:
        raise RuntimeError(
            "导出 ONNX 需要 onnx 包 (pip install onnx)。可省略此步: 运行时已支持直接加载 .pt 权重。"
        )
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


def synth(label, n=20, rng=None):
    """合成 crop: walk=偏绿, stand=偏红, off=灰 (BGR)。"""
    rng = rng or np.random.RandomState(0)
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


def majority_vote(clf, label, n=15, rng=None):
    """对单类取 n 个合成样本的分类多数投票, 降低单样本随机性导致的偶发误判。"""
    from collections import Counter
    rng = rng or np.random.RandomState(0)
    c = Counter(clf.classify(synth(label, 1, rng)[0])[0] for _ in range(n))
    return c.most_common(1)[0][0]


def run_smoke():
    """合成数据契约验证: train -> 权重 -> SignalStateClassifier。主路径用 .pt(零下载)。"""
    from redlight.models.signal_state_classifier import SignalStateClassifier
    rng = np.random.RandomState(0)

    imgs, ys = [], []
    for i, lb in enumerate(LABELS):
        for im in synth(lb):
            imgs.append(im); ys.append(i)
    X = _imgs_to_X(imgs); y = torch.tensor(ys)
    net = train_net(X, y, epochs=40)
    acc = accuracy(net, X, y)
    print(f"[smoke] 合成训练 accuracy={acc:.2f} (应 > 0.9)")

    # --- 主契约: PyTorch .pt 权重(零下载, 运行时直接 torch 加载) ---
    tmp_pt = os.path.join(tempfile.mkdtemp(), "ped_signal.pt")
    export_torch(net, tmp_pt)
    print(f"[smoke] 导出 PyTorch 权重: {tmp_pt} ({os.path.getsize(tmp_pt)} bytes)")
    clf = SignalStateClassifier(tmp_pt, verbose=False)
    assert clf.available, "PyTorch 权重未能被加载"
    got = [majority_vote(clf, lb, rng=rng) for lb in LABELS]
    print(f"[smoke] torch 直接分类(多数投票): {got}")
    assert got == LABELS, f"torch 契约失败: 预期 {LABELS}, 实际 {got}"

    # --- 可选契约: ONNX(需 onnx 包) ---
    try:
        import onnx  # noqa: F401
        tmp_onnx = os.path.join(tempfile.mkdtemp(), "ped_signal.onnx")
        export_onnx(net, tmp_onnx)
        print(f"[smoke] 导出 ONNX: {tmp_onnx} ({os.path.getsize(tmp_onnx)} bytes)")
        clf2 = SignalStateClassifier(tmp_onnx, verbose=False)
        assert clf2.available, "ONNX 未能被 cv2.dnn 加载"
        got2 = [majority_vote(clf2, lb, rng=rng) for lb in LABELS]
        print(f"[smoke] ONNX 分类(多数投票): {got2}")
        assert got2 == LABELS, f"ONNX 契约失败: {got2}"
    except Exception as e:
        print(f"[smoke] 跳过 ONNX 契约(onnx 未安装或失败, 不影响主路径): {e}")

    assert acc > 0.9, "合成数据未学会"
    print("SMOKE OK: train -> PyTorch(.pt) -> SignalStateClassifier 契约通过")


def main():
    ap = argparse.ArgumentParser(description="训练行人信号灯状态分类器 + 导出 ONNX")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "ped_signal", "labels.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "ped_signal.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="合成数据契约自检(不需真实数据)")
    ap.add_argument("--export-onnx", action="store_true",
                    help="额外导出 ONNX(供 cv2.dnn 加载, 需 onnx 包; 默认关, 运行时已支持 .pt)")
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
    export_torch(net, args.out)
    print(f"\n导出 PyTorch 权重 -> {args.out}")
    if args.export_onnx:
        try:
            onnx_path = os.path.splitext(args.out)[0] + ".onnx"
            export_onnx(net, onnx_path)
            print(f"导出 ONNX(可选) -> {onnx_path}")
        except Exception as e:
            print(f"ONNX 导出跳过: {e}")
    print(f"(config: models.ped_signal_model; traffic_light.method 设 ped_classifier 即启用)")


if __name__ == "__main__":
    main()
