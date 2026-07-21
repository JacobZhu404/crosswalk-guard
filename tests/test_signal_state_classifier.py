"""SignalStateClassifier / _build_net TDD: dropout 正则的"生产安全"契约。

核心契约: _build_net(dropout=0.0) 架构必须与旧版逐字节一致(8 模块, 无 Dropout),
否则生产权重 ped_signal.pt / ped_signal_v2.pt 等旧 .pt 无法 load_state_dict。
仅当 dropout>0 才多插一个无参 Dropout 模块, 且 eval 时自动关闭, 不影响推理。

跑法(项目 venv):
  PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_signal_state_classifier.py -q
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch
import numpy as np

from redlight.models.signal_state_classifier import SignalStateClassifier, _build_net, LABELS
from train_ped_signal import train_net, export_torch, _rows_to_dataset, _imgs_to_X

MODELS_DIR = os.path.join(ROOT, "models")
PROD_MODEL = os.path.join(MODELS_DIR, "ped_signal_v2.pt")  # 旧架构(dropout=0)基线


def test_build_net_default_is_legacy_8_modules():
    """dropout 默认 0.0 -> 仍 8 模块, 无 Dropout(生产权重可加载的硬契约)。"""
    net = _build_net()
    assert len(net) == 8, f"dropout=0 必须 8 模块, 实际 {len(net)}"
    kinds = [type(m).__name__ for m in net]
    assert "Dropout" not in kinds, "dropout=0 不应含 Dropout 模块"


def test_build_net_dropout_inserts_module():
    """dropout>0 -> 9 模块且含 Dropout(正则生效, 但架构变 9)。"""
    net = _build_net(dropout=0.3)
    assert len(net) == 9, f"dropout=0.3 必须 9 模块, 实际 {len(net)}"
    kinds = [type(m).__name__ for m in net]
    assert "Dropout" in kinds, "dropout>0 必须插入 Dropout 模块"


def test_prod_model_loads_with_dropout_zero():
    """生产/基线 .pt(dropout=0 架构)用 dropout=0 加载成功且可 classify。"""
    assert os.path.isfile(PROD_MODEL), f"基线权重缺失: {PROD_MODEL}"
    clf = SignalStateClassifier(PROD_MODEL, verbose=False, dropout=0.0)
    assert clf.available, "dropout=0 必须能加载旧架构权重"
    img = (np.random.rand(48, 48, 3) * 255).astype(np.uint8)
    label, conf = clf.classify(img)
    assert label in LABELS, "classify 应返回合法标签"


def test_prod_model_rejected_on_dropout_mismatch():
    """旧架构权重用 dropout=0.3 加载必须失败(架构不匹配, 不能静默误载)。"""
    assert os.path.isfile(PROD_MODEL), f"基线权重缺失: {PROD_MODEL}"
    clf = SignalStateClassifier(PROD_MODEL, verbose=False, dropout=0.3)
    assert not clf.available, "dropout 不匹配必须 available=False(架构护栏)"


def _synth_Xy(per=20):
    imgs, ys = [], []
    for i, lb in enumerate(LABELS):
        rng = np.random.RandomState(0)
        for _ in range(per):
            im = (rng.rand(48, 48, 3) * 40).astype(np.uint8)
            if lb == "walk":
                im[:, :, 1] = np.clip(im[:, :, 1] + 180, 0, 255)
            elif lb == "stand":
                im[:, :, 2] = np.clip(im[:, :, 2] + 180, 0, 255)
            else:
                im[:] = np.clip(im + 100, 0, 255)
            imgs.append(im)
            ys.append(i)
    return _imgs_to_X(imgs), torch.tensor(ys)


def test_train_net_dropout_roundtrip(tmp_path):
    """train_net(dropout=0.3) 导出 -> 必须用 dropout=0.3 加载; dropout=0 加载应失败。"""
    X, y = _synth_Xy()
    net = train_net(X, y, epochs=10, balanced=True, seed=1, dropout=0.3, weight_decay=1e-4)
    out = str(tmp_path / "reg.pt")
    export_torch(net, out)

    ok = SignalStateClassifier(out, verbose=False, dropout=0.3)
    assert ok.available, "dropout=0.3 训练的模型须用 dropout=0.3 加载"
    assert ok.classify((np.random.rand(48, 48, 3) * 255).astype(np.uint8))[0] in LABELS

    bad = SignalStateClassifier(out, verbose=False, dropout=0.0)
    assert not bad.available, "dropout 不匹配必须加载失败(护栏)"
