"""Phase B 训练 TDD: split 无泄漏 / 排除 delete / 源下采样 / 训练契约(smoke)。

跑法(项目 venv):
  PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_train_classifier_retrain.py -q
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch

from train_classifier_retrain import (
    load_manifest_split, exclude_deleted, downsample_source, split_rows,
)
from train_ped_signal import train_net, export_torch, _rows_to_dataset, _imgs_to_X
from redlight.models.signal_state_classifier import SignalStateClassifier, LABELS

MANIFEST = os.path.join(ROOT, "datasets", "classifier_retrain", "manifest.json")
ALL_VIDEOS = {f"违章{i:02d}" for i in range(1, 12)}


def test_load_manifest_split_no_leak():
    tr, va = load_manifest_split(MANIFEST)
    assert set(tr) & set(va) == set(), "train/val 必须不交"
    assert set(tr) | set(va) == ALL_VIDEOS, "并集必须覆盖全 11 视频"


def test_exclude_deleted():
    rows = [{"label": "walk"}, {"label": "delete"}, {"label": "off"},
            {"label": ""}, {"label": None}]
    out = exclude_deleted(rows)
    assert [r["label"] for r in out] == ["walk", "off"]


def test_downsample_source_half():
    rows = ([{"source": "impostor_outside", "label": "off"} for _ in range(100)] +
            [{"source": "prior_roi", "label": "walk"} for _ in range(50)])
    out = downsample_source(rows, "impostor_outside", 0.5, seed=1)
    n_out = sum(1 for r in out if r["source"] == "impostor_outside")
    n_pri = sum(1 for r in out if r["source"] == "prior_roi")
    assert n_out == 50, "下采样后该源行数应≈原半数"
    assert n_pri == 50, "其余源不受影响"


def test_downsample_source_drop():
    rows = ([{"source": "impostor_outside", "label": "off"} for _ in range(100)] +
            [{"source": "prior_roi", "label": "walk"} for _ in range(50)])
    out = downsample_source(rows, "impostor_outside", 0.0, seed=1)
    assert sum(1 for r in out if r["source"] == "impostor_outside") == 0
    assert sum(1 for r in out if r["source"] == "prior_roi") == 50


def test_downsample_seeded_reproducible():
    rows = [{"source": "impostor_outside", "label": "off"} for _ in range(100)]
    a = downsample_source(rows, "impostor_outside", 0.5, seed=7)
    b = downsample_source(rows, "impostor_outside", 0.5, seed=7)
    assert [id(r) for r in a] == [id(r) for r in b], "同 seed 应可复现"


def test_split_rows():
    tr, va = load_manifest_split(MANIFEST)
    rows = [{"video": tr[0], "label": "walk"}, {"video": va[0], "label": "stand"}]
    tr_r, va_r = split_rows(rows, tr, va)
    assert len(tr_r) == 1 and len(va_r) == 1


def _synth(label, n=20, rng=None):
    rng = rng or np.random.RandomState(0)
    out = []
    for _ in range(n):
        im = (rng.rand(48, 48, 3) * 40).astype(np.uint8)
        if label == "walk":
            im[:, :, 1] = np.clip(im[:, :, 1] + 180, 0, 255)
        elif label == "stand":
            im[:, :, 2] = np.clip(im[:, :, 2] + 180, 0, 255)
        else:
            im[:] = np.clip(im + 100, 0, 255)
        out.append(im)
    return out


def test_train_smoke_exports_v2(tmp_path):
    """合成数据 train -> 导出 v2 -> SignalStateClassifier 加载 + 三类多数投票正确。"""
    from collections import Counter
    rng = np.random.RandomState(0)
    imgs, ys = [], []
    for i, lb in enumerate(LABELS):
        for im in _synth(lb):
            imgs.append(im)
            ys.append(i)
    X = _imgs_to_X(imgs)
    y = torch.tensor(ys)
    net = train_net(X, y, epochs=20)

    out = str(tmp_path / "ped_signal_v2.pt")
    export_torch(net, out)
    clf = SignalStateClassifier(out, verbose=False)
    assert clf.available, "v2 权重应能被加载"

    def mv(lb):
        c = Counter(clf.classify(_synth(lb, 1, rng)[0])[0] for _ in range(15))
        return c.most_common(1)[0][0]
    assert [mv(lb) for lb in LABELS] == LABELS, "v2 应正确区分三类"


def _synth_Xy(per=20):
    imgs, ys = [], []
    for i, lb in enumerate(LABELS):
        for im in _synth(lb, n=per):
            imgs.append(im)
            ys.append(i)
    return _imgs_to_X(imgs), torch.tensor(ys)


def test_train_net_seed_threads_to_sampler():
    """seed 须穿透到平衡采样器(cc ruling 62aeaf1 盲区):

    - 同 seed + 同 torch init -> 权重完全一致(可复现);
    - 不同 seed -> 权重不同(证明 RandomState(seed) 真被采样序消费, 非冻在默认 0)。
    权重 init 用 torch 默认生成器, 故每调用前固定 torch.manual_seed 以隔离 seed 参数效应。
    """
    X, y = _synth_Xy()
    torch.manual_seed(0)
    n1 = train_net(X, y, epochs=5, balanced=True, seed=3)
    torch.manual_seed(0)
    n2 = train_net(X, y, epochs=5, balanced=True, seed=3)
    close = all(torch.allclose(p1, p2, atol=1e-5)
                for p1, p2 in zip(n1.parameters(), n2.parameters()))
    assert close, "同 seed 应完全可复现(权重 init 已固定)"

    torch.manual_seed(0)
    n3 = train_net(X, y, epochs=5, balanced=True, seed=9)
    diff = any(not torch.allclose(p1, p2, atol=1e-4)
               for p1, p2 in zip(n1.parameters(), n3.parameters()))
    assert diff, "不同 seed 应改变平衡采样序 -> 权重不同(seed 须穿透, 否则 ≥5 seed 共享同序)"
