"""analyze_signal_presence.py TDD: 锁定 255 bug 修复(布尔像素占比, 非 0/255 求和)。

跑法:
  PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_analyze_signal_presence.py -q
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analyze_signal_presence import signal_ratio


def _make_img(h=20, w=20, green_block=(2, 2)):
    """灰度底 + 左上角 green_block 大小纯绿(0,255,0) 块, 其余中性灰。"""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    gh, gw = green_block
    img[0:gh, 0:gw] = (0, 255, 0)
    return img


def test_signal_ratio_all_green_is_one():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = (0, 255, 0)  # 全绿
    assert abs(signal_ratio(img, "green") - 1.0) < 1e-6


def test_signal_ratio_small_green_block_boolean_not_inflated():
    # 2x2 绿块 / 20x20 = 4/400 = 0.01。旧 bug(mask.sum()/size)会得 255*0.01=2.55->夹紧1.0, 此测锁死布尔占比。
    img = _make_img(20, 20, green_block=(2, 2))
    assert abs(signal_ratio(img, "green") - 0.01) < 1e-6


def test_signal_ratio_quarter_green():
    img = _make_img(20, 20, green_block=(10, 10))  # 100/400 = 0.25
    assert abs(signal_ratio(img, "green") - 0.25) < 1e-6


def test_signal_ratio_red_on_red_block():
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    img[0:3, 0:3] = (0, 0, 255)  # 3x3 纯红
    assert abs(signal_ratio(img, "red") - 9 / 100) < 1e-6


def test_signal_ratio_empty_returns_none():
    assert signal_ratio(np.zeros((0, 0, 3), dtype=np.uint8), "green") is None
