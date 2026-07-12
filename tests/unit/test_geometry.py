"""geometry 工具单元测试: 重点覆盖 D2 占用比例分母 (box vs mask)。"""
import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.geometry import compute_overlap_ratio


def _mask_and_box():
    # 斑马线 mask: 200x200 的方块(面积 40000)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[50:250, 50:250] = 255
    # 车框: 完全落在 mask 内, 但只占 mask 一部分(100x100=10000)
    box = [75, 75, 175, 175]
    return mask, box


def test_denom_box_is_fraction_of_box():
    mask, box = _mask_and_box()
    # box 面积 = 100x100 = 10000, 全部在 mask 内 -> inside=10000, 分母=box -> 1.0
    r = compute_overlap_ratio(box, mask, denom="box")
    assert abs(r - 1.0) < 1e-6


def test_denom_mask_is_fraction_of_mask():
    mask, box = _mask_and_box()
    # inside=10000, mask 面积=40000 -> 0.25 (斑马线被该车覆盖 25%)
    r = compute_overlap_ratio(box, mask, denom="mask")
    assert abs(r - 0.25) < 1e-6


def test_footprint_shrinks_intersection():
    mask, box = _mask_and_box()
    # footprint=0.5 只取下半部(75..175 的 y 下半 -> 125..175, 50x100=5000)
    r = compute_overlap_ratio(box, mask, footprint=0.5, denom="mask")
    assert abs(r - 0.125) < 1e-6   # inside=5000 / mask 40000


def test_no_mask_returns_zero():
    box = [0, 0, 10, 10]
    assert compute_overlap_ratio(box, None) == 0.0


def test_box_outside_mask_zero():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:50, 0:50] = 255
    box = [60, 60, 90, 90]   # 完全在 mask 外
    assert compute_overlap_ratio(box, mask, denom="mask") == 0.0
