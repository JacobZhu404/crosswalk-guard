"""Part B 模块级评测纯函数单元测试: band-IoU + 跟踪归属并集 + 静止准确率。

对应 docs/plans/2026-07-16-wb-plan-v4-fix-eval-measurement.md Part B。
TDD 先行: 重点验证 B2 "并集归属" 不会低估碎片化(这是 cc 收紧的点)。
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.module_metrics import (
    band_iou, mask_band, iou_box, attribution_union, stationary_accuracy,
)


def test_band_iou_full_overlap():
    assert abs(band_iou((100, 200), (100, 200)) - 1.0) < 1e-9


def test_band_iou_partial():
    # [100,200] vs [150,300]: 重叠 50, 并集 200 -> 0.25
    assert abs(band_iou((100, 200), (150, 300)) - 0.25) < 1e-9


def test_band_iou_no_overlap():
    assert band_iou((100, 200), (300, 400)) == 0.0


def test_band_iou_empty():
    assert band_iou(None, (100, 200)) == 0.0
    assert band_iou((100, 200), None) == 0.0


def test_mask_band_and_iou_with_gt():
    import numpy as np
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[373:464, :] = 255  # 斑马线带
    assert mask_band(mask) == (373, 463)
    # 预测带恰好命中 -> IoU=1
    assert abs(band_iou(mask_band(mask), (373, 463)) - 1.0) < 1e-9
    # 预测带错位(11 那种) -> 低 IoU
    assert band_iou(mask_band(mask), (256, 302)) == 0.0


def test_iou_box():
    a = [10, 10, 50, 50]
    b = [30, 30, 70, 70]
    # 交 20x20=400, 各自 1600, 并 2800 -> 400/2800
    assert abs(iou_box(a, b) - 400 / 2800) < 1e-9
    assert iou_box(a, [200, 200, 210, 210]) == 0.0
    assert iou_box(None, a) == 0.0


def test_attribution_union_counts_fragments_not_just_best():
    """核心: 同一辆车被切成 2 个 track(碎片化), 并集应计 2, 而非只取 IoU 最高那个=1。"""
    anchor_boxes = [[100, 370, 200, 460]]  # 违章车锚框
    track_samples = {
        # track 1: 窗内前段, 框与锚框 IoU 高
        1: [{"ts": 22.0, "stationary": True, "box": [102, 372, 198, 458]},
            {"ts": 24.0, "stationary": True, "box": [101, 371, 199, 459]}],
        # track 2: 窗内后段(碎片化), 框同样匹配锚框, 但 track_id 不同
        2: [{"ts": 26.0, "stationary": True, "box": [103, 373, 197, 461]},
            {"ts": 28.0, "stationary": True, "box": [100, 370, 200, 460]}],
        # track 9: 邻车, 框与锚框 IoU 低 -> 不应归属
        9: [{"ts": 23.0, "stationary": True, "box": [400, 380, 500, 470]}],
    }
    attr = attribution_union(track_samples, anchor_boxes, window=(21.0, 68.0), T=0.5)
    assert attr == {1, 2}          # 碎片化数 = 2(理想应为 1, 这里暴露缺陷)
    assert 9 not in attr           # 邻车不误归


def test_attribution_union_single_track_ideal():
    anchor_boxes = [[100, 370, 200, 460]]
    track_samples = {
        1: [{"ts": 22.0, "stationary": True, "box": [100, 370, 200, 460]},
            {"ts": 30.0, "stationary": True, "box": [101, 371, 199, 459]}],
    }
    attr = attribution_union(track_samples, anchor_boxes, window=(21.0, 68.0), T=0.5)
    assert attr == {1}


def test_stationary_accuracy():
    track_samples = {
        1: [{"ts": 22.0, "stationary": True},
            {"ts": 24.0, "stationary": True},
            {"ts": 26.0, "stationary": False}],  # 1/3 非静止
    }
    acc = stationary_accuracy(track_samples, {1}, window=(21.0, 68.0))
    assert abs(acc - 2 / 3) < 1e-9
    # 窗外样本不计入
    acc2 = stationary_accuracy(track_samples, {1}, window=(22.0, 24.0))
    assert acc2 == 1.0
