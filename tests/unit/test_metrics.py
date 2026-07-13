"""评测指标单元测试 (要求 #6 的硬性覆盖)。

覆盖: 检测 P/R/F1/mAP, OCR 字符/整牌/编辑距离/省份, 区间 Temporal-IoU, 事件 P/R/F1。
全部用手工构造数据, 不依赖任何模型, 秒级跑完。
"""
import sys
import os
import math

import pytest

# 让 tests/ 能 import src/redlight
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from redlight.evaluation.metrics import (
    precision_recall_f1, match_detections, detection_metrics, compute_map,
    levenshtein, edit_similarity, char_accuracy, plate_accuracy, province_accuracy,
    temporal_iou, event_metrics, mask_iou, mota_metrics,
)


# ---------------- 1. 检测 P/R/F1 ----------------
def test_prf1_perfect():
    p, r, f1 = precision_recall_f1(10, 0, 0)
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_prf1_all_wrong():
    p, r, f1 = precision_recall_f1(0, 5, 5)
    assert p == 0.0 and r == 0.0 and f1 == 0.0


def test_prf1_no_predictions_safe():
    p, r, f1 = precision_recall_f1(0, 0, 3)
    assert p == 0.0 and r == 0.0 and f1 == 0.0


def test_detection_metrics_perfect_match():
    preds = [[0, 0, 10, 10], [20, 20, 30, 30]]
    gts = [[0, 0, 10, 10], [20, 20, 30, 30]]
    m = detection_metrics(preds, gts, iou_thr=0.5)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0
    assert math.isclose(m["f1"], 1.0)


def test_detection_metrics_fp_fn():
    # 一个预测偏移到无重叠 -> FP; 一个 gt 未被匹配 -> FN
    preds = [[0, 0, 10, 10], [200, 200, 210, 210]]
    gts = [[0, 0, 10, 10], [100, 100, 110, 110]]
    m = detection_metrics(preds, gts, iou_thr=0.5)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    # P=1/2, R=1/2, F1=0.5
    assert math.isclose(m["precision"], 0.5)
    assert math.isclose(m["recall"], 0.5)
    assert math.isclose(m["f1"], 0.5)


def test_match_detections_iou_threshold():
    # 低重叠不应匹配
    preds = [[0, 0, 10, 10]]
    gts = [[8, 8, 18, 18]]  # IoU ~0.01
    tp, fp, fn, matches = match_detections(preds, gts, iou_thr=0.5)
    assert tp == 0 and fp == 1 and fn == 1


# ---------------- 1b. mAP ----------------
def test_compute_map_perfect():
    preds = [[{"box": [0, 0, 10, 10], "score": 0.9, "cls": "car"}]]
    gts = [[{"box": [0, 0, 10, 10], "cls": "car"}]]
    res = compute_map(preds, gts, iou_thr=0.5)
    assert math.isclose(res["mAP"], 1.0, abs_tol=1e-6)


def test_compute_map_all_fp():
    preds = [[{"box": [200, 200, 210, 210], "score": 0.9, "cls": "car"}]]
    gts = [[{"box": [0, 0, 10, 10], "cls": "car"}]]
    res = compute_map(preds, gts, iou_thr=0.5)
    assert res["mAP"] == 0.0


def test_compute_map_two_classes():
    preds = [[
        {"box": [0, 0, 10, 10], "score": 0.9, "cls": "car"},
        {"box": [0, 0, 10, 10], "score": 0.8, "cls": "bus"},
    ]]
    gts = [[
        {"box": [0, 0, 10, 10], "cls": "car"},
        {"box": [0, 0, 10, 10], "cls": "bus"},
    ]]
    res = compute_map(preds, gts, iou_thr=0.5)
    assert math.isclose(res["mAP"], 1.0, abs_tol=1e-6)
    assert res["per_class"]["car"] == 1.0 and res["per_class"]["bus"] == 1.0


# ---------------- 2. OCR ----------------
def test_levenshtein_basic():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "ab") == 1
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3


def test_edit_similarity():
    assert math.isclose(edit_similarity("京A12345", "京A12345"), 1.0)
    assert math.isclose(edit_similarity("京A12345", "京B12345"), 6 / 7)
    assert edit_similarity("", "") == 1.0


def test_char_accuracy():
    # 末位错, 7 字符对 6 -> 6/7
    assert math.isclose(char_accuracy("京A1234X", "京A12345"), 6 / 7)


def test_plate_accuracy():
    preds = ["京A12345", "京B67890", "京C11111"]
    gts = ["京A12345", "京B67890", "京C99999"]
    assert math.isclose(plate_accuracy(preds, gts), 2 / 3)


def test_province_accuracy():
    preds = ["京A12345", "沪B67890"]
    gts = ["京C11111", "沪D22222"]
    assert math.isclose(province_accuracy(preds, gts), 1.0)


# ---------------- 3. Temporal-IoU ----------------
def test_temporal_iou_full_overlap():
    assert temporal_iou((0, 10), (0, 10)) == 1.0


def test_temporal_iou_partial():
    # [0,10] vs [5,15]: inter=5, union=15 -> 1/3
    assert math.isclose(temporal_iou((0, 10), (5, 15)), 1 / 3)


def test_temporal_iou_no_overlap():
    assert temporal_iou((0, 5), (10, 15)) == 0.0


# ---------------- 4. 事件 P/R/F1 ----------------
def test_event_metrics_perfect():
    preds = [{"track_id": 1, "start_ts": 0, "end_ts": 10, "status": "confirmed"}]
    gts = [{"track_id": 1, "start_ts": 0, "end_ts": 10, "status": "confirmed"}]
    m = event_metrics(preds, gts, tiou_thr=0.5)
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0
    assert math.isclose(m["f1"], 1.0)


def test_event_metrics_fp_fn():
    preds = [{"track_id": 1, "start_ts": 0, "end_ts": 10, "status": "confirmed"}]
    gts = [{"track_id": 2, "start_ts": 50, "end_ts": 60, "status": "confirmed"}]
    m = event_metrics(preds, gts, tiou_thr=0.5, match_by_track=False)
    # 不同 track + 无时间重叠 -> 不匹配
    assert m["tp"] == 0 and m["fp"] == 1 and m["fn"] == 1


def test_event_metrics_track_mismatch_blocks():
    preds = [{"track_id": 1, "start_ts": 0, "end_ts": 10, "status": "confirmed"}]
    gts = [{"track_id": 2, "start_ts": 0, "end_ts": 10, "status": "confirmed"}]
    m = event_metrics(preds, gts, tiou_thr=0.5, match_by_track=True)
    # 同时间段但 track 不同 -> 不认为匹配
    assert m["fp"] == 1 and m["fn"] == 1


def test_event_metrics_tiou_threshold():
    preds = [{"track_id": 1, "start_ts": 0, "end_ts": 10, "status": "confirmed"}]
    gts = [{"track_id": 1, "start_ts": 9, "end_ts": 11, "status": "confirmed"}]
    # [0,10] vs [9,11]: inter=1, union=12 -> 1/12 < 0.5 -> 不匹配
    m = event_metrics(preds, gts, tiou_thr=0.5, match_by_track=True)
    assert m["fp"] == 1 and m["fn"] == 1


# ---------------- 5. Mask IoU ----------------
def test_mask_iou_perfect_overlap():
    m = np.zeros((100, 100), dtype=np.uint8)
    m[20:80, 20:80] = 255
    assert math.isclose(mask_iou(m, m), 1.0)


def test_mask_iou_no_overlap():
    a = np.zeros((100, 100), dtype=np.uint8)
    a[0:50, 0:50] = 255
    b = np.zeros((100, 100), dtype=np.uint8)
    b[50:100, 50:100] = 255
    assert mask_iou(a, b) == 0.0


def test_mask_iou_partial():
    a = np.zeros((100, 100), dtype=np.uint8)
    a[0:60, 0:60] = 255   # area 3600
    b = np.zeros((100, 100), dtype=np.uint8)
    b[40:100, 40:100] = 255  # area 3600
    # inter = [40:60, 40:60] = 20*20 = 400
    # union = 3600 + 3600 - 400 = 6800
    assert math.isclose(mask_iou(a, b), 400 / 6800, abs_tol=1e-9)


def test_mask_iou_both_empty():
    a = np.zeros((100, 100), dtype=np.uint8)
    b = np.zeros((100, 100), dtype=np.uint8)
    assert mask_iou(a, b) == 1.0  # 两者皆空视为完全一致


def test_mask_iou_one_empty():
    a = np.zeros((100, 100), dtype=np.uint8)
    a[0:50, 0:50] = 255
    b = np.zeros((100, 100), dtype=np.uint8)
    assert mask_iou(a, b) == 0.0


def test_mask_iou_shape_mismatch():
    a = np.zeros((100, 100), dtype=np.uint8)
    b = np.zeros((50, 50), dtype=np.uint8)
    assert mask_iou(a, b) == 0.0


# ---------------- 6. MOTA ----------------
def test_mota_perfect():
    """两帧, 每帧 1 个 pred 完美匹配 1 个 gt, ID 稳定 -> MOTA=1.0。"""
    pred_frames = [
        {1: [0, 0, 10, 10]},
        {1: [0, 0, 10, 10]},
    ]
    gt_frames = [
        {10: [0, 0, 10, 10]},
        {10: [0, 0, 10, 10]},
    ]
    m = mota_metrics(pred_frames, gt_frames, iou_thr=0.5)
    assert m["mota"] == 1.0
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["id_switches"] == 0


def test_mota_id_switch():
    """两帧, 第1帧 gt10 匹配 pred1; 第2帧 gt10 匹配 pred2 -> 1 次 ID switch。"""
    pred_frames = [
        {1: [0, 0, 10, 10]},
        {2: [0, 0, 10, 10]},   # 同一个物理目标被不同 pred track 接走
    ]
    gt_frames = [
        {10: [0, 0, 10, 10]},
        {10: [0, 0, 10, 10]},
    ]
    m = mota_metrics(pred_frames, gt_frames, iou_thr=0.5)
    # tp=2, fp=0, fn=0, idsw=1, gt=2
    # mota = 1 - (0+0+1)/2 = 0.5
    assert m["tp"] == 2
    assert m["id_switches"] == 1
    assert math.isclose(m["mota"], 0.5)


def test_mota_fp_fn():
    """第1帧: 2 pred vs 1 gt -> 1 FP; 第2帧: 1 pred vs 2 gt -> 1 FN。"""
    pred_frames = [
        {1: [0, 0, 10, 10], 2: [100, 100, 110, 110]},
        {1: [0, 0, 10, 10]},
    ]
    gt_frames = [
        {10: [0, 0, 10, 10]},
        {10: [0, 0, 10, 10], 20: [50, 50, 60, 60]},
    ]
    m = mota_metrics(pred_frames, gt_frames, iou_thr=0.5)
    # frame1: tp=1, fp=1, fn=0
    # frame2: tp=1, fp=0, fn=1
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["id_switches"] == 0
    # mota = 1 - (1+1+0)/3 = 1/3
    assert math.isclose(m["mota"], 1 / 3)


def test_mota_no_gt():
    """无 gt 帧 -> MOTA 定义为 1.0 (无 ground truth 可错)。"""
    pred_frames = [{1: [0, 0, 10, 10]}]
    gt_frames = [{}]
    m = mota_metrics(pred_frames, gt_frames, iou_thr=0.5)
    assert m["mota"] == 1.0
    assert m["fp"] == 1  # 但 FP 仍记录
    assert m["n_gt"] == 0
