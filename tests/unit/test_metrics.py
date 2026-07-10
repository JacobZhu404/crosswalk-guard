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

from redlight.evaluation.metrics import (
    precision_recall_f1, match_detections, detection_metrics, compute_map,
    levenshtein, edit_similarity, char_accuracy, plate_accuracy, province_accuracy,
    temporal_iou, event_metrics,
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
