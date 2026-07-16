import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.evaluation.violation_eval import (
    overlap_seconds, plate_match, match_violation_events, aggregate, _union_length,
)


def _ev(start, end, plate=""):
    return {"start_ts": start, "end_ts": end, "plate": plate}


def _gt(start, end, plates=None):
    return {"start_s": start, "end_s": end, "plates": plates or []}


# ---- 基础工具 ----

def test_overlap_seconds():
    assert overlap_seconds((0, 10), (5, 20)) == 5
    assert overlap_seconds((0, 5), (5, 10)) == 0     # 相切=0
    assert overlap_seconds((0, 5), (10, 20)) == 0    # 无交


def test_union_length_merges_overlaps():
    assert _union_length([[0, 5], [3, 8], [10, 12]]) == 10.0  # (0-8)=8 + (10-12)=2


def test_plate_match():
    assert plate_match("京LNE560", ["京LNE560", "京N541E6"]) is True
    assert plate_match("京JL1300", ["京LNE560"]) is False
    assert plate_match("", ["京LNE560"]) is False
    assert plate_match("京LNE560", []) is False


# ---- 事件级匹配 ----

def test_perfect_match_one_tp():
    r = match_violation_events([_ev(21, 68)], [_gt(21, 68)])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert r["precision"] == r["recall"] == r["f1"] == 1.0


def test_missed_violation_is_fn():
    # 违章04 场景: GT 有违章, 无预测 -> FN, recall=0
    r = match_violation_events([], [_gt(42, 43.2)])
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 1)
    assert r["recall"] == 0.0


def test_event_in_nonviolation_is_fp():
    # 预测事件落在无 GT 违章处(如红灯段误报) -> FP
    r = match_violation_events([_ev(3, 10)], [_gt(21, 68)])
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)
    assert r["precision"] == 0.0


def test_partial_overlap_counts_tp_but_low_coverage():
    # 违章09 场景: GT 61s, 预测只 1.75s 且落在段内 -> 仍算检出(TP), 但覆盖率极低
    r = match_violation_events([_ev(16.43, 18.18)], [_gt(11, 72)])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert r["recall"] == 1.0
    assert r["mean_coverage"] < 0.05    # 1.75/61 ≈ 0.029


def test_min_overlap_threshold_filters_touch():
    # 重叠 0.3s < 默认 0.5s -> 不算匹配
    r = match_violation_events([_ev(67.7, 68.3)], [_gt(21, 68)], min_overlap_s=0.5)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1


def test_greedy_one_to_one_no_double_count():
    # 两个预测事件都压在同一个 GT 上 -> 只配 1 对, 另一个算 FP
    r = match_violation_events([_ev(21, 40), _ev(41, 60)], [_gt(21, 68)])
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 0


def test_plate_secondary_metric():
    r = match_violation_events([_ev(21, 68, plate="京LNE560")], [_gt(21, 68, ["京LNE560"])])
    assert r["plate_total"] == 1 and r["plate_hits"] == 1
    r2 = match_violation_events([_ev(21, 68, plate="京JL1300")], [_gt(21, 68, ["京LNE560"])])
    assert r2["plate_total"] == 1 and r2["plate_hits"] == 0


def test_aggregate_sums_across_videos():
    r1 = match_violation_events([_ev(21, 68)], [_gt(21, 68)])          # tp1
    r2 = match_violation_events([], [_gt(42, 43.2)])                    # fn1
    r3 = match_violation_events([_ev(3, 10)], [_gt(21, 68)])            # fp1 fn1
    agg = aggregate([r1, r2, r3])
    assert agg["tp"] == 1 and agg["fp"] == 1 and agg["fn"] == 2
    assert agg["recall"] == round(1 / 3, 3)
