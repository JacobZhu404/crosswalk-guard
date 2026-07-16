"""端到端违章评测: 负例纳入 + FP 分类(真误报/碎片) 单元测试。

对应 docs/plans/2026-07-16-wb-plan-v4-fix-eval-measurement.md Part A。
TDD 先行: 4 个用例覆盖 cc 裁定的判据。纯函数, 不依赖 cv2/视频。

运行:
    cd <project> && .venv/bin/python -m pytest tests/unit/test_violation_eval.py -q
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.violation_eval import (
    match_violation_events, aggregate,
    classify_false_positives,
)


def _ev(s, e, light="green"):
    return {"start_ts": s, "end_ts": e, "plate": "", "light_state": light}


def _gt(s, e):
    return {"start_s": s, "end_s": e, "plates": []}


def test_negative_video_confirmed_is_true_fp():
    """用例1: 负例视频, 1 个 confirmed -> fp=1, neg_true_fp, 头条 P=0.0 (A1+A2)。"""
    gt_v = []  # 负例无 GT 违章段
    conf = [_ev(48.0, 60.0, light="green")]
    r = match_violation_events(conf, gt_v, min_overlap_s=0.5)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 0
    assert r["precision"] == 0.0
    cls = classify_false_positives(r, conf, gt_v, is_negative=True, min_overlap_s=0.5)
    assert cls["neg_true_fp"] == [0]
    assert cls["neg_count"] == 1 and cls["oow_count"] == 0 and cls["fragment_count"] == 0
    assert cls["detail"][0]["category"] == "neg_true_fp"


def test_fragment_within_gt_window():
    """用例2: 正例, GT[21,68]; confirmed [21,68](TP) + [30,34](重叠>=0.5 未配对) -> fragment。"""
    gt_v = [_gt(21.0, 68.0)]
    conf = [_ev(21.0, 68.0), _ev(30.0, 34.0)]
    r = match_violation_events(conf, gt_v, min_overlap_s=0.5)
    assert r["tp"] == 1 and r["fp"] == 1
    cls = classify_false_positives(r, conf, gt_v, is_negative=False, min_overlap_s=0.5)
    assert cls["fragment"] == [1]
    assert cls["fragment_count"] == 1 and cls["oow_count"] == 0 and cls["neg_count"] == 0
    d1 = next(d for d in cls["detail"] if d["idx"] == 1)
    assert d1["category"] == "fragment"
    assert d1["gt_window"] == "[21-68]"


def test_out_of_window_true_fp_zero_overlap():
    """用例3: 正例, GT[11,72]; confirmed 含合法 TP[11,72] +  spurious[94,96] 零重叠 -> oow_true_fp。"""
    gt_v = [_gt(11.0, 72.0)]
    conf = [_ev(11.0, 72.0), _ev(94.0, 96.0)]
    r = match_violation_events(conf, gt_v, min_overlap_s=0.5)
    assert r["tp"] == 1 and r["fp"] == 1
    cls = classify_false_positives(r, conf, gt_v, is_negative=False, min_overlap_s=0.5)
    assert cls["oow_true_fp"] == [1]
    assert cls["oow_count"] == 1 and cls["fragment_count"] == 0
    d1 = next(d for d in cls["detail"] if d["idx"] == 1)
    assert d1["category"] == "oow_true_fp"
    assert d1["overlap_gt_window_s"] == 0.0


def test_aggregate_breakdown_across_videos():
    """用例4: 聚合上述三类 -> 头条 + true_fp_total + fragment_total + P(仅真误报)。"""
    # 负例
    c1 = [_ev(48.0, 60.0, light="green")]
    r1 = match_violation_events(c1, [], min_overlap_s=0.5)
    k1 = classify_false_positives(r1, c1, [], is_negative=True, min_overlap_s=0.5)
    r1.update({"neg_count": k1["neg_count"], "oow_count": k1["oow_count"], "fragment_count": k1["fragment_count"]})
    # 碎片
    c2 = [_ev(21.0, 68.0), _ev(30.0, 34.0)]
    g2 = [_gt(21.0, 68.0)]
    r2 = match_violation_events(c2, g2, min_overlap_s=0.5)
    k2 = classify_false_positives(r2, c2, g2, is_negative=False, min_overlap_s=0.5)
    r2.update({"neg_count": k2["neg_count"], "oow_count": k2["oow_count"], "fragment_count": k2["fragment_count"]})
    # 窗外真误报(同时含合法 TP)
    c3 = [_ev(11.0, 72.0), _ev(94.0, 96.0)]
    g3 = [_gt(11.0, 72.0)]
    r3 = match_violation_events(c3, g3, min_overlap_s=0.5)
    k3 = classify_false_positives(r3, c3, g3, is_negative=False, min_overlap_s=0.5)
    r3.update({"neg_count": k3["neg_count"], "oow_count": k3["oow_count"], "fragment_count": k3["fragment_count"]})

    agg = aggregate([r1, r2, r3])
    assert agg["tp"] == 2 and agg["fp"] == 3 and agg["fn"] == 0
    assert agg["true_fp_total"] == 2   # neg 1 + oow 1
    assert agg["fragment_total"] == 1
    assert abs(agg["p_only_true_fp"] - 2 / (2 + 2)) < 1e-6  # TP/(TP+真误报)
    # 头条口径未被 classify 改变
    assert abs(agg["precision"] - 2 / (2 + 3)) < 1e-6
