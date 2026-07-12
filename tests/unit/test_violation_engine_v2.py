"""ViolationEngineV2 单元测试 (语义反转版: 绿灯/闪烁=违规)。

全部手工构造 track_states / mask / light, 不依赖模型。
"""
import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.pipeline.violation_engine import ViolationEngineV2, OCCLUSION_MIN_AREA_RATIO
from redlight.pipeline.tracker import SENSITIVITY_PRESETS


def _make_mask(occluded=False):
    mask = np.zeros((400, 400), dtype=np.uint8)
    if occluded:
        mask[0:3, 0:3] = 255   # 极小面积 -> 视为遮挡
    else:
        mask[100:200, 100:300] = 255
    return mask


def _state(tid=1, stationary=True, box=(100, 100, 200, 200)):
    return {tid: {"active": True, "stationary": stationary, "box": list(box),
                  "cls": "car", "conf": 0.9}}


def test_green_confirmed():
    eng = ViolationEngineV2("balanced")
    mask = _make_mask()
    evs = []
    for i in range(6):
        evs += eng.evaluate(_state(1, True), mask, "green", i * 0.5)
    confirmed = [e for e in evs if e["status"] == "confirmed"]
    assert len(confirmed) == 1
    assert confirmed[0]["track_id"] == 1
    assert confirmed[0]["light_state"] == "green"


def test_unknown_forward_fill_green():
    """未知灯短时向前填充: 前后为绿灯, 中间 brief unknown 仍按绿灯判违规 (08中段)。"""
    eng = ViolationEngineV2("balanced", fill_gap_sec=2.0)
    mask = _make_mask()
    evs = []
    seq = ["green", "green", "unknown", "unknown", "green", "green"]
    for i, ls in enumerate(seq):
        evs += eng.evaluate(_state(1, True), mask, ls, i * 0.5)  # 0.5s 间隔 < 2s
    confirmed = [e for e in evs if e["status"] == "confirmed"]
    assert len(confirmed) == 1, f"forward-fill 应触发1次违规, got {len(confirmed)}"


def test_unknown_long_gap_no_fill():
    """超过填充窗口的长时间 unknown 不向前填充 -> 不判违规。"""
    eng = ViolationEngineV2("balanced", fill_gap_sec=2.0)
    mask = _make_mask()
    evs = []
    for t in (0.0, 0.5, 1.0, 1.5, 2.0):   # 足够前导绿帧触发首次违规
        evs += eng.evaluate(_state(1, True), mask, "green", t)
    evs += eng.evaluate(_state(1, True), mask, "unknown", 6.0)  # 间隔4s > 2s
    evs += eng.evaluate(_state(1, True), mask, "unknown", 6.5)
    confirmed = [e for e in evs if e["status"] == "confirmed"]
    assert len(confirmed) == 1, f"长间隔unknown不应填充, got {len(confirmed)}"


def test_flashing_confirmed():
    eng = ViolationEngineV2("balanced")
    mask = _make_mask()
    evs = []
    for i in range(6):
        evs += eng.evaluate(_state(1, True), mask, "flashing", i * 0.5)
    confirmed = [e for e in evs if e["status"] == "confirmed"]
    assert len(confirmed) == 1
    assert confirmed[0]["light_state"] == "flashing"


def test_red_not_violation():
    """🔴红灯 = 车辆可通行 -> 静止压线也不算违规 (语义反转核心)。"""
    eng = ViolationEngineV2("balanced")
    mask = _make_mask()
    evs = []
    for i in range(6):
        evs += eng.evaluate(_state(1, True), mask, "red", i * 0.5)
    assert len(evs) == 0


def test_missing_overlap_no_event():
    eng = ViolationEngineV2("balanced")
    mask = np.zeros((400, 400), dtype=np.uint8)
    evs = []
    for i in range(6):
        evs += eng.evaluate(_state(1, True), mask, "green", i * 0.5)
    assert len(evs) == 0


def test_not_stationary_no_event():
    eng = ViolationEngineV2("balanced")
    mask = _make_mask()
    evs = []
    for i in range(6):
        evs += eng.evaluate(_state(1, False), mask, "green", i * 0.5)
    assert len(evs) == 0


def test_unknown_occluded_to_review():
    """unknown + 斑马线被遮挡(触及画面边界, 看不全) -> review (Q2)。"""
    eng = ViolationEngineV2("balanced", unknown_to_review=True)
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[100:400, 100:300] = 255  # 触及底边(斑马线被画面下沿截断, 看不全) 且 覆盖车体 -> 压线且遮挡
    evs = []
    for i in range(6):
        evs += eng.evaluate(_state(1, True), mask, "unknown", i * 0.5)
    reviews = [e for e in evs if e["status"] == "review"]
    assert len(reviews) == 1


def test_unknown_not_occluded_no_event():
    """unknown + 斑马线可见 -> 默认不判违规, 连 review 都不发。"""
    eng = ViolationEngineV2("balanced", unknown_to_review=True)
    mask = _make_mask(occluded=False)
    evs = []
    for i in range(6):
        evs += eng.evaluate(_state(1, True), mask, "unknown", i * 0.5)
    assert len(evs) == 0


def test_unknown_no_review_when_disabled():
    eng = ViolationEngineV2("balanced", unknown_to_review=False)
    mask = _make_mask(occluded=True)
    evs = []
    for i in range(6):
        evs += eng.evaluate(_state(1, True), mask, "unknown", i * 0.5)
    assert len(evs) == 0


def test_gap_enforcement():
    eng = ViolationEngineV2("balanced", min_event_gap_sec=5)
    mask = _make_mask()
    evs = []
    for i in range(6):
        evs += eng.evaluate(_state(1, True), mask, "green", i * 0.5)
    for i in range(6, 12):
        evs += eng.evaluate(_state(1, True), mask, "green", i * 0.5)
    confirmed = [e for e in evs if e["status"] == "confirmed"]
    assert len(confirmed) == 1  # 间隔 3s < 5s gap, 不发第二次


def test_preset_selects_overlap():
    strict = ViolationEngineV2("strict")
    loose = ViolationEngineV2("loose")
    assert strict.overlap == SENSITIVITY_PRESETS["strict"]["overlap"]
    assert loose.overlap == SENSITIVITY_PRESETS["loose"]["overlap"]
    assert strict.overlap > loose.overlap


def test_loose_catches_partial_overlap():
    """loose(overlap=0.15) 能抓部分压线, strict(0.30) 抓不到。"""
    eng_loose = ViolationEngineV2("loose")
    eng_strict = ViolationEngineV2("strict")
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[100:200, 100:300] = 255
    box = (280, 100, 360, 200)  # 重叠 2000 / box 8000 = 0.25, 落在 (0.15, 0.30)
    evs_l, evs_s = [], []
    for i in range(6):
        ts = i * 0.5
        evs_l += eng_loose.evaluate(_state(1, True, box), mask, "green", ts)
        evs_s += eng_strict.evaluate(_state(1, True, box), mask, "green", ts)
    assert len([e for e in evs_l if e["status"] == "confirmed"]) == 1
    assert len([e for e in evs_s if e["status"] == "confirmed"]) == 0


def test_occlusion_helper():
    assert ViolationEngineV2._is_occluded(None) is True
    assert ViolationEngineV2._is_occluded(_make_mask(occluded=True)) is True
    assert ViolationEngineV2._is_occluded(_make_mask(occluded=False)) is False
