import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.temporal_fusion import (
    fuse_light, intervals_from_flags, fuse_occupancy, interval_intersect, tag_evidence,
)
from redlight.pipeline.intermediate_state import make_light_segment


def _obs(states, dt=0.125):
    return [(i * dt, s, 0.9) for i, s in enumerate(states)]


def _states(segs):
    return [s["state"] for s in segs]


def test_stable_green():
    segs = fuse_light(_obs(["green"] * 12), window=8, hysteresis=0.68, flicker_toggle=4)
    assert _states(segs) == ["green"]


def test_stable_red():
    segs = fuse_light(_obs(["red"] * 12), window=8, hysteresis=0.68, flicker_toggle=4)
    assert _states(segs) == ["red"]


def test_green_then_red_two_segments():
    segs = fuse_light(_obs(["green"] * 10 + ["red"] * 10), window=6, hysteresis=0.68, flicker_toggle=4)
    assert _states(segs) == ["green", "red"]


def test_flashing_detected():
    segs = fuse_light(_obs(["green", "red"] * 8), window=8, hysteresis=0.68, flicker_toggle=4)
    assert "flashing" in _states(segs)


def test_single_unknown_does_not_break_segment():
    segs = fuse_light(_obs(["green"] * 5 + [None] + ["green"] * 6),
                      window=6, hysteresis=0.68, flicker_toggle=4, unknown_hold=4)
    assert _states(segs) == ["green"]


def test_sustained_unknown_opens_new_segment():
    segs = fuse_light(_obs(["green"] * 6 + [None] * 14), window=6, hysteresis=0.68,
                      flicker_toggle=4, unknown_hold=4)
    assert _states(segs) == ["green", "unknown"]


def test_intermittent_green_not_flashing():
    segs = fuse_light(_obs(["green", "green", "off", "green", "green"] * 3),
                      window=8, hysteresis=0.68, flicker_toggle=4)
    assert "flashing" not in _states(segs)
    assert "green" in _states(segs)


# ---- 时变区间聚合(② tracks/occupancy, 供③区间代数) ----
def test_intervals_from_flags_contiguous_runs():
    samples = [(0.0, True), (1.0, True), (2.0, False), (3.0, True)]
    assert intervals_from_flags(samples) == [[0.0, 1.0], [3.0, 3.0]]


def test_intervals_from_flags_all_false():
    assert intervals_from_flags([(0.0, False), (1.0, False)]) == []


def test_intervals_from_flags_all_true():
    assert intervals_from_flags([(0.0, True), (1.0, True), (2.0, True)]) == [[0.0, 2.0]]


def test_fuse_occupancy_discontinuous_crossing():
    # 越线-回退-再越线 -> 两段, 各带 max/avg
    samples = [(0.0, 0.0), (1.0, 0.3), (2.0, 0.5), (3.0, 0.0), (4.0, 0.4)]
    out = fuse_occupancy(samples, base_thr=0.0)
    assert len(out) == 2
    assert out[0] == {"start_s": 1.0, "end_s": 2.0, "max_overlap": 0.5, "avg_overlap": 0.4}
    assert out[1] == {"start_s": 4.0, "end_s": 4.0, "max_overlap": 0.4, "avg_overlap": 0.4}


def test_fuse_occupancy_base_threshold():
    # base_thr 过滤极小占道噪声
    samples = [(0.0, 0.02), (1.0, 0.3), (2.0, 0.3)]
    out = fuse_occupancy(samples, base_thr=0.05)
    assert len(out) == 1 and out[0]["start_s"] == 1.0


def test_interval_intersect_basic():
    assert interval_intersect([[0, 10]], [[5, 15]]) == [[5, 10]]
    assert interval_intersect([[0, 5], [10, 15]], [[3, 12]]) == [[3, 5], [10, 12]]
    assert interval_intersect([[0, 5]], [[6, 10]]) == []        # 不相交
    assert interval_intersect([[0, 5]], [[5, 10]]) == []        # 相切(零长)不算


# ---- tag_evidence: 给 unknown 段按遮挡打 evidence (恢复 review, D1) ----
def test_tag_evidence_visible_for_color():
    segs = [make_light_segment(0, 20, "green", 0.9), make_light_segment(20, 40, "red", 0.8)]
    out = tag_evidence(segs, [])
    assert out[0]["evidence"] == "visible" and out[1]["evidence"] == "visible"


def test_tag_evidence_unknown_occluded():
    segs = [make_light_segment(0, 20, "green", 0.9), make_light_segment(20, 50, "unknown", 0.3)]
    occ = [(t, False) for t in range(0, 20)] + [(t, True) for t in range(20, 50)]
    out = tag_evidence(segs, occ)
    assert out[1]["evidence"] == "occluded"


def test_tag_evidence_unknown_not_occluded_stays_none():
    segs = [make_light_segment(20, 50, "unknown", 0.3)]
    occ = [(t, False) for t in range(20, 50)]
    out = tag_evidence(segs, occ)
    assert out[0]["evidence"] is None
