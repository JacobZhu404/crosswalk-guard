import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.temporal_fusion import fuse_light, intervals_from_flags, fuse_occupancy


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
