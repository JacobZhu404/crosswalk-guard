import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.temporal_fusion import fuse_light


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
