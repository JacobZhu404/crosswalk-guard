import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.intermediate_state import make_light_segment, merge_adjacent_segments, LIGHT_STATES


def test_light_states_enum():
    assert set(LIGHT_STATES) == {"green", "red", "flashing", "unknown"}


def test_make_light_segment_fields():
    s = make_light_segment(0.0, 5.0, "green", 0.9, evidence="visible")
    assert s == {"start_s": 0.0, "end_s": 5.0, "state": "green", "conf": 0.9, "evidence": "visible"}


def test_merge_adjacent_same_state():
    segs = [make_light_segment(0, 2, "green", 0.8), make_light_segment(2, 5, "green", 0.9),
            make_light_segment(5, 7, "red", 0.7)]
    merged = merge_adjacent_segments(segs)
    assert len(merged) == 2
    assert merged[0] == {"start_s": 0, "end_s": 5, "state": "green", "conf": 0.9, "evidence": None}
    assert merged[1]["state"] == "red"
