import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.intermediate_state import (
    make_light_segment, merge_adjacent_segments, LIGHT_STATES,
    make_occupancy_interval, make_track,
)


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


def test_make_occupancy_interval_fields():
    o = make_occupancy_interval(21, 68, 0.42, 0.31)
    assert o == {"start_s": 21, "end_s": 68, "max_overlap": 0.42, "avg_overlap": 0.31}


def test_make_track_fields_and_defaults():
    t = make_track(3, vehicle_class="car")
    assert t["track_id"] == 3 and t["vehicle_class"] == "car"
    assert t["plate"] is None
    assert t["stationary_intervals"] == [] and t["occupancy_intervals"] == []
    t2 = make_track(4, plate={"text": "京LNE560", "conf": 0.9},
                    stationary_intervals=[[21, 68]],
                    occupancy_intervals=[make_occupancy_interval(21, 68, 0.42, 0.31)])
    assert t2["plate"]["text"] == "京LNE560"
    assert t2["occupancy_intervals"][0]["max_overlap"] == 0.42
