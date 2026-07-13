import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.decision import decide_violations
from redlight.pipeline.intermediate_state import (
    make_light_segment, make_track, make_occupancy_interval,
)


def _state(segs, tracks):
    return {"light_segments": segs, "tracks": tracks}


def _occ(mx):
    return [make_occupancy_interval(22, 68, mx, mx - 0.1)]


def test_green_stationary_occupied_confirmed():
    segs = [make_light_segment(0, 20, "red", 0.9, "visible"),
            make_light_segment(20, 80, "green", 0.9, "visible")]
    tr = [make_track(3, vehicle_class="car", stationary_intervals=[[25, 70]],
                     occupancy_intervals=_occ(0.42))]
    evs = decide_violations(_state(segs, tr), overlap_thr=0.2, min_duration_s=3)
    conf = [e for e in evs if e["status"] == "confirmed"]
    assert len(conf) == 1
    assert conf[0]["track_id"] == 3 and conf[0]["light_state"] == "green"
    assert conf[0]["start_s"] == 25 and conf[0]["end_s"] == 68   # 交集 [25,68]
    assert conf[0]["max_overlap"] == 0.42


def test_red_no_violation():
    segs = [make_light_segment(0, 80, "red", 0.9, "visible")]
    tr = [make_track(3, stationary_intervals=[[25, 70]], occupancy_intervals=_occ(0.42))]
    assert decide_violations(_state(segs, tr), 0.2, 3) == []


def test_overlap_below_threshold_none():
    segs = [make_light_segment(20, 80, "green", 0.9, "visible")]
    tr = [make_track(3, stationary_intervals=[[25, 70]], occupancy_intervals=_occ(0.15))]
    assert decide_violations(_state(segs, tr), 0.2, 3) == []


def test_short_intersection_below_duration_none():
    segs = [make_light_segment(20, 26, "green", 0.9, "visible")]  # go 仅 20-26
    tr = [make_track(3, stationary_intervals=[[25, 70]], occupancy_intervals=_occ(0.42))]
    # 交集 [25,26] 时长 1 < 3 -> 不确认
    conf = [e for e in decide_violations(_state(segs, tr), 0.2, 3) if e["status"] == "confirmed"]
    assert conf == []


def test_flashing_also_confirms():
    segs = [make_light_segment(20, 80, "flashing", 0.7, "visible")]
    tr = [make_track(3, stationary_intervals=[[25, 70]], occupancy_intervals=_occ(0.42))]
    conf = [e for e in decide_violations(_state(segs, tr), 0.2, 3) if e["status"] == "confirmed"]
    assert len(conf) == 1 and conf[0]["light_state"] == "flashing"


def test_unknown_occluded_review():
    segs = [make_light_segment(20, 80, "unknown", 0.5, "occluded")]
    tr = [make_track(3, stationary_intervals=[[25, 70]], occupancy_intervals=_occ(0.42))]
    evs = decide_violations(_state(segs, tr), 0.2, 3)
    assert [e for e in evs if e["status"] == "review"]          # 遮挡+未知 -> review (D1)
    assert not [e for e in evs if e["status"] == "confirmed"]


def test_moving_car_not_confirmed():
    # 绿灯+压线但不静止(无 stationary_intervals) -> 无违规
    segs = [make_light_segment(20, 80, "green", 0.9, "visible")]
    tr = [make_track(3, stationary_intervals=[], occupancy_intervals=_occ(0.42))]
    assert decide_violations(_state(segs, tr), 0.2, 3) == []
