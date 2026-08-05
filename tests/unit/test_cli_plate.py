"""_episode_plate: episode 跨 member_tracks 选最佳车牌(代表 track 无牌时兜底)。"""
import os
import sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

pytest.importorskip("cv2")  # cli 顶层 import cv2; 无 cv2 机器干净跳过
from redlight.app.cli import _episode_plate


def _plates(**kw):
    # kw: tid(str不便, 用 int key 通过 dict) -> (text, weight)
    return {tid: {"text": t, "weight": w} for tid, (t, w) in kw.items()}


def test_picks_representative_plate_when_present():
    cp = {1: {"text": "京LNE560", "weight": 2.0}}
    ev = {"track_id": 1, "member_tracks": [1]}
    assert _episode_plate(cp, ev) == "京LNE560"


def test_falls_back_to_member_when_representative_has_no_plate():
    # 代表 track 3 无牌, 成员 track 2 有牌 -> 用成员的
    cp = {2: {"text": "京ABV3428", "weight": 1.5}}
    ev = {"track_id": 3, "member_tracks": [3, 2]}
    assert _episode_plate(cp, ev) == "京ABV3428"


def test_picks_highest_weight_across_members():
    cp = {1: {"text": "京A11111", "weight": 0.4},
          2: {"text": "京B22222", "weight": 1.9},
          3: {"text": "京C33333", "weight": 1.0}}
    ev = {"track_id": 1, "member_tracks": [1, 2, 3]}
    assert _episode_plate(cp, ev) == "京B22222"


def test_empty_when_no_member_has_plate():
    ev = {"track_id": 5, "member_tracks": [5, 6]}
    assert _episode_plate({}, ev) == ""


def test_ignores_empty_text_entries():
    cp = {1: {"text": "", "weight": 5.0}, 2: {"text": "京D44444", "weight": 0.1}}
    ev = {"track_id": 1, "member_tracks": [1, 2]}
    assert _episode_plate(cp, ev) == "京D44444"


# ================= P2 约束回填(2026-08-05, cc gate cf1d246) =================
def _consensus_with(records):
    from redlight.pipeline.plate_consensus import PlateConsensus
    pc = PlateConsensus(keep_history=1000)
    for tid, recs in records.items():
        for text, conf, ts, box in recs:
            pc.update(tid, text, conf, ts, box)
    return pc


def _tracks(stationary_map):
    # stationary_map: tid -> bool(窗口内是否静止) 生成简化轨迹(窗口 [4,10] 内 3 样本)
    out = {}
    for tid, sta in stationary_map.items():
        out[tid] = [{"ts": 5.0, "stationary": sta, "box": [0, 0, 200, 200], "overlap": 0.5},
                    {"ts": 7.0, "stationary": sta, "box": [0, 0, 200, 200], "overlap": 0.5},
                    {"ts": 9.0, "stationary": sta, "box": [0, 0, 200, 200], "overlap": 0.5}]
    return out


def test_p2_rejects_non_stationary_vehicle():
    # tid1 窗口内非静止(过路车) -> 其牌不参与回填(宁缺毋滥)
    pc = _consensus_with({1: [("京A11111", 1.0, 5.0, None)] * 6})
    ts = _tracks({1: False})
    ev = {"track_id": 1, "member_tracks": [1], "start_ts": 4.0, "end_ts": 10.0}
    assert _episode_plate(pc, ev, ts) == ""


def test_p2_rejects_low_global_frequency_hallucination():
    # 孤证幻觉牌(全局仅 2 帧)即使挂在违章车上也不回填
    pc = _consensus_with({1: [("京X99999", 1.0, 5.0, None), ("京X99999", 1.0, 6.0, None)]})
    ts = _tracks({1: True})
    ev = {"track_id": 1, "member_tracks": [1], "start_ts": 4.0, "end_ts": 10.0}
    assert _episode_plate(pc, ev, ts) == ""


def test_p2_backfills_stationary_vehicle_plate():
    # 违章车(静止)的高频牌 -> 回填
    pc = _consensus_with({1: [("京LNE560", 1.0, 5.0, None)] * 6})
    ts = _tracks({1: True})
    ev = {"track_id": 1, "member_tracks": [1], "start_ts": 4.0, "end_ts": 10.0}
    assert _episode_plate(pc, ev, ts) == "京LNE560"
