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
