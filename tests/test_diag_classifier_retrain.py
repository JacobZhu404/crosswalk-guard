"""Phase B gate 诊断 TDD: 探针窗口解析 / 四关阈值判定函数单测。

跑法(项目 venv):
  PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_diag_classifier_retrain.py -q
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diag_classifier_retrain import (
    parse_probe_window, gate_neg_off_ratio, gate_true_green_recall, gate_probe_window,
)

NEG = {"违章01", "违章10"}


def test_parse_probe_window():
    v, t0, t1 = parse_probe_window("违章04:42.0:43.2")
    assert v == "违章04" and t0 == 42.0 and t1 == 43.2


def test_parse_probe_window_bad():
    try:
        parse_probe_window("违章04:42.0")
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_gate_neg_off_ratio_pass():
    videos = [{"video": "违章01", "crops": 100, "off": 80},
              {"video": "违章10", "crops": 100, "off": 90}]
    ok, val = gate_neg_off_ratio(videos, NEG, 0.75)
    assert ok and abs(val - 0.85) < 1e-9


def test_gate_neg_off_ratio_fail():
    videos = [{"video": "违章01", "crops": 100, "off": 50}]
    ok, val = gate_neg_off_ratio(videos, NEG, 0.75)
    assert (not ok) and abs(val - 0.5) < 1e-9


def test_gate_true_green_recall_fail_below():
    # 收率 0.89 < 0.90 -> 不过
    videos = [{"video": "违章07", "crops": 100, "walk": 80, "stand": 9, "off": 11}]
    ok, val = gate_true_green_recall(videos, {"违章07"}, 0.90)
    assert (not ok) and abs(val - 0.89) < 1e-9


def test_gate_true_green_recall_pass():
    videos = [{"video": "违章07", "crops": 100, "walk": 88, "stand": 2, "off": 10}]
    ok, val = gate_true_green_recall(videos, {"违章07"}, 0.90)
    assert ok and abs(val - 0.90) < 1e-9


def test_gate_probe_window_pass():
    ok, val = gate_probe_window(walk=6, off=2, other=2, thr=0.5)
    assert ok and abs(val - 0.6) < 1e-9


def test_gate_probe_window_fail():
    ok, val = gate_probe_window(walk=3, off=8, other=1, thr=0.5)
    assert (not ok) and abs(val - 0.25) < 1e-9
