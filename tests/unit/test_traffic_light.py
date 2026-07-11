"""TrafficLightDetector v2 (strengthened) 单元测试。

用合成帧验证: 稳定红/绿, 闪烁绿->flashing, 时隐时现红块->unknown(尾灯拒绝), 无信号->unknown。
"""
import sys
import os
import types
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.models.traffic_light import TrafficLightDetector


def _cfg(window=6):
    return types.SimpleNamespace(
        traffic_light=types.SimpleNamespace(
            smoothing_window=window, min_area_ratio=0.0006, max_aspect_ratio=3.0,
            search_band=(0.0, 0.65), flicker_toggle_count=2,
            color_min_pixels=60, persistence_min_ratio=0.5),
    )


def _frame(h=480, w=640, color=None, box=None):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    if color is not None and box is not None:
        x1, y1, x2, y2 = box
        f[y1:y2, x1:x2] = color
    return f


def test_stable_green():
    det = TrafficLightDetector(_cfg(), verbose=False)
    green = (0, 255, 0)
    box = (300, 40, 340, 80)  # 上中部
    for _ in range(6):
        det.detect(_frame(color=green, box=box))
    assert det.detect(_frame(color=green, box=box))["state"] == "green"


def test_stable_red():
    det = TrafficLightDetector(_cfg(), verbose=False)
    red = (0, 0, 255)
    box = (300, 40, 340, 80)
    for _ in range(6):
        det.detect(_frame(color=red, box=box))
    assert det.detect(_frame(color=red, box=box))["state"] == "red"


def test_flickering_green_is_flashing():
    det = TrafficLightDetector(_cfg(), verbose=False)
    green = (0, 255, 0)
    box = (300, 40, 340, 80)
    for i in range(6):
        if i % 2 == 0:
            det.detect(_frame(color=green, box=box))
        else:
            det.detect(_frame())  # 灭
    assert det.detect(_frame(color=green, box=box))["state"] == "flashing"


def test_intermittent_red_rejected():
    """时隐时现的红块(尾灯特征) -> unknown, 不误判红灯放过违规 (Q3)。"""
    det = TrafficLightDetector(_cfg(), verbose=False)
    red = (0, 0, 255)
    box = (300, 40, 340, 80)
    for i in range(6):
        if i % 3 == 0:
            det.detect(_frame(color=red, box=box))
        else:
            det.detect(_frame())
    assert det.detect(_frame())["state"] == "unknown"


def test_no_signal_unknown():
    det = TrafficLightDetector(_cfg(), verbose=False)
    for _ in range(6):
        det.detect(_frame())
    assert det.detect(_frame())["state"] == "unknown"
