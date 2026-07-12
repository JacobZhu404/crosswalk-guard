"""TrafficLightDetector v6 (自适应亮斑 + 信号位置跟踪器) 单元测试。

用合成帧验证: 稳定红/绿, 闪烁(红绿交替)->flashing, 暗淡绿灯可检, 琥珀归红,
移动红块(车灯)->unknown(无持久轨迹), 满屏瞬态反光->unknown, 无信号->unknown。
"""
import sys
import os
import types
import random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.models.traffic_light import TrafficLightDetector


def _cfg(window=24):
    return types.SimpleNamespace(
        traffic_light=types.SimpleNamespace(
            smoothing_window=window,
            value_floor=60, sat_min=130,
            min_area_px=20, max_area_ratio=0.008, max_aspect_ratio=3.5,
            color_s_min=22,
            match_radius_ratio=0.06, min_persist_frames=5,
            track_persist_min=0.08, signal_cy_cutoff=0.6, flicker_toggle_count=4,
            lit_emission_floor=25.0, lit_frac_min=0.30,
        ),
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
    """真实闪烁清空相位: 绿/红交替出现(都出现过)且绿亮灭多次跳变 -> flashing。"""
    det = TrafficLightDetector(_cfg(), verbose=False)
    green = (0, 255, 0)
    red = (0, 0, 255)
    box = (300, 40, 340, 80)
    for i in range(16):
        if i % 2 == 0:
            det.detect(_frame(color=green, box=box))
        else:
            det.detect(_frame(color=red, box=box))
    assert det.detect(_frame(color=green, box=box))["state"] == "flashing"


def test_intermittent_green_only_not_flashing():
    """仅绿闪烁、无红 -> 不是清空相位闪烁(可能是噪声), 退为 green, 不假报 flashing。"""
    det = TrafficLightDetector(_cfg(), verbose=False)
    green = (0, 255, 0)
    box = (300, 40, 340, 80)
    for i in range(6):
        if i % 2 == 0:
            det.detect(_frame(color=green, box=box))
        else:
            det.detect(_frame())
    st = det.detect(_frame(color=green, box=box))["state"]
    assert st != "flashing"


def test_moving_red_rejected():
    """移动红块(车灯随车移动, 每帧不同位置) -> 无持久轨迹 -> unknown, 不误判红灯 (Q3)。"""
    det = TrafficLightDetector(_cfg(), verbose=False)
    red = (0, 0, 255)
    positions = [(100, 40, 140, 80), (300, 40, 340, 80), (500, 40, 540, 80),
                 (200, 200, 240, 240), (600, 300, 640, 340), (80, 350, 120, 390)]
    for i in range(6):
        det.detect(_frame(color=red, box=positions[i % len(positions)]))
    assert det.detect(_frame())["state"] == "unknown"


def test_dim_green_detected():
    """暗淡绿灯(低亮度, V=110 远低于旧固定阈值200)也能被自适应阈值检出 -> green。"""
    det = TrafficLightDetector(_cfg(), verbose=False)
    dim_green = (0, 110, 0)
    box = (300, 40, 340, 80)
    for _ in range(8):
        det.detect(_frame(color=dim_green, box=box))
    assert det.detect(_frame(color=dim_green, box=box))["state"] == "green"


def test_amber_treated_as_red():
    """琥珀/橙色信号灯(车载信号常见)保守归为 red 侧。"""
    det = TrafficLightDetector(_cfg(), verbose=False)
    amber = (0, 120, 255)  # BGR -> 纯橙 (H~28)
    box = (300, 40, 340, 80)
    for _ in range(8):
        det.detect(_frame(color=amber, box=box))
    assert det.detect(_frame(color=amber, box=box))["state"] == "red"


def test_transient_reflections_unknown():
    """满屏瞬态亮斑(如 01 的反光)但无持久信号轨迹 -> unknown。"""
    det = TrafficLightDetector(_cfg(), verbose=False)
    random.seed(0)
    for _ in range(10):
        x = random.randint(0, 600); y = random.randint(0, 440)
        det.detect(_frame(color=(200, 200, 200), box=(x, y, x + 30, y + 30)))
    assert det.detect(_frame())["state"] == "unknown"


def test_no_signal_unknown():
    det = TrafficLightDetector(_cfg(), verbose=False)
    for _ in range(6):
        det.detect(_frame())
    assert det.detect(_frame())["state"] == "unknown"


def test_low_saturation_green_rejected():
    """草地/绿漆: 大面积但饱和度低(S~100) -> 不应被判为绿灯(过检防护)。

    真实信号灯 LED 饱和度 >=200, 此处用 S~100 的"植物绿"验证 sat_min 门限生效。
    """
    det = TrafficLightDetector(_cfg(), verbose=False)
    # 草绿色 BGR: 在 HSV 下 S 约 75 (低于 sat_min=130, 真实 LED 在 200+)
    grass = (120, 170, 120)   # 低饱和淡绿(植物/绿漆)
    box = (50, 50, 590, 430)  # 大块, 模拟满屏草地
    for _ in range(8):
        det.detect(_frame(color=grass, box=box))
    assert det.detect(_frame(color=grass, box=box))["state"] == "unknown"


