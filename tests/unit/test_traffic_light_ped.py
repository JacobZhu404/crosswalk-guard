import os, sys, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.models.traffic_light import TrafficLightDetector


def _cfg(method):
    tl = types.SimpleNamespace(method=method, smoothing_window=8, sat_min=130,
                               value_floor=60, min_area_px=30, max_area_ratio=0.008,
                               max_aspect_ratio=3.5, color_s_min=22)
    return types.SimpleNamespace(traffic_light=tl)


class _StubClf:
    """按固定 label 返回状态, 便于测试编排。"""
    available = True
    def __init__(self, label): self._label = label
    def classify(self, roi): return (self._label, 0.9)


def test_ped_classifier_stable_green():
    det = TrafficLightDetector(_cfg("ped_classifier"), verbose=False)
    det.classifier = _StubClf("walk")            # 注入桩分类器
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[20:40, 100:120] = (0, 255, 0)          # 给个亮斑
    states = []
    for _ in range(8):
        states.append(det.detect(frame, yolo_light_boxes=[(100, 20, 120, 40)])["state"])
    assert states[-1] == "green"


def test_ped_classifier_falls_back_when_unavailable():
    det = TrafficLightDetector(_cfg("ped_classifier"), verbose=False)
    det.classifier = types.SimpleNamespace(available=False,
                                           classify=lambda r: ("off", 0.0))
    frame = np.zeros((200, 200, 3), np.uint8)
    out = det.detect(frame, yolo_light_boxes=[(100, 20, 120, 40)])
    assert out["state"] in ("unknown", "red", "green", "flashing")  # 不崩, 走 color 路径
