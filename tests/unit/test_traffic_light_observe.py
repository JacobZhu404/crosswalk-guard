import os, sys, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.models.traffic_light import TrafficLightDetector


def _cfg():
    tl = types.SimpleNamespace(method="color", smoothing_window=8, sat_min=130,
                               value_floor=60, min_area_px=30, max_area_ratio=0.008,
                               max_aspect_ratio=3.5, color_s_min=22)
    return types.SimpleNamespace(traffic_light=tl)


def test_observe_returns_per_frame_obs():
    det = TrafficLightDetector(_cfg(), verbose=False)
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[20:35, 100:115] = (0, 255, 0)
    out = det.observe(frame)
    assert set(out) >= {"obs", "conf", "candidates"}
    assert out["obs"] in ("green", "red", "off", None)


def test_observe_is_stateless_across_calls():
    det = TrafficLightDetector(_cfg(), verbose=False)
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[20:35, 100:115] = (0, 255, 0)
    a = det.observe(frame)["obs"]
    b = det.observe(frame)["obs"]
    assert a == b
