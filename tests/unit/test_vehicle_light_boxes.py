import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.models.vehicle import extract_light_boxes


def test_extract_only_traffic_light():
    raw = [
        {"name": "car", "xyxy": [0, 0, 10, 10], "conf": 0.9},
        {"name": "traffic light", "xyxy": [5, 5, 9, 20], "conf": 0.6},
    ]
    boxes = extract_light_boxes(raw, conf_min=0.25)
    assert boxes == [(5, 5, 9, 20)]


def test_conf_filter():
    raw = [{"name": "traffic light", "xyxy": [1, 2, 3, 4], "conf": 0.1}]
    assert extract_light_boxes(raw, conf_min=0.25) == []
