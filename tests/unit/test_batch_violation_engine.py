"""BatchViolationEngine 端到端(②③ 接线): accumulate -> decide。纯 numpy, 无需 cv2。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.violation_engine import BatchViolationEngine


def _states(tid, box, stationary=True):
    return {tid: {"active": True, "stationary": stationary, "box": list(box),
                  "cls": "car", "conf": 0.9}}


def _mask(occluded=False):
    m = np.zeros((400, 400), np.uint8)
    if occluded:
        m[100:400, 100:300] = 255   # 触底边 -> 遮挡
    else:
        m[100:200, 100:300] = 255
    return m


def test_batch_green_stationary_confirmed():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    for i in range(16):
        eng.accumulate(_states(1, (120, 100, 220, 200)), _mask(False),
                       {"obs": "green", "conf": 0.9}, i * 0.5)
    evs = eng.decide()
    conf = [e for e in evs if e["status"] == "confirmed"]
    assert len(conf) == 1 and conf[0]["track_id"] == 1 and conf[0]["light_state"] == "green"


def test_batch_unknown_occluded_review():
    """灯 unknown + 斑马线遮挡 + 车静止压线 -> review (D1, evidence 打标恢复)。"""
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    for i in range(16):
        eng.accumulate(_states(1, (120, 100, 220, 400)), _mask(occluded=True),
                       {"obs": "off", "conf": 0.0}, i * 0.5)   # obs off -> unknown 段
    evs = eng.decide()
    assert [e for e in evs if e["status"] == "review"]          # review 复活
    assert not [e for e in evs if e["status"] == "confirmed"]   # 未知不确认


def test_batch_red_no_event():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    for i in range(16):
        eng.accumulate(_states(1, (120, 100, 220, 200)), _mask(False),
                       {"obs": "red", "conf": 0.9}, i * 0.5)
    assert eng.decide() == []
