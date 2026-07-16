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


def _states_multi(specs, stationary=True):
    """多 track 同帧: specs = {tid: box}。"""
    return {tid: {"active": True, "stationary": stationary, "box": list(box),
                  "cls": "car", "conf": 0.9} for tid, box in specs.items()}


def test_batch_fragmented_tracks_merge_to_one_episode():
    """碎片化: 同一违章窗内两个 track_id(物理同车被切开)都绿灯+静止+压线,
    应合并为**一个** confirmed episode(输出粒度=违章时间窗, 对齐 events.csv GT),
    而非各自成事件造成 FP。两 track 都记入 member_tracks。"""
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    for i in range(16):
        eng.accumulate(
            _states_multi({1: (120, 100, 220, 200), 2: (122, 101, 222, 201)}),
            _mask(False), {"obs": "green", "conf": 0.9}, i * 0.5)
    conf = [e for e in eng.decide() if e["status"] == "confirmed"]
    assert len(conf) == 1, f"跨 track 重叠事件应合并为1个, 实得 {len(conf)}"
    assert set(conf[0]["member_tracks"]) == {1, 2}


# ---- _dedup 全局时序合并(直接测被改单元, 不走 fuse 平滑) ----

def _raw(tid, s, e, status="confirmed", ov=0.5, light="green"):
    return {"track_id": tid, "status": status, "start_s": s, "end_s": e,
            "light_state": light, "max_overlap": ov}


def test_dedup_merges_overlapping_cross_track():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)  # gap=5s 默认
    out = eng._dedup([_raw(1, 91.5, 110.9), _raw(2, 91.5, 97.0),
                      _raw(3, 94.0, 109.9), _raw(4, 110.6, 121.2)])
    assert len(out) == 1
    assert out[0]["start_s"] == 91.5 and out[0]["end_s"] == 121.2
    assert set(out[0]["member_tracks"]) == {1, 2, 3, 4}


def test_dedup_keeps_far_apart_events():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    # 间隔 >gap(5s): 8s 结束, 下一个 20s 开始 -> 不合并
    out = eng._dedup([_raw(1, 0.0, 8.0), _raw(2, 20.0, 30.0)])
    assert len(out) == 2


def test_dedup_representative_track_is_max_overlap():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    out = eng._dedup([_raw(1, 10, 20, ov=0.3), _raw(2, 11, 25, ov=0.8)])
    assert len(out) == 1
    assert out[0]["track_id"] == 2                 # 代表 track = 压线比例最大者(供车牌/证据)
    assert out[0]["max_overlap"] == 0.8


def test_dedup_review_priority_on_merge():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    out = eng._dedup([_raw(1, 10, 20, status="confirmed"),
                      _raw(2, 12, 22, status="review", light="unknown")])
    assert len(out) == 1 and out[0]["status"] == "review"   # 安全侧: review 优先
