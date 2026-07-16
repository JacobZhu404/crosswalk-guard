"""TrackStateManagerV2 单元测试 (修复 E7: 抖动/蠕行漏检)。

全部手工构造轨迹, 不依赖任何模型。
"""
import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.pipeline.tracker import TrackStateManagerV2, SENSITIVITY_PRESETS


def _det(tid, xyxy, cls="car", conf=0.9):
    return {"id": tid, "xyxy": list(xyxy), "cls": cls, "conf": conf}


def _feed(mgr, tid, boxes, dt=0.125):
    """按固定时间间隔 dt 连续喂入某 track 的框序列, 返回最后的状态。"""
    states = None
    for i, b in enumerate(boxes):
        ts = i * dt
        states = mgr.update([_det(tid, b)], ts)
    return states


def test_preset_attributes():
    for name, p in SENSITIVITY_PRESETS.items():
        m = TrackStateManagerV2(name)
        assert m.speed_thres == p["speed"]
    # strict 比 loose 更严
    assert SENSITIVITY_PRESETS["strict"]["speed"] < SENSITIVITY_PRESETS["loose"]["speed"]
    assert SENSITIVITY_PRESETS["strict"]["overlap"] > SENSITIVITY_PRESETS["loose"]["overlap"]


def test_stopped_car_becomes_stationary():
    # 真停车: 框完全不动 -> 足够样本后应判静止 (balanced)
    m = TrackStateManagerV2("balanced")
    boxes = [[100, 100, 200, 200] for _ in range(10)]
    states = _feed(m, 1, boxes)
    assert states[1]["stationary"] is True


def test_moving_car_not_stationary():
    # 持续移动: 每帧 +50px -> 速度远超阈值, 不应判静止
    m = TrackStateManagerV2("balanced")
    boxes = [[100 + i * 50, 100, 200 + i * 50, 200] for i in range(10)]
    states = _feed(m, 1, boxes)
    assert states[1]["stationary"] is False


def test_jittering_stop_is_stationary():
    """关键修复验证 (E7): 停车但跟踪抖动 ±2px, v1 会因首尾距离归零,
    V2 用瞬时速度+占比判据 -> 应判静止。"""
    m = TrackStateManagerV2("balanced")
    boxes = []
    # 真实跟踪抖动: 单轴小幅度波动 (等效位移 <=2.8px/帧 => 速度 <=22px/s < 30 阈值)
    jitter = [0, 1, -1, 0, 1, -1, 0, 1, -1, 0]
    for j in jitter:
        boxes.append([100 + j, 100 + j, 200 + j, 200 + j])
    states = _feed(m, 1, boxes)
    assert states[1]["stationary"] is True


def test_slow_creep_caught_by_balanced_not_strict():
    # 静止判定改用相邻帧框 IoU(抗 YOLO 框抖动, 修违章02真停车判非静止).
    # 缓慢蠕行 20px/s(框200宽, 每帧移2.5px): balanced IoU阈值0.70, 帧间IoU~0.97 -> 判静止.
    # 违章语义下压斑马线慢蠕行≈没让行, 判静止合理.
    dt = 0.125
    creep = 20 * dt  # 每帧位移 2.5px
    boxes = [[100 + i * creep, 100, 200 + i * creep, 200] for i in range(10)]

    mb = TrackStateManagerV2("balanced")
    mb_states = _feed(mb, 1, boxes)
    assert mb_states[1]["stationary"] is True, "balanced 应捕获 20px/s 蠕行(IoU高)"


def test_insufficient_samples_not_stationary():
    # 样本太少, 不应过早判静止
    m = TrackStateManagerV2("balanced")
    boxes = [[100, 100, 200, 200] for _ in range(3)]
    states = _feed(m, 1, boxes)
    assert states[1]["stationary"] is False


def test_multiple_tracks_independent():
    m = TrackStateManagerV2("balanced")
    stop = [[100, 100, 200, 200] for _ in range(10)]
    move = [[50 + i * 60, 50, 150 + i * 60, 150] for i in range(10)]
    for i in range(10):
        ts = i * 0.125
        m.update([_det(1, stop[i])], ts)
        m.update([_det(2, move[i])], ts)
    assert m.states[1]["stationary"] is True
    assert m.states[2]["stationary"] is False
