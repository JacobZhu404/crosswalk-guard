"""集成测试: Pipeline DAG 把真实 Tracker V2 + BatchViolationEngine 串起来 (②③ 接线)。

用轻量 mock 替代重模型 (vehicle/crosswalk/light/plate/viz),
但 tracker/engine 用真实实现, 验证节点接线与端到端违规判定。
"""
import os
import sys
import types

import numpy as np
import pytest

# dag/violation_engine 需真实 cv2; 无 cv2 环境整文件跳过。
# 严禁往 sys.modules 注入残缺 mock cv2 —— 会污染同进程后续测试(cvtColor/imencode/resize 丢失),
# 正是本轮 19 个测试挂的根因。importorskip 在 cv2 机上照常真跑, cv2-less 干净跳过, 零污染。
pytest.importorskip("cv2")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.pipeline.dag import PipelineDAG, build_default_dag
from redlight.pipeline.tracker import TrackStateManagerV2
from redlight.pipeline.violation_engine import BatchViolationEngine


class _Mock:
    def __init__(self, ret):
        self.ret = ret
        self.calls = 0
    def detect(self, frame, *a, **k):
        self.calls += 1
        return self.ret
    def observe(self, frame):
        # ②③ 接线: DAG light 节点现在同时调用 observe()
        obs = "off"
        if isinstance(self.ret, dict):
            obs = self.ret.get("state", "off")
        elif self.ret in ("green", "red"):
            obs = self.ret
        return {"obs": obs, "conf": 0.9, "candidates": []}
    def draw(self, *a, **k):
        return a[0]


def _minimal_cfg():
    return types.SimpleNamespace(
        inference=types.SimpleNamespace(
            crosswalk_interval=1, light_interval=1, plate_interval=1,
            plate_conf=0.4, fps=8),
        output=types.SimpleNamespace(
            evidence_images=False, annotated_video=False, csv_report=False,
            unknown_light_to_review=True, signal_assumption="pedestrian"),
    )


def test_dag_topological_order():
    d = PipelineDAG()
    d.add_node("a", lambda c: None)
    d.add_node("b", lambda c: None)
    d.add_edge("a", "b")
    assert d.topological_sort() == ["a", "b"]


def test_dag_cycle_raises():
    d = PipelineDAG()
    d.add_node("a", lambda c: None)
    d.add_node("b", lambda c: None)
    d.add_edge("a", "b")
    d.add_edge("b", "a")
    with pytest.raises(RuntimeError):
        d.topological_sort()


def test_build_default_dag_wiring():
    cfg = _minimal_cfg()
    comp = {
        "vehicle": _Mock([{"id": 1, "xyxy": [100, 100, 200, 200], "cls": "car", "conf": 0.9}]),
        "crosswalk": _Mock(None),
        "light": _Mock("red"),
        "plate": _Mock([]),
        "trackstate": TrackStateManagerV2("balanced"),
        "engine": BatchViolationEngine("balanced", sample_fps=cfg.inference.fps),
        "viz": _Mock(None),
    }
    dag = build_default_dag(cfg, comp)
    ctx = {"proc": 1, "frame": None, "ts": 0.0, "dets": [], "states": {},
           "mask": None, "light": "unknown", "light_state": "unknown",
           "plates": [],
           "video_writer": None, "evidence_dir": "/tmp",
           "cfg": cfg, "disp": None}
    dag.run(ctx)
    assert ctx["dets"] and ctx["dets"][0]["id"] == 1
    assert 1 in ctx["states"]
    assert ctx["light"] == "red"
    assert ctx["light_state"] == "red"
    assert "disp" in ctx


def test_dag_end_to_end_violation():
    """真实 tracker+engine 经 DAG: 行人绿灯+停车+压线 持续多帧 -> 确认违规 (语义反转后应为 green)。"""
    cfg = _minimal_cfg()
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[100:200, 100:300] = 255
    engine = BatchViolationEngine("balanced", sample_fps=cfg.inference.fps)
    comp = {
        # 每帧返回同一辆静止车; 车体下半部(footprint=0.5)落在 mask 内。
        # 占用按 D2 分母=mask: 车下半部(y150-200)∩mask = 50*100 = 5000 / (mask 20000) = 0.25
        # >= balanced.overlap(0.20), 满足压线。
        "vehicle": _Mock([{"id": 1, "xyxy": [120, 100, 220, 200], "cls": "car", "conf": 0.9}]),
        "crosswalk": _Mock(mask),
        "light": _Mock("green"),
        "plate": _Mock([]),
        "trackstate": TrackStateManagerV2("balanced"),
        "engine": engine,
        "viz": _Mock(None),
    }
    dag = build_default_dag(cfg, comp)
    base = {"mask": None, "light": "unknown", "light_state": "unknown",
            "plates": [],
            "video_writer": None, "evidence_dir": "/tmp",
            "cfg": cfg, "disp": None}
    # 需要足够帧让 tracker 累积出"静止", 再满足 duration=5 才确认
    for i in range(16):
        ctx = dict(base, proc=i + 1, frame=None, ts=i * 0.5, dets=[], states={})
        dag.run(ctx)
    # ②③ 接线: 批处理决策在视频结束后产出事件
    events = engine.decide()
    confirmed = [e for e in events if e["status"] == "confirmed"]
    assert len(confirmed) == 1
    assert confirmed[0]["track_id"] == 1
    assert confirmed[0]["light_state"] == "green"


def test_dag_passes_yolo_light_boxes():
    """M1: dag 把 vehicle.last_light_boxes 传给 light.detect(yolo_light_boxes=...)。"""
    cfg = _minimal_cfg()
    captured = {}

    class _LightMock:
        def detect(self, frame, crosswalk_mask=None, yolo_light_boxes=None):
            captured["boxes"] = yolo_light_boxes
            return {"state": "green"}
        def observe(self, frame):
            return {"obs": "green", "conf": 0.9, "candidates": []}

    class _VehMock:
        last_light_boxes = [(1, 2, 3, 4)]
        def detect(self, frame):
            return [{"id": 1, "xyxy": [0, 0, 10, 10], "cls": "car", "conf": 0.9}]

    comp = {
        "vehicle": _VehMock(), "crosswalk": _Mock(None), "light": _LightMock(),
        "plate": _Mock([]), "trackstate": TrackStateManagerV2("balanced"),
        "engine": BatchViolationEngine("balanced", sample_fps=cfg.inference.fps),
        "viz": _Mock(None),
    }
    dag = build_default_dag(cfg, comp)
    ctx = {"proc": 1, "frame": None, "ts": 0.0, "dets": [], "states": {},
           "mask": None, "light": "unknown", "light_state": "unknown",
           "plates": [], "video_writer": None,
           "evidence_dir": "/tmp", "cfg": cfg, "disp": None}
    dag.run(ctx)
    assert captured["boxes"] == [(1, 2, 3, 4)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
