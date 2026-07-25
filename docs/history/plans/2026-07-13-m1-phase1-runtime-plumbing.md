# M1 Phase 1 — 行人信号灯检测器运行时管线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `TrafficLightDetector` 里新增 `method="ped_classifier"` 路径——候选并集(YOLO灯框∪HSV亮斑)→ 状态分类器 → 复用现有锚点/时序平滑——并在无分类器权重时优雅回退到现有 `color` 路径。本期只做**运行时管线 + 单测**, 不训练模型。

**Architecture:** 三个纯逻辑单元(候选并集去重、分类器 ONNX 包装、编排分支)+ 两处接线(vehicle 暴露 COCO traffic-light 框、dag 路由)。对外接口 `detect(frame)->{state,confidence,...}` 不变(drop-in)。无 ONNX 权重时 `available=False` → 回退 color 路径, 所以本期可独立合入且不改变现有行为。

**Tech Stack:** Python, OpenCV(cv2), numpy, cv2.dnn(ONNX 推理, 运行时不依赖 torch)。测试用 pytest(本机 `.venv`, 或 `PYTHONPATH=src`)。

**权威依据:** `docs/plans/2026-07-13-m1-ped-signal-detector-design.md`(spec)。语义: walk=行人绿灯, stand=行人红灯, off=非信号/熄灭。

---

### Task 1: 候选并集 + IoU 去重 (`signal_candidates.py`)

纯几何函数: 把 YOLO 的 traffic-light 框与 HSV 亮斑候选合并, 重叠(IoU≥阈值)的去重(保留面积大者), 归一化输出。复用 `infrastructure/geometry.iou`。

**Files:**
- Create: `src/redlight/models/signal_candidates.py`
- Test: `tests/unit/test_signal_candidates.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_signal_candidates.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.models.signal_candidates import build_candidates


def test_union_keeps_disjoint_boxes():
    yolo = [(10, 10, 30, 30)]           # (x1,y1,x2,y2) 像素
    hsv = [(100, 100, 120, 120)]
    out = build_candidates(yolo, hsv, frame_w=200, frame_h=200, iou_thr=0.5)
    assert len(out) == 2
    assert {c["source"] for c in out} == {"yolo", "hsv"}


def test_overlapping_boxes_deduped_prefer_larger():
    yolo = [(10, 10, 50, 50)]           # area 1600 (大)
    hsv = [(12, 12, 30, 30)]            # area 324, 与上高度重叠
    out = build_candidates(yolo, hsv, frame_w=200, frame_h=200, iou_thr=0.3)
    assert len(out) == 1
    assert out[0]["source"] == "yolo"   # 保留面积大者
    assert out[0]["box"] == (10, 10, 50, 50)


def test_normalized_center_reported():
    out = build_candidates([(0, 0, 100, 100)], [], frame_w=200, frame_h=200)
    assert out[0]["cx"] == 0.25 and out[0]["cy"] == 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_signal_candidates.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'redlight.models.signal_candidates'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/redlight/models/signal_candidates.py
"""L3 行人信号灯候选并集 (M1 spec M1-D2): YOLO traffic-light 框 ∪ HSV 亮斑候选, IoU 去重。

纯几何, 无 cv2/torch 依赖, 便于单测。输出归一化中心便于下游锚点/聚类复用。
"""
from ..infrastructure.geometry import iou


def build_candidates(yolo_boxes, hsv_boxes, frame_w, frame_h, iou_thr=0.5):
    """合并两路候选框, 重叠(IoU>=iou_thr)者保留面积大者。

    yolo_boxes / hsv_boxes: list of (x1,y1,x2,y2) 像素。
    返回: list of {"box":(x1,y1,x2,y2), "source":"yolo"|"hsv", "area":int, "cx":float, "cy":float}
    """
    tagged = [(b, "yolo") for b in yolo_boxes] + [(b, "hsv") for b in hsv_boxes]
    # 面积降序: 先放大框, 后来的小框若与已保留框重叠则丢弃
    tagged.sort(key=lambda t: -_area(t[0]))
    kept = []
    for box, src in tagged:
        if any(iou(box, k["box"]) >= iou_thr for k in kept):
            continue
        kept.append({
            "box": tuple(int(v) for v in box),
            "source": src,
            "area": int(_area(box)),
            "cx": ((box[0] + box[2]) / 2.0) / frame_w if frame_w else 0.0,
            "cy": ((box[1] + box[3]) / 2.0) / frame_h if frame_h else 0.0,
        })
    return kept


def _area(b):
    return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_signal_candidates.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/redlight/models/signal_candidates.py tests/unit/test_signal_candidates.py
git commit -m "feat(m1): 行人信号灯候选并集 + IoU 去重 (Phase1 T1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 状态分类器 ONNX 包装 + 优雅回退 (`signal_state_classifier.py`)

`classify(roi_bgr)->(label,conf)`, label∈{walk,stand,off}。运行时用 `cv2.dnn` 跑 ONNX(不依赖 torch)。无权重/加载失败 → `available=False`, 编排层据此回退。`_infer` 独立便于 mock 测试(无需真实 ONNX)。

**Files:**
- Create: `src/redlight/models/signal_state_classifier.py`
- Test: `tests/unit/test_signal_state_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_signal_state_classifier.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.models.signal_state_classifier import SignalStateClassifier, LABELS


def test_unavailable_when_no_weights():
    clf = SignalStateClassifier(model_path=None)
    assert clf.available is False
    assert clf.classify(np.zeros((10, 10, 3), np.uint8)) == ("off", 0.0)


def test_argmax_maps_to_label(monkeypatch):
    clf = SignalStateClassifier(model_path=None)
    clf.available = True
    # LABELS = ["walk","stand","off"]; 让 stand(idx1) 概率最高
    monkeypatch.setattr(clf, "_infer", lambda blob: np.array([0.1, 0.7, 0.2]))
    label, conf = clf.classify(np.zeros((48, 48, 3), np.uint8))
    assert label == "stand"
    assert abs(conf - 0.7) < 1e-6


def test_preprocess_shape():
    clf = SignalStateClassifier(model_path=None)
    blob = clf._preprocess(np.zeros((20, 60, 3), np.uint8))
    assert blob.shape == (1, 3, 48, 48)   # NCHW, 48x48
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_signal_state_classifier.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/redlight/models/signal_state_classifier.py
"""L3 行人信号灯状态分类器 (M1 spec M1-D4): ROI -> {walk,stand,off}。

运行时用 cv2.dnn 跑 ONNX(不依赖 torch, 跨 Mac/Windows)。无权重时 available=False,
编排层回退到 color 路径。训练/导出见 Phase 2。
"""
import os
import numpy as np

LABELS = ["walk", "stand", "off"]
_INPUT = 48


class SignalStateClassifier:
    def __init__(self, model_path=None, verbose=True):
        self.model_path = model_path
        self.net = None
        self.available = False
        if model_path and os.path.isfile(model_path):
            try:
                import cv2
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.available = True
                if verbose:
                    print(f"[信号分类器] ONNX 已加载: {model_path}")
            except Exception as e:
                if verbose:
                    print(f"[信号分类器] ONNX 加载失败({e}) -> 回退 color 路径")

    def _preprocess(self, roi_bgr):
        import cv2
        img = cv2.resize(roi_bgr, (_INPUT, _INPUT)).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[None, ...]   # NCHW

    def _infer(self, blob):
        self.net.setInput(blob)
        out = self.net.forward().reshape(-1)
        return out

    def classify(self, roi_bgr):
        """返回 (label, conf)。不可用或 ROI 空 -> ('off', 0.0)。"""
        if not self.available or roi_bgr is None or roi_bgr.size == 0:
            return ("off", 0.0)
        probs = self._infer(self._preprocess(roi_bgr))
        i = int(np.argmax(probs))
        return (LABELS[i], float(probs[i]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_signal_state_classifier.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/redlight/models/signal_state_classifier.py tests/unit/test_signal_state_classifier.py
git commit -m "feat(m1): 信号灯状态分类器 ONNX 包装 + 优雅回退 (Phase1 T2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: VehicleDetector 暴露 COCO traffic-light 框

复用同一次 YOLO 推理, 额外收集 COCO "traffic light" 类框到 `self.last_light_boxes`(不进车辆跟踪)。抽出纯函数 `extract_light_boxes(raw_dets)` 便于测试。

**Files:**
- Modify: `src/redlight/models/vehicle.py`
- Test: `tests/unit/test_vehicle_light_boxes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vehicle_light_boxes.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_vehicle_light_boxes.py -q`
Expected: FAIL — `cannot import name 'extract_light_boxes'`

- [ ] **Step 3: Write minimal implementation**

Add near the top of `src/redlight/models/vehicle.py` (module-level, after imports):

```python
def extract_light_boxes(raw_dets, conf_min=0.25):
    """从原始检测(每项含 name/xyxy/conf)过滤 COCO 'traffic light' 框。

    返回 [(x1,y1,x2,y2), ...] 像素整数框。用于给 M1 做候选(spec M1-D2)。
    """
    out = []
    for d in raw_dets:
        if d.get("name") == "traffic light" and float(d.get("conf", 0.0)) >= conf_min:
            out.append(tuple(int(v) for v in d["xyxy"]))
    return out
```

In `VehicleDetector.__init__`, add `self.last_light_boxes = []`. In `_detect_ultra`, collect raw traffic-light detections and set `self.last_light_boxes` before returning (vehicles unchanged):

```python
    def _detect_ultra(self, frame):
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)
        dets = []
        raw = []
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                name = self.model.names[cls]
                xyxy = [float(v) for v in b.xyxy[0].tolist()]
                raw.append({"name": name, "xyxy": xyxy, "conf": float(b.conf[0])})
                if name not in self.classes:
                    continue
                x1, y1, x2, y2 = xyxy
                dets.append({"id": -1, "xyxy": [x1, y1, x2, y2],
                             "conf": float(b.conf[0]), "cls": name})
        self.last_light_boxes = extract_light_boxes(raw, conf_min=0.25)
        return dets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_vehicle_light_boxes.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/redlight/models/vehicle.py tests/unit/test_vehicle_light_boxes.py
git commit -m "feat(m1): VehicleDetector 复用推理暴露 COCO traffic-light 框 (Phase1 T3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: TrafficLightDetector 新增 `ped_classifier` 编排分支

`detect(frame, crosswalk_mask=None, yolo_light_boxes=None)` 新增可选参数。当 `method=="ped_classifier"` 且分类器可用: 构建候选并集 → 每候选 ROI 分类 → walk→green/stand→red/off→丢弃 → 生成本帧 obs 喂给**现有** `global_recent` 时序平滑与 `_state_from_global`。否则(或分类器不可用)走现有 color 路径。接口返回结构不变。

**Files:**
- Modify: `src/redlight/models/traffic_light.py`
- Test: `tests/unit/test_traffic_light_ped.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_traffic_light_ped.py
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
    """按固定序列返回状态, 便于测试编排。"""
    available = True
    def __init__(self, label): self._label = label
    def classify(self, roi): return (self._label, 0.9)


def test_ped_classifier_stable_green():
    det = TrafficLightDetector(_cfg("ped_classifier"), verbose=False)
    det.classifier = _StubClf("walk")            # 注入桩分类器
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[20:40, 100:120] = (0, 255, 0)          # 给个亮斑做候选
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_traffic_light_ped.py -q`
Expected: FAIL — `AttributeError`/编排分支不存在(detect 不接受 yolo_light_boxes 或无 ped 分支)。

- [ ] **Step 3: Write minimal implementation**

In `TrafficLightDetector.__init__`, read method + build classifier (lazy, weights optional):

```python
        self.method = str(getattr(tl, "method", "color"))
        clf_path = getattr(getattr(cfg, "models", None), "ped_signal_onnx", None) \
            if getattr(cfg, "models", None) else None
        from .signal_state_classifier import SignalStateClassifier
        self.classifier = SignalStateClassifier(clf_path, verbose=verbose)
```

Change `detect` signature and add the branch (keep the existing color body as the else):

```python
    def detect(self, frame, crosswalk_mask=None, yolo_light_boxes=None):
        self._fi += 1
        if frame is not None:
            self._last_w = frame.shape[1]
        if self.method == "ped_classifier" and getattr(self.classifier, "available", False):
            return self._detect_ped(frame, yolo_light_boxes or [])
        # --- 现有 color 路径(不变) ---
        spots = self._candidates(frame)
        ...
```

Add the new method (uses candidate union + classifier + existing smoothing):

```python
    def _detect_ped(self, frame, yolo_light_boxes):
        from .signal_candidates import build_candidates
        h, w = (frame.shape[0], frame.shape[1]) if frame is not None else (1, 1)
        hsv_boxes = [s["box"] for s in self._candidates(frame)]   # 复用 HSV 亮斑框
        cands = build_candidates(yolo_light_boxes, hsv_boxes, w, h)
        obs = None
        best_conf = 0.0
        for c in cands:
            x1, y1, x2, y2 = c["box"]
            roi = frame[max(0, y1):y2, max(0, x1):x2] if frame is not None else None
            label, conf = self.classifier.classify(roi)
            if label == "off":
                continue
            if conf > best_conf:
                best_conf = conf
                obs = "green" if label == "walk" else "red"
        self.global_recent.append(obs)
        state, reason, conf = self._state_from_global()
        return {"state": state, "confidence": conf, "stable": obs is not None,
                "is_flashing": state == "flashing", "reason": "ped_" + reason,
                "candidates": cands, "track": None, "anchor": None,
                "g_px": 0, "r_px": 0}
```

Note: `_candidates` returns dicts with a `"box"` key (see existing code); it already runs on saturated bright-spots. `_state_from_global` and `global_recent` are the existing smoothing.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_traffic_light_ped.py tests/unit/test_traffic_light.py -q`
Expected: PASS — new ped tests pass AND the existing 10 color-path tests still pass (color path unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/redlight/models/traffic_light.py tests/unit/test_traffic_light_ped.py
git commit -m "feat(m1): TrafficLightDetector 新增 ped_classifier 编排分支 (Phase1 T4)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: DAG 路由 yolo_light_boxes 到 light 节点

`n_light` 从 `ctx` 取 vehicle 检测器暴露的灯框传给 `detect`。检测器实例在 `comp["vehicle"]`, 灯框在 `last_light_boxes`。

**Files:**
- Modify: `src/redlight/pipeline/dag.py`
- Test: `tests/integration/test_dag.py`(扩展)

- [ ] **Step 1: Write the failing test**

Add to `tests/integration/test_dag.py`:

```python
def test_dag_passes_yolo_light_boxes():
    cfg = _minimal_cfg()
    captured = {}

    class _LightMock:
        def detect(self, frame, crosswalk_mask=None, yolo_light_boxes=None):
            captured["boxes"] = yolo_light_boxes
            return {"state": "green"}

    class _VehMock:
        last_light_boxes = [(1, 2, 3, 4)]
        def detect(self, frame):
            return [{"id": 1, "xyxy": [0, 0, 10, 10], "cls": "car", "conf": 0.9}]

    comp = {
        "vehicle": _VehMock(), "crosswalk": _Mock(None), "light": _LightMock(),
        "plate": _Mock([]), "trackstate": TrackStateManagerV2("balanced"),
        "engine": ViolationEngineV2("balanced"), "viz": _Mock(None),
    }
    dag = build_default_dag(cfg, comp)
    ctx = {"proc": 1, "frame": None, "ts": 0.0, "dets": [], "states": {},
           "mask": None, "light": "unknown", "light_state": "unknown",
           "plates": [], "new_events": [], "csv_rows": [], "video_writer": None,
           "evidence_dir": "/tmp", "cfg": cfg, "disp": None}
    dag.run(ctx)
    assert captured["boxes"] == [(1, 2, 3, 4)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/integration/test_dag.py::test_dag_passes_yolo_light_boxes -q`
Expected: FAIL — `captured["boxes"]` is None (dag 未传灯框)。

- [ ] **Step 3: Write minimal implementation**

In `dag.py`, `build_default_dag` captures the vehicle component, and `n_light` passes its boxes:

```python
    det = comp["vehicle"]
    ...
    def n_light(ctx):
        if ctx["proc"] % light_int == 0:
            boxes = getattr(det, "last_light_boxes", None)
            res = tl.detect(ctx["frame"], yolo_light_boxes=boxes)
            ctx["light"] = res
            ctx["light_state"] = res.get("state", "unknown") if isinstance(res, dict) else res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/integration/test_dag.py -q`
Expected: PASS (all dag tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/redlight/pipeline/dag.py tests/integration/test_dag.py
git commit -m "feat(m1): DAG 把 vehicle 暴露的 traffic-light 框路由给 light 节点 (Phase1 T5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 全量回归 + config 文档

确认全套测试仍绿, 并在 `config.yaml` 记录新 method 与可选权重路径。

**Files:**
- Modify: `configs/config.yaml`

- [ ] **Step 1: Add config keys**

在 `models:` 段增加(可选权重, 缺省即回退 color):

```yaml
  ped_signal_onnx: "models/ped_signal.onnx"   # 行人信号状态分类器(可选, 缺省回退 color 路径; Phase2 训练产出)
```

在 `traffic_light:` 段的 `method` 注释补充: `# color(默认) | ped_classifier(候选并集+状态分类器, 需 ped_signal.onnx) | model`

- [ ] **Step 2: Run full suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -q && python scripts/run_tl_tests.py`
Expected: 全绿(既有 + 新增 signal_candidates/classifier/vehicle_light/ped/dag 测试), run_tl_tests 10/10。

- [ ] **Step 3: Commit**

```bash
git add configs/config.yaml
git commit -m "docs(m1): config 记录 ped_classifier method 与 ped_signal.onnx 可选权重 (Phase1 T6)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 后续 (不在本 Phase)

- **Phase 2 — 数据+训练**(需视频资产+标注): `build_ped_signal_crops.py`(候选裁 crop + 弱标签自举) + `train_ped_signal.py`(训练 tiny-CNN / 特征分类器, 导出 `models/ped_signal.onnx`) + 灯态画廊人工校验。产出权重后 `ped_classifier` 才真正生效。
- **Phase 3 — 评测**(需资产, 且 owns `eval_light_*`): leave-one-video-out 分类器交叉验证 + prior-free 端到端 M1 评测; 新旧路径 A/B, 确认净提升后删除 `color` 路径与 `light_priors.json` shipped 用法。

## 环境说明
- 本 Phase 全部单测在无视频资产/无 torch 下可跑(本机 `.venv` 已装 cv2/numpy; ONNX 分类器路径用桩测试, 不需真实权重)。
- 多 agent: 本 Phase 只碰 `models/{signal_candidates,signal_state_classifier,vehicle,traffic_light}.py` + `dag.py` + 对应测试 + config; **勿用 `git add -A`**, 按各 Task 的 scoped `git add` 提交。
</content>
