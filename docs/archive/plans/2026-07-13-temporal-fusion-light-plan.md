# TemporalFusion(②)— 灯态时序融合 + 中间态 schema  Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务实现。步骤用 `- [ ]` 复选框。
> 权威 spec:`docs/plans/2026-07-13-layered-architecture-refactor-design.md`(v2)。

**Goal:** 把灯态的跨帧时序逻辑(窗口投票/迟滞/闪烁/unknown 分段)从 `TrafficLightDetector` 抽到独立的 `TemporalFusion` 层,作为**纯函数**消费"逐帧灯态观测序列"→ 产出 `light_segments` 中间态;并锁定中间态 schema 的灯态部分。

**Architecture:** 新增 `pipeline/temporal_fusion.py`(纯函数 `fuse_light`)+ `pipeline/intermediate_state.py`(schema/契约 + 合并助手)。检测器新增"只出单帧观测"的入口。**本期不改运行时接线**(cli/dag 仍走现有流式路径),避免与未重构的判定层③半破坏;TemporalFusion 作为新能力独立落地并被单测锁定,待③改区间代数时再接入。

**Tech Stack:** Python,numpy 无关(纯逻辑),pytest。语义须与现 `traffic_light._state_from_global` + hysteresis 对齐(用 `run_tl_tests` 的既有语义做特征化测试)。

**关键约束(spec A-D2)**:时变状态只在同质段内有界填充;单帧 `unknown` 段内跳过,连续 `unknown` 超 `anchor_hold` 开新 `unknown` 段,绝不前填。`flashing` 是时序模式(绿红跳变≥N)由本层显式标记。

---

### Task 1: 中间态 schema — light_segments 契约 + 合并助手

**Files:**
- Create: `src/redlight/pipeline/intermediate_state.py`
- Test: `tests/unit/test_intermediate_state.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_intermediate_state.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.intermediate_state import make_light_segment, merge_adjacent_segments, LIGHT_STATES


def test_light_states_enum():
    assert set(LIGHT_STATES) == {"green", "red", "flashing", "unknown"}


def test_make_light_segment_fields():
    s = make_light_segment(0.0, 5.0, "green", 0.9, evidence="visible")
    assert s == {"start_s": 0.0, "end_s": 5.0, "state": "green", "conf": 0.9, "evidence": "visible"}


def test_merge_adjacent_same_state():
    segs = [make_light_segment(0, 2, "green", 0.8), make_light_segment(2, 5, "green", 0.9),
            make_light_segment(5, 7, "red", 0.7)]
    merged = merge_adjacent_segments(segs)
    assert len(merged) == 2
    assert merged[0] == {"start_s": 0, "end_s": 5, "state": "green", "conf": 0.9, "evidence": None}  # conf 取 max
    assert merged[1]["state"] == "red"
```

- [ ] **Step 2: 运行验证失败**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_intermediate_state.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: 实现**

```python
# src/redlight/pipeline/intermediate_state.py
"""L5 中间态契约(架构 spec §5): TemporalFusion 产出、判定层③/呈现④/评测消费。

本期只定义灯态部分(light_segments);tracks/occupancy/review_flags 待后续 TemporalFusion
扩展时补齐。保持纯 dict(便于 JSON 落盘 + 跨语言/跨 agent 消费)。
"""

LIGHT_STATES = ("green", "red", "flashing", "unknown")


def make_light_segment(start_s, end_s, state, conf, evidence=None):
    """构造一个灯态段。state ∈ LIGHT_STATES; evidence ∈ visible|inferred|occluded|None。"""
    assert state in LIGHT_STATES, f"非法灯态: {state}"
    return {"start_s": start_s, "end_s": end_s, "state": state,
            "conf": conf, "evidence": evidence}


def merge_adjacent_segments(segments):
    """合并相邻同 state 段(conf 取 max, end 取后者)。输入按时间有序。"""
    out = []
    for s in segments:
        if out and out[-1]["state"] == s["state"]:
            out[-1]["end_s"] = s["end_s"]
            out[-1]["conf"] = max(out[-1]["conf"], s["conf"])
        else:
            out.append(dict(s))
    return out
```

- [ ] **Step 4: 运行验证通过**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_intermediate_state.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: 提交**

```bash
git add src/redlight/pipeline/intermediate_state.py tests/unit/test_intermediate_state.py
git commit -m "feat(arch): 中间态 light_segments 契约 + 合并助手 (② schema)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `TemporalFusion.fuse_light` — 纯函数窗口投票 + 迟滞 + 闪烁 + unknown 分段

把 `traffic_light._state_from_global`(窗口多数决/闪烁)+ hysteresis + unknown 有界处理,重写为消费**完整观测序列**的纯函数(批处理,对应 Q4)。

**Files:**
- Create: `src/redlight/pipeline/temporal_fusion.py`
- Test: `tests/unit/test_temporal_fusion.py`

- [ ] **Step 1: 写失败测试(移植 run_tl_tests 的语义到序列级)**

```python
# tests/unit/test_temporal_fusion.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.temporal_fusion import fuse_light

def _obs(states, dt=0.125):
    # 生成 (ts, obs, conf) 序列; obs ∈ green|red|off|None
    return [(i * dt, s, 0.9) for i, s in enumerate(states)]

def _states(segs):
    return [s["state"] for s in segs]

def test_stable_green():
    segs = fuse_light(_obs(["green"] * 12), window=8, hysteresis=0.68, flicker_toggle=4)
    assert _states(segs) == ["green"]

def test_stable_red():
    segs = fuse_light(_obs(["red"] * 12), window=8, hysteresis=0.68, flicker_toggle=4)
    assert _states(segs) == ["red"]

def test_green_then_red_two_segments():
    segs = fuse_light(_obs(["green"] * 10 + ["red"] * 10), window=6, hysteresis=0.68, flicker_toggle=4)
    assert _states(segs) == ["green", "red"]

def test_flashing_detected():
    # 绿红逐帧交替 -> flashing(跳变多)
    segs = fuse_light(_obs(["green", "red"] * 8), window=8, hysteresis=0.68, flicker_toggle=4)
    assert "flashing" in _states(segs)

def test_single_unknown_does_not_break_segment():
    # 段内单帧 unknown(None) 被跳过, 不切段
    segs = fuse_light(_obs(["green"] * 5 + [None] + ["green"] * 6),
                      window=6, hysteresis=0.68, flicker_toggle=4, unknown_hold=4)
    assert _states(segs) == ["green"]

def test_sustained_unknown_opens_new_segment():
    # 连续 unknown 超过 unknown_hold -> 开新 unknown 段, 不前填
    # (window=6 需窗口全 off 才判 unknown, 故 None 数需足够多: 6 填满窗口 + >unknown_hold 持续)
    segs = fuse_light(_obs(["green"] * 6 + [None] * 14), window=6, hysteresis=0.68,
                      flicker_toggle=4, unknown_hold=4)
    assert _states(segs) == ["green", "unknown"]

def test_intermittent_green_not_flashing():
    # 偶发 off 的绿, 不应误判 flashing
    segs = fuse_light(_obs(["green", "green", "off", "green", "green"] * 3),
                      window=8, hysteresis=0.68, flicker_toggle=4)
    assert "flashing" not in _states(segs)
    assert "green" in _states(segs)
```

- [ ] **Step 2: 运行验证失败**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_temporal_fusion.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: 实现**

```python
# src/redlight/pipeline/temporal_fusion.py
"""L5 时序后处理(②, 架构 spec §2/§4)。本期: 灯态融合 fuse_light。

消费逐帧观测序列 -> light_segments(纯函数, 批处理)。承接原
TrafficLightDetector._state_from_global + hysteresis 的语义, 但操作完整序列而非流式 deque。
时变红线(A-D2): unknown 段内单帧跳过; 连续 unknown 超阈开新 unknown 段, 不前填。
"""
from collections import deque
from .intermediate_state import make_light_segment, merge_adjacent_segments


def _window_state(win, flicker_toggle):
    """窗口(green/red/off 的最近序列)-> (state, conf)。移植 _state_from_global。"""
    seq = [c for c in win if c in ("green", "red")]
    green = seq.count("green"); red = seq.count("red"); seen = green + red
    if seen == 0:
        return "unknown", 0.0
    gr, rr = green / seen, red / seen
    toggles = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    if green > 0 and red > 0 and toggles >= flicker_toggle and gr < 0.6 and rr < 0.6:
        return "flashing", round(max(gr, rr), 3)
    if gr >= 0.6:
        return "green", round(gr, 3)
    if rr >= 0.6:
        return "red", round(rr, 3)
    return "unknown", round(max(gr, rr), 3)


def fuse_light(observations, window=24, hysteresis=0.68, flicker_toggle=4, unknown_hold=8):
    """observations: 有序 [(ts, obs, conf)], obs ∈ green|red|off|None(None/off=未点亮/未知)。
    返回 light_segments(见 intermediate_state)。
    """
    win = deque(maxlen=window)
    raw = []                       # 每帧 (ts, committed_state, conf)
    committed = None               # 迟滞已提交状态
    unknown_run = 0
    for ts, obs, _c in observations:
        win.append(obs if obs in ("green", "red") else "off")
        st, conf = _window_state(win, flicker_toggle)
        if st == "unknown":
            unknown_run += 1
            if unknown_run <= unknown_hold and committed is not None:
                raw.append((ts, committed, conf))      # 段内短 unknown: 维持已提交(段内跳过)
            else:
                committed = "unknown"                   # 超阈: 真开 unknown 段
                raw.append((ts, "unknown", conf))
            continue
        unknown_run = 0
        # 迟滞: 已有状态时, 需被反状态占比达 hysteresis 才翻转
        if committed is None or st == committed or conf >= hysteresis:
            committed = st
        raw.append((ts, committed, conf))
    # 折叠成段
    segs = []
    for i, (ts, state, conf) in enumerate(raw):
        end = raw[i + 1][0] if i + 1 < len(raw) else ts
        segs.append(make_light_segment(ts, end, state, conf))
    return merge_adjacent_segments(segs)
```

- [ ] **Step 4: 运行验证通过**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_temporal_fusion.py -q`
Expected: PASS (7 passed)。若个别边界(flicker/hysteresis 常数)与预期差,微调 `_window_state`/迟滞分支使语义与 `run_tl_tests` 一致,不改测试意图。

- [ ] **Step 5: 提交**

```bash
git add src/redlight/pipeline/temporal_fusion.py tests/unit/test_temporal_fusion.py
git commit -m "feat(arch): TemporalFusion.fuse_light — 灯态窗口投票+迟滞+闪烁+unknown分段 (纯函数, ②)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 检测器暴露"单帧观测"入口 `observe(frame)`

让 `TrafficLightDetector` 提供纯逐帧观测出口(不依赖内部时序 deque),供②消费。**不删现有 `detect`**(流式路径仍在用,待③接入后再收敛),只新增 `observe`。

**Files:**
- Modify: `src/redlight/models/traffic_light.py`
- Test: `tests/unit/test_traffic_light_observe.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_traffic_light_observe.py
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
    frame[20:35, 100:115] = (0, 255, 0)  # 绿亮斑
    out = det.observe(frame)
    assert set(out) >= {"obs", "conf", "candidates"}
    assert out["obs"] in ("green", "red", "off", None)

def test_observe_is_stateless_across_calls():
    # observe 不应改变会影响下次结果的内部时序(同帧多次调用结果一致)
    det = TrafficLightDetector(_cfg(), verbose=False)
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[20:35, 100:115] = (0, 255, 0)
    a = det.observe(frame)["obs"]
    b = det.observe(frame)["obs"]
    assert a == b
```

- [ ] **Step 2: 运行验证失败**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_traffic_light_observe.py -q`
Expected: FAIL — `AttributeError: observe`。

- [ ] **Step 3: 实现**

在 `TrafficLightDetector` 加(不改 `detect`):

```python
    def observe(self, frame):
        """单帧灯态观测(供 TemporalFusion 消费): 只出这帧看到什么, 不做跨帧时序。

        obs ∈ 'green'|'red'|'off'|None。复用现有单帧候选/选灯的**空间**部分,
        但不 append global_recent、不跑 _state_from_global(那是②的活)。
        """
        spots = self._candidates(frame)
        # _classify 已给每个候选颜色; 取主色作为单帧 obs(不投票)
        greens = sum(s["area"] for s in spots if s.get("color") == "green")
        reds = sum(s["area"] for s in spots if s.get("color") == "red")
        if greens == 0 and reds == 0:
            obs, conf = ("off", 0.0)
        elif greens >= reds:
            obs, conf = ("green", round(greens / (greens + reds), 3))
        else:
            obs, conf = ("red", round(reds / (greens + reds), 3))
        return {"obs": obs, "conf": conf, "candidates": spots}
```

> 注:本 Task 先给最简单帧观测(按候选主色),不含空间锚。锚点的空间稳定是后续 Task(spec §6 干净接口:①出 cx/cy/lamp_score,②只读坐标)。M1 `ped_classifier` 就绪后,`observe` 可改为调分类器出 obs。

- [ ] **Step 4: 运行验证通过**

Run: `PYTHONPATH=src python -m pytest tests/unit/test_traffic_light_observe.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: 提交**

```bash
git add src/redlight/models/traffic_light.py tests/unit/test_traffic_light_observe.py
git commit -m "feat(arch): TrafficLightDetector.observe() 单帧观测入口 (①→②解耦, detect 不动)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 全量回归 + schema 锁定通告

- [ ] **Step 1: 全量测试**

Run: `source .venv/bin/activate && python -m pytest tests/ -q && python scripts/run_tl_tests.py`
Expected: 全绿(现有 + 新增 intermediate_state/temporal_fusion/observe 测试);`run_tl_tests` 10/10(现有流式路径未动)。

- [ ] **Step 2: 在 spec 里把 §5 灯态 schema 标记为"已锁定 v1"**,并通告 Lingma:`eval_temporal_fusion.py` 可按 `intermediate_state.make_light_segment` 的 `light_segments` 契约开写(消费 `fuse_light` 输出 vs `light_states.csv`)。

```bash
# 编辑 docs/plans/2026-07-13-layered-architecture-refactor-design.md: §5 加 "灯态 schema 已锁定(intermediate_state.py)"
git add docs/plans/2026-07-13-layered-architecture-refactor-design.md
git commit -m "docs(arch): 灯态中间态 schema 锁定, 解锁 Lingma eval_temporal_fusion"
```

---

## 范围外(后续 plan)
- 空间锚拆分(anchor_hold/reanchor 跨帧保持搬②;①出 cx/cy/lamp_score)。
- tracks/occupancy/stationary/plate 并入中间态(TrackStateManagerV2 + PlateConsensus 归②)。
- 运行时接线(cli/dag 收集 observe 序列 → fuse_light 批处理)——与③判定改区间代数一起做,避免半破坏。
- M1 `ped_classifier` 就绪后 `observe` 改调分类器。
</content>
