"""L5 任务编排: 违规判定状态机 (V2)。

违规语义 (E12 反转, 权威见 docs/plans/2026-07-12-design-requirements-v2.md):
    违规 = 行人绿灯/闪烁清空相位 AND 车辆静止 AND 车辆压斑马线(overlap>=threshold),
    且持续 duration 个采样帧。
    🔴 红灯 = 车辆可通行, 不算违规。
    ❓ 灯态未知 + 斑马线被遮挡 -> review (交人复核, D1), 绝不自动 confirmed。

E13 教训: 旧"红灯压线=违规"语义已废弃。曾有 mode=red_light/pedestrian_green 双模式开关,
默认指向旧语义, 是 footgun (破坏了本模块单测并与权威需求相悖), 已彻底删除。
"""
import numpy as np
from ..infrastructure.geometry import compute_overlap_ratio
from .tracker import SENSITIVITY_PRESETS

OCCLUSION_MIN_AREA_RATIO = 0.005


class ViolationEngineV2:
    def __init__(self, preset="balanced", unknown_to_review=True, min_event_gap_sec=5,
                 fill_gap_sec=2.0):
        if preset not in SENSITIVITY_PRESETS:
            preset = "balanced"
        self.preset_name = preset
        p = SENSITIVITY_PRESETS[preset]
        self.overlap = p["overlap"]
        self.duration = p["duration"]
        self.gap = min_event_gap_sec
        self.unknown_to_review = unknown_to_review
        self.fill_gap = fill_gap_sec   # 未知灯短时向前填充窗口(秒)
        self.events = []
        self.active = {}
        self.last_event_time = {}
        self._last_known = None        # (state, ts): 最近一次确知灯态
        self._eid = 0

    def _resolve_light(self, light_state, timestamp):
        """未知灯短时向前填充: brief unknown 沿用最近已知灯态。

        覆盖 08 中段手机未拍到灯的场景——前后确认绿灯, 中间几帧 unknown
        仍按绿灯处理, 不丢失违规。超过 fill_gap 的长时间 unknown 维持 unknown。
        """
        if light_state != "unknown":
            self._last_known = (light_state, timestamp)
            return light_state
        if self._last_known is not None:
            st, ts = self._last_known
            if timestamp - ts <= self.fill_gap:
                return st
        return "unknown"

    @staticmethod
    def _is_occluded(mask):
        if mask is None:
            return True
        if getattr(mask, "ndim", 0) != 2:
            return True
        h, w = mask.shape
        area = float(np.count_nonzero(mask))
        if area < OCCLUSION_MIN_AREA_RATIO * h * w:
            return True
        # E14 fix: 全宽斑马线常态触左右/上边 -> 这些不算遮挡;
        # 仅当掩膜触**底边**(斑马线被画面下沿截断, 看不到完整)才判遮挡
        if mask[-1, :].any():
            return True
        return False

    def evaluate(self, track_states, mask, light_state, timestamp):
        if isinstance(light_state, dict):
            light_state = light_state.get("state", "unknown")
        # 未知灯短时向前填充: brief unknown 沿用最近已知灯态(覆盖 08 中段缺失)
        light_state = self._resolve_light(light_state, timestamp)
        occluded = self._is_occluded(mask)
        new_events = []
        for tid, st in track_states.items():
            if not st.get("active", False):
                continue
            if not st.get("stationary", False):
                self._reset(tid)
                continue
            ratio = compute_overlap_ratio(st["box"], mask, footprint=0.5, denom="mask")
            on_crosswalk = ratio >= self.overlap
            a = self.active.setdefault(
                tid, {"sustained": 0, "emitted": False, "cond_start": timestamp}
            )
            # 违规 = 行人绿灯/闪烁 + 静止 + 压线 (E12 语义; 红灯不违规)
            if on_crosswalk and light_state in ("green", "flashing"):
                self._accumulate(a, tid, st, light_state, timestamp, new_events, "confirmed")
            # 灯态未知 + 斑马线被遮挡 -> review (D1, 安全侧交人复核)
            elif on_crosswalk and light_state == "unknown" and self.unknown_to_review and occluded:
                self._accumulate(a, tid, st, light_state, timestamp, new_events, "review")
            else:
                self._reset(tid)
        return new_events

    def _accumulate(self, a, tid, st, light_state, timestamp, new_events, status):
        if a["sustained"] == 0:
            a["cond_start"] = timestamp
        a["sustained"] += 1
        if a["sustained"] >= self.duration and not a["emitted"]:
            last = self.last_event_time.get(tid, -1e9)
            if timestamp - last >= self.gap:
                self._eid += 1
                ev = {
                    "event_id": self._eid,
                    "track_id": tid,
                    "status": status,
                    "start_ts": round(a["cond_start"], 2),
                    "end_ts": round(timestamp, 2),
                    "vehicle_class": st.get("cls", ""),
                    "confidence": round(float(st.get("conf", 0.0)), 3),
                    "light_state": light_state,
                    "evidence_image": "",
                }
                new_events.append(ev)
                self.events.append(ev)
                self.last_event_time[tid] = timestamp
                a["emitted"] = True

    def _reset(self, tid):
        if tid in self.active:
            self.active[tid] = {"sustained": 0, "emitted": False, "cond_start": 0.0}
