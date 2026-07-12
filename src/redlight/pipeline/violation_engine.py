"""L5 任务编排: 违规判定状态机 (V2, 双模式版)。

模式:
    - red_light: 红灯期间车辆静止压线 = 违规
    - pedestrian_green: 行人绿灯/闪烁期间车辆静止压线 = 违规

判定条件 (全部满足且持续 duration 个采样帧):
    light_state in 违规灯态 AND 车辆静止 AND 车辆压斑马线(overlap>=threshold)
"""
import numpy as np
from ..infrastructure.geometry import compute_overlap_ratio
from .tracker import SENSITIVITY_PRESETS

OCCLUSION_MIN_AREA_RATIO = 0.005


class ViolationEngineV2:
    def __init__(self, preset="balanced", unknown_to_review=True, min_event_gap_sec=5,
                 fill_gap_sec=2.0, mode="red_light"):
        if preset not in SENSITIVITY_PRESETS:
            preset = "balanced"
        self.preset_name = preset
        p = SENSITIVITY_PRESETS[preset]
        self.overlap = p["overlap"]
        self.duration = p["duration"]
        self.gap = min_event_gap_sec
        self.unknown_to_review = unknown_to_review
        self.fill_gap = fill_gap_sec   # 未知灯短时向前填充窗口(秒)
        self.mode = mode if mode in ("red_light", "pedestrian_green") else "red_light"
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
            if self.mode == "red_light":
                if on_crosswalk and light_state == "red":
                    self._accumulate(a, tid, st, light_state, timestamp, new_events, "confirmed")
                elif on_crosswalk and light_state == "unknown" and self.unknown_to_review and occluded:
                    self._accumulate(a, tid, st, light_state, timestamp, new_events, "review")
                else:
                    self._reset(tid)
            else:
                if on_crosswalk and light_state in ("green", "flashing"):
                    self._accumulate(a, tid, st, light_state, timestamp, new_events, "confirmed")
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
