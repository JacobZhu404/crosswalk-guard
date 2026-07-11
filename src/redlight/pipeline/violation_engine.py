"""L5 任务编排: 违规判定状态机 (V2, 语义反转版)。

判定条件 (全部满足且持续 duration 个采样帧):
    light_state in ('green','flashing') AND 车辆静止 AND 车辆压斑马线

语义 (识别对象 = 斑马线行人信号灯, 见设计文档 §5.3.1):
    - 🔴 red(行人禁行): 车辆可通行 -> 不违规
    - 🟢 green(行人通行): 车辆须让行 -> 静止压线 = 违规(confirmed)
    - 🟡 flashing(行人清空闪烁): 车辆须让行 -> 静止压线 = 违规(confirmed)
    - ❓ unknown(未检出灯): 默认不违规; 仅当斑马线被遮挡(occlusion_flag)时发 review

注: 2026-07-11 起移除原 red_light 双模式 —— 旧"红灯压线=违规"语义已被推翻,
系统唯一正确的假设是"斑马线行人信号灯", 故不再保留可切换的旧逻辑(避免 footgun)。

去重: 同一 track 一次持续条件内只发一次; 两次事件间隔需 >= min_event_gap_sec。"""
import numpy as np
from ..infrastructure.geometry import compute_overlap_ratio
from .tracker import SENSITIVITY_PRESETS

OCCLUSION_MIN_AREA_RATIO = 0.005  # 掩膜面积低于此比例视为斑马线看不全 -> 遮挡


class ViolationEngineV2:
    def __init__(self, preset="balanced", unknown_to_review=True, min_event_gap_sec=5):
        if preset not in SENSITIVITY_PRESETS:
            preset = "balanced"
        self.preset_name = preset
        p = SENSITIVITY_PRESETS[preset]
        self.overlap = p["overlap"]
        self.duration = p["duration"]
        self.gap = min_event_gap_sec
        self.unknown_to_review = unknown_to_review
        self.events = []
        self.active = {}
        self.last_event_time = {}
        self._eid = 0

    @staticmethod
    def _is_occluded(mask):
        """斑马线掩膜是否疑似被遮挡/看不全 (无法可靠判定压线)。

        判定: 掩膜为 None / 非二维 / 面积过小(基本看不到) /
        触及左·右·上边框(斑马线被画面截断, 看不全)。
        注: 不检查底边框 —— 地面斑马线常位于画面底部, 属正常。
        """
        if mask is None:
            return True
        if getattr(mask, "ndim", 0) != 2:
            return True
        h, w = mask.shape
        area = float(np.count_nonzero(mask))
        if area < OCCLUSION_MIN_AREA_RATIO * h * w:
            return True
        if mask[0, :].any() or mask[:, 0].any() or mask[:, -1].any():
            return True
        return False

    def evaluate(self, track_states, mask, light_state, timestamp):
        if isinstance(light_state, dict):
            light_state = light_state.get("state", "unknown")
        occluded = self._is_occluded(mask)
        new_events = []
        for tid, st in track_states.items():
            if not st.get("active", False):
                continue
            if not st.get("stationary", False):
                self._reset(tid)
                continue
            ratio = compute_overlap_ratio(st["box"], mask)
            on_crosswalk = ratio >= self.overlap
            a = self.active.setdefault(
                tid, {"sustained": 0, "emitted": False, "cond_start": timestamp}
            )
            # 唯一正确语义 (§5.3.1): 行人绿灯/闪烁 + 静止 + 压线 = 违规
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
