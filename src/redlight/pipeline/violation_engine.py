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


# ---------------------------------------------------------------------------
# ②③ 接线: 批处理版违规判定引擎 (替代流式 V2)
# ---------------------------------------------------------------------------

class BatchViolationEngine:
    """批处理版违规判定引擎 (②③ 接线, 架构 spec §2/§4)。

    逐帧 accumulate() 收集原始观测与跟踪状态,
    视频结束后 decide() 调用 fuse_light + interval 聚合 + decide_violations 产出事件。
    事件格式与 ViolationEngineV2 兼容, CLI 可无缝替换。
    """

    def __init__(self, preset="balanced", sample_fps=8.0, unknown_to_review=True,
                 min_event_gap_sec=5.0, fuse_kwargs=None, occ_denom="mask",
                 min_persistent_green_run_s=6.0):
        if preset not in SENSITIVITY_PRESETS:
            preset = "balanced"
        self.preset_name = preset
        p = SENSITIVITY_PRESETS[preset]
        # 占道分母解耦(改定): box 用独立阈值 box_overlap, mask 用 overlap(保留 D2 现状)
        self.overlap_thr = p["box_overlap"] if occ_denom == "box" else p["overlap"]
        # 占道分母: "mask"=占斑马线比例(D2 现状) | "box"=车足迹占多少压线(诊断/未来)
        self.occ_denom = occ_denom if occ_denom in ("mask", "box") else "mask"
        # duration 在 preset 中是"采样帧数", 转换为秒供 decide_violations
        self.min_duration_s = p["duration"] / max(sample_fps, 1e-3)
        self.gap = min_event_gap_sec
        self.unknown_to_review = unknown_to_review
        self.min_persistent_green_run_s = min_persistent_green_run_s
        self.fuse_kwargs = dict(fuse_kwargs) if fuse_kwargs else {}
        self._light_obs = []       # [(ts, obs, conf), ...]
        self._occ_samples = []     # [(ts, occluded_bool), ...] 供 evidence 打标(review, D1)
        self._track_samples = {}   # tid -> [{ts, stationary, box, overlap, cls, conf}, ...]
        self.events = []

    def accumulate(self, track_states, mask, light_observation, timestamp):
        """逐帧收集数据(建议在 DAG visualize 节点内联调用)。

        Args:
            track_states: tracker.update() 输出 {tid: {active, stationary, box, cls, conf}}
            mask: 斑马线掩膜(ndarray|None)
            light_observation: TrafficLightDetector.observe() 输出 {"obs": ..., "conf": ...}
            timestamp: 当前时间戳(秒)
        """
        obs = "off"
        conf = 0.0
        if isinstance(light_observation, dict):
            obs = light_observation.get("obs", "off")
            conf = light_observation.get("conf", 0.0)
        self._light_obs.append((timestamp, obs, conf))
        # 逐帧遮挡(灯 unknown + 斑马线被挡时用于打 evidence=occluded -> review, D1)
        self._occ_samples.append((timestamp, ViolationEngineV2._is_occluded(mask)))

        for tid, st in track_states.items():
            if not st.get("active"):
                continue
            ratio = 0.0
            if mask is not None:
                ratio = compute_overlap_ratio(st["box"], mask, footprint=0.5, denom=self.occ_denom)
            self._track_samples.setdefault(tid, []).append({
                "ts": timestamp,
                "stationary": st.get("stationary", False),
                "box": st["box"],
                "overlap": ratio,
                "cls": st.get("cls", ""),
                "conf": st.get("conf", 0.0),
            })

    def decide(self):
        """视频结束后调用, 产出与 ViolationEngineV2 兼容格式的事件列表."""
        from .temporal_fusion import (
            fuse_light, intervals_from_flags, fuse_occupancy, tag_evidence,
        )
        from .decision import decide_violations
        from .intermediate_state import make_track

        # ②层: 灯态时序融合 + evidence 打标(unknown+遮挡 -> occluded, 恢复 review, D1)
        light_segments = fuse_light(self._light_obs, **self.fuse_kwargs)
        light_segments = tag_evidence(light_segments, self._occ_samples)

        # ②层: track 区间聚合
        tracks = []
        for tid, samples in self._track_samples.items():
            stat_flags = [(s["ts"], s["stationary"]) for s in samples]
            stat_intervals = intervals_from_flags(stat_flags)

            occ_samples = [(s["ts"], s["overlap"]) for s in samples]
            occ_intervals = fuse_occupancy(occ_samples, base_thr=0.0)

            tracks.append(make_track(
                track_id=tid,
                vehicle_class=samples[0].get("cls") if samples else None,
                stationary_intervals=stat_intervals,
                occupancy_intervals=occ_intervals,
            ))

        intermediate_state = {
            "light_segments": light_segments,
            "tracks": tracks,
        }

        # ③层: 区间代数判定
        raw_events = decide_violations(
            intermediate_state, self.overlap_thr, self.min_duration_s,
            self.min_persistent_green_run_s,
        )

        # 同 track 事件去重: 间隔 < gap 的合并 (复刻 V2 行为)
        raw_events = self._dedup(raw_events)

        # 映射回 CLI 期望的事件格式
        self.events = []
        for ev in raw_events:
            tid = ev["track_id"]
            samples = self._track_samples.get(tid, [])
            cls = samples[0].get("cls", "") if samples else ""
            conf = round(samples[0].get("conf", 0.0), 3) if samples else 0.0
            self.events.append({
                "event_id": len(self.events) + 1,
                "track_id": tid,
                "status": ev["status"],
                "start_ts": round(ev["start_s"], 2),
                "end_ts": round(ev["end_s"], 2),
                "vehicle_class": cls,
                "confidence": conf,
                "light_state": ev["light_state"],
                "evidence_image": "",
                "plate": "",
                "max_overlap": ev.get("max_overlap", 0.0),
                "member_tracks": ev.get("member_tracks", [tid]),
            })
        return self.events

    def _dedup(self, events):
        """全局时序合并: 把**时间重叠或间隔<gap 的事件跨 track 合并**为单个违章 episode
        (输出粒度 = 违章时间窗, 对齐 datasets/gt/events.csv 的 per-窗 GT)。

        修复(2026-07-16): 旧版仅按 track_id 分组合并 -> 碎片化的多 track(YOLO框抖动/遮挡把
        物理同车切成多 ID)在同一违章窗内各自成事件, 跨 track 永不合并 -> 端到端 Precision
        灾难(实测 26 FP vs 7 TP)。改为全局按时序合并。

        合并语义:
          - start=min, end=max(union 时间跨度)。
          - member_tracks = 并入的所有 track_id; 代表 track_id/light_state 取 max_overlap
            最大者(最显著违规车, 供车牌/证据回填)。
          - max_overlap 取最大; 任一成员为 review -> episode 记 review(安全侧优先)。
        """
        if not events:
            return []
        episodes = []
        cur = None
        for e in sorted(events, key=lambda x: x["start_s"]):
            if cur is not None and e["start_s"] - cur["end_s"] < self.gap:
                self._absorb(cur, e)
            else:
                cur = self._new_episode(e)
                episodes.append(cur)
        return episodes

    @staticmethod
    def _new_episode(e):
        ep = dict(e)
        ep["member_tracks"] = [e["track_id"]]
        return ep

    @staticmethod
    def _absorb(cur, e):
        cur["end_s"] = max(cur["end_s"], e["end_s"])
        if e["track_id"] not in cur["member_tracks"]:
            cur["member_tracks"].append(e["track_id"])
        # 代表 track/灯态 = 压线比例最大者(最显著违规车)
        if e.get("max_overlap", 0.0) > cur.get("max_overlap", 0.0):
            cur["track_id"] = e["track_id"]
            cur["light_state"] = e["light_state"]
        cur["max_overlap"] = max(cur.get("max_overlap", 0.0), e.get("max_overlap", 0.0))
        # review 优先级高于 confirmed(安全侧交人复核)
        if e["status"] == "review" or cur["status"] == "review":
            cur["status"] = "review"
