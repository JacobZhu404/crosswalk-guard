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
                 min_persistent_green_run_s=6.0,
                 b2_centroid_d=200.0, b2_gap_merge=3.0):
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
        # b2 脏袋收窄(2026-08-19): member 归组阈值 —— 同车碎片=质心距<b2_centroid_d
        # 且时序重叠/相邻(|A.end-B.start|<b2_gap_merge); 过路车组从 member 移除
        # (episode 窗口/状态判定不变, 只收窄 member_tracks 供车牌回填, 零新增 FP)。
        self.b2_centroid_d = b2_centroid_d
        self.b2_gap_merge = b2_gap_merge
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

        # b2 脏袋收窄(2026-08-19): episode 窗口/状态判定不变(Fix A 语义保留, 零新增 FP),
        # 只收窄 member_tracks = 代表 track + 时空连续同车碎片(过路车组移除, 供车牌回填)。
        raw_events = self._narrow_members(raw_events)

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
                # b2(2026-08-19): member_tracks 已被收窄为车组代表(碎片化/事件成形指标);
                # member_tracks_all = _dedup 原始全 member, 专供车牌回填(_episode_plate_all),
                # 必须随事件透传 —— 否则车牌线回退到收窄集合, 会复活 05 京N541E6/08 京PK9B77
                # 等跨车污染误罚(收窄后质心 span 变小, 车牌线空间聚集约束失效)。
                "member_tracks": ev.get("member_tracks", [tid]),
                "member_tracks_all": ev.get("member_tracks_all", ev.get("member_tracks", [tid])),
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
        # #3 dedup 交互修复(Plan A, cc brief 4329e78 / cc ruling cf2e94e):
        # transient_green review(#3 瞬态绿)不得毒化同 episode 内证据充分的 confirmed 核;
        # 仅非瞬态(=D1 light_uncertain 或 legacy 未标注)review 保留 review 优先安全语义。
        # 否则 confirmed 核胜出(瞬态 review 被吸收)。
        e_poisons = (e["status"] == "review" and e.get("review_reason") != "transient_green")
        cur_poisons = (cur["status"] == "review" and cur.get("review_reason") != "transient_green")
        if e_poisons or cur_poisons:
            cur["status"] = "review"
        elif e["status"] == "confirmed" or cur["status"] == "confirmed":
            cur["status"] = "confirmed"
        # else: 两成员皆 transient_green review(无 confirmed 核) -> episode 保持 review(默认)
    def _narrow_members(self, episodes):
        """b2 脏袋收窄(2026-08-19, cc 任务单 2026-08-18-cc-task-qw-b2):
        把 episode 的 member_tracks 从"全部并入 track"收窄为"车组代表"。

        设计(cc 五条硬条件 + 车牌线信号互斥分析):
          - episode 窗口/状态判定不变(Fix A 保留), 不拆 episode(零新增 FP);
          - 车组归组: member tracks 按"互相时空连续"(质心距<b2_centroid_d ∧
            时序重叠/相邻 |A.end-B.start|<b2_gap_merge, union-find 传递闭包)聚成车组
            (物理同车碎片合并, 如 05 白车链 tid1→11→19→25/26);
          - 过路/漂移车组移除: 车组在事件窗口内 stationary 占比 < 0.6(非违章车);
          - member_tracks = 各保留车组的代表 track(max overlap);
          - **原始全 member 保留在 member_tracks_all**: 车牌回填(_episode_plate_all)
            用原始 member —— 因 b2 车组归组会消解"跨车关联污染"特征(05 京N541E6 归黑车组后
            质心 span 变小, 车牌线 span 绕行失效), 车牌线与 b2 收窄信号互斥, 故车牌线
            维持原 member + span 绕行不变(硬条件: 车牌误罚仍 0)。
        """
        for ep in episodes:
            members = ep.get("member_tracks", [])
            rep = ep["track_id"]
            ep["member_tracks_all"] = list(members)  # 原始全 member(车牌回填用)
            if len(members) <= 1:
                continue
            w0, w1 = ep["start_s"], ep["end_s"]
            # 每个 member track: 质心 + 全时跨度 + 窗口内 stationary 占比
            info = {}
            for tid in members:
                samples = self._track_samples.get(tid, [])
                if not samples:
                    continue
                cx = sum((s["box"][0] + s["box"][2]) / 2 for s in samples) / len(samples)
                cy = sum((s["box"][1] + s["box"][3]) / 2 for s in samples) / len(samples)
                win = [s for s in samples if w0 <= s["ts"] <= w1]
                sta = (sum(1 for s in win if s.get("stationary")) / len(win)) if win else 0.0
                ov = max((s.get("overlap", 0.0) for s in win), default=0.0)
                info[tid] = {"cx": cx, "cy": cy, "t0": samples[0]["ts"],
                             "t1": samples[-1]["ts"], "sta": sta, "ov": ov,
                             "n": len(samples)}
            if rep not in info:
                continue
            # union-find 车组归组(互相时空连续)
            parent = {tid: tid for tid in members if tid in info}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            tids = list(parent.keys())
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    a, b = info[tids[i]], info[tids[j]]
                    dist = ((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2) ** 0.5
                    temporal_ok = not (a["t1"] + self.b2_gap_merge < b["t0"]
                                       or b["t1"] + self.b2_gap_merge < a["t0"])
                    if dist < self.b2_centroid_d and temporal_ok:
                        union(tids[i], tids[j])
            # 车组 -> 成员列表
            groups = {}
            for tid in tids:
                groups.setdefault(find(tid), []).append(tid)
            # 保留违章车组(方案 §3.1: 窗口内 stationary>=0.6 ∧ overlap>0.15),
            # 代表 = max overlap; 代表车(=episode 代表 track)所在车组以其为代表(一车一代表);
            # 代表车组即使不满足候选也强制保留(代表车=最显著违规车)。
            kept_groups = []
            for root, gtids in groups.items():
                sta_max = max(info[t]["sta"] for t in gtids)
                ov_max = max(info[t]["ov"] for t in gtids)
                if (sta_max >= 0.6 and ov_max > 0.15) or rep in gtids:
                    rep_tid = max(gtids, key=lambda t: (info[t]["ov"], info[t]["n"]))
                    if rep in gtids:
                        rep_tid = rep          # 代表车所在车组 -> 以代表车为组代表
                    kept_groups.append((rep_tid, info[rep_tid]["ov"]))
            # member_tracks = 车组代表(按 overlap 降序), 代表 track 恒在
            kept_groups.sort(key=lambda x: -x[1])
            narrowed = [t for t, _ in kept_groups]
            if rep not in narrowed:
                narrowed.insert(0, rep)
            ep["member_tracks"] = narrowed
        return episodes
