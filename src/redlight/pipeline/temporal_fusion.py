"""L5 时序后处理 (②, 架构 spec §2/§4)。本期: 灯态融合 fuse_light。

消费逐帧观测序列 -> light_segments(纯函数, 批处理)。承接原
TrafficLightDetector._state_from_global + hysteresis 的语义, 但操作完整序列而非流式 deque。
时变红线(A-D2): unknown 段内单帧跳过; 连续 unknown 超阈开新 unknown 段, 不前填。
"""
from collections import deque
from .intermediate_state import (
    make_light_segment, merge_adjacent_segments, make_occupancy_interval,
    enforce_transition_limit,
)


def _window_state(win, flicker_toggle):
    """窗口(green/red/off 的最近序列)-> (state, conf)。移植 _state_from_global。"""
    seq = [c for c in win if c in ("green", "red")]
    green = seq.count("green")
    red = seq.count("red")
    seen = green + red
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


def _raw_green_runs(observations):
    """从逐帧 obs 序列提取连续 green run [(start_ts, end_ts), ...] (含单帧 run, 时长0)。
    口径对齐 diag_temporal_separability._axis_a_green_runs 的 max_green_run_s:
    run 时长 = 末帧 ts - 首帧 ts。#3 时序门控(plan-gate #5, 6124470/a9c48a8)据此判瞬态。"""
    runs = []
    i, n = 0, len(observations)
    while i < n:
        ts, obs, _c = observations[i]
        if obs == "green":
            j = i
            while j < n and observations[j][1] == "green":
                j += 1
            runs.append((observations[i][0], observations[j - 1][0]))
            i = j
        else:
            i += 1
    return runs


def _max_run_in_segment(runs, s, e):
    """返回与 [s,e] 相交的 raw 绿 run 中, 落在段内的最长时长(秒)。无则 0.0。"""
    best = 0.0
    for rs, re in runs:
        if re >= s and rs <= e:        # 与段相交
            ov = min(re, e) - max(rs, s)
            if ov > best:
                best = ov
    return round(best, 3)


def fuse_light(observations, window=24, hysteresis=0.68, flicker_toggle=4, unknown_hold=8,
               max_transitions=2, transition_min_dur=3.0):
    """observations: 有序 [(ts, obs, conf)], obs ∈ green|red|off|None。返回 light_segments。

    max_transitions: 全局约束灯态转换次数(强先验: ~2min视频最多2次转换)。用高置信段兜底
      低置信/unknown碎段, 消除单帧抖动产生的 flashing/unknown 碎段。<=0 时不约束(旧行为)。
    """
    win = deque(maxlen=window)
    raw = []                       # 每帧 (ts, committed_state, conf)
    committed = None
    unknown_run = 0
    for ts, obs, _c in observations:
        win.append(obs if obs in ("green", "red") else "off")
        st, conf = _window_state(win, flicker_toggle)
        if st == "unknown":
            unknown_run += 1
            if unknown_run <= unknown_hold and committed is not None:
                raw.append((ts, committed, conf))       # 段内短 unknown: 维持已提交
            else:
                committed = "unknown"                    # 超阈: 真开 unknown 段
                raw.append((ts, "unknown", conf))
            continue
        unknown_run = 0
        if st == "flashing":
            committed = "flashing"          # 闪烁是时序模式(conf≈0.5), 不受 green↔red 迟滞门控
        elif committed is None or st == committed or conf >= hysteresis:
            # 迟滞: 仅 green↔red 翻转需反色占比达 hysteresis
            committed = st
        raw.append((ts, committed, conf))
    segs = []
    for i, (ts, state, conf) in enumerate(raw):
        end = raw[i + 1][0] if i + 1 < len(raw) else ts
        segs.append(make_light_segment(ts, end, state, conf))
    segs = merge_adjacent_segments(segs)
    # 全局后处理: 转换次数约束 + 高置信兜底(消除单帧抖动碎段)
    if max_transitions is not None and max_transitions > 0:
        segs = enforce_transition_limit(segs, max_transitions=max_transitions,
                                        min_seg_dur=transition_min_dur)
    # #3 时序门控标注 (plan-gate #5, 6124470/a9c48a8): 给绿/闪烁段标注段内最长连续 raw 绿 run
    # (融合前逐帧 obs, 非融合后总时长)。decide_violations 据此对瞬态绿(< min_persistent_green_run_s)
    # 降级 review。判别量口径与 diag_temporal_separability 一致: run 时长 = 末帧ts - 首帧ts。
    green_runs = _raw_green_runs(observations)
    for seg in segs:
        if seg["state"] in ("green", "flashing"):
            seg["max_raw_green_run_s"] = _max_run_in_segment(green_runs, seg["start_s"], seg["end_s"])
        else:
            seg["max_raw_green_run_s"] = 0.0
    return segs


# ---- 时变区间聚合(② tracks/occupancy, 供③区间代数) ----
def intervals_from_flags(samples):
    """samples=[(ts, bool)] -> [[start_s, end_s], ...] 每段极大 True 连续段。

    用于静止区间(stationary): 把逐帧静止布尔序列折成区间。end=段内最后一个 True 的 ts。
    """
    out = []
    run_start = None
    last_true = None
    for ts, flag in samples:
        if flag:
            if run_start is None:
                run_start = ts
            last_true = ts
        else:
            if run_start is not None:
                out.append([run_start, last_true])
                run_start = None
    if run_start is not None:
        out.append([run_start, last_true])
    return out


def fuse_occupancy(samples, base_thr=0.0):
    """samples=[(ts, overlap)] -> [occupancy_interval, ...] 每段 overlap>base_thr 连续区间。

    支持断续压线(越线-回退-再越线 -> 多段)。每段带 max_overlap / avg_overlap。
    base_thr 仅过滤极小噪声; 判定阈值(preset overlap)留在判定层③施加(A-D3)。
    """
    out = []
    run = []          # [(ts, overlap), ...]
    for ts, ov in samples:
        if ov > base_thr:
            run.append((ts, ov))
        else:
            if run:
                out.append(_occ(run))
                run = []
    if run:
        out.append(_occ(run))
    return out


def _occ(run):
    ovs = [ov for _t, ov in run]
    return make_occupancy_interval(run[0][0], run[-1][0],
                                   round(max(ovs), 3), round(sum(ovs) / len(ovs), 3))


def tag_evidence(segments, occ_samples):
    """给 light_segments 打 evidence(恢复 decide 的 review 分支, D1)。

    有色段(green/red/flashing) -> "visible"(直接看到灯色);
    unknown 段: 若其时段被遮挡(occ_samples) -> "occluded"(灯存在但被挡, 交 review), 否则保持 None。
    occ_samples: [(ts, occluded_bool)](由 BatchViolationEngine.accumulate 用 _is_occluded 逐帧记录)。
    """
    occ_intervals = intervals_from_flags(occ_samples)
    out = []
    for s in segments:
        seg = dict(s)
        if seg["state"] in ("green", "red", "flashing"):
            seg["evidence"] = "visible"
        elif seg["state"] == "unknown" and _overlaps_any(seg["start_s"], seg["end_s"], occ_intervals):
            seg["evidence"] = "occluded"
        out.append(seg)
    return out


def _overlaps_any(s, e, intervals):
    return any(b >= s and a <= e for a, b in intervals)   # 闭区间相交(含零长段)


def interval_intersect(a_list, b_list):
    """两组区间 [[s,e],...] 的交集(供判定层③做 绿段∩静止∩压线)。零长(相切)不计。"""
    out = []
    for a in a_list:
        for b in b_list:
            s, e = max(a[0], b[0]), min(a[1], b[1])
            if e > s:
                out.append([s, e])
    out.sort()
    return out
