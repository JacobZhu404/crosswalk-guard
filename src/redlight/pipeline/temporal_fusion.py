"""L5 时序后处理 (②, 架构 spec §2/§4)。本期: 灯态融合 fuse_light。

消费逐帧观测序列 -> light_segments(纯函数, 批处理)。承接原
TrafficLightDetector._state_from_global + hysteresis 的语义, 但操作完整序列而非流式 deque。
时变红线(A-D2): unknown 段内单帧跳过; 连续 unknown 超阈开新 unknown 段, 不前填。
"""
from collections import deque
from .intermediate_state import (
    make_light_segment, merge_adjacent_segments, make_occupancy_interval,
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


def fuse_light(observations, window=24, hysteresis=0.68, flicker_toggle=4, unknown_hold=8):
    """observations: 有序 [(ts, obs, conf)], obs ∈ green|red|off|None。返回 light_segments。"""
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
    return merge_adjacent_segments(segs)


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
