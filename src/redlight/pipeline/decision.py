"""L5 判定层 (③, 架构 spec §2/§4 A-D3): 消费中间态, 区间交集代数出违规事件。

违规(E12) = 行人绿灯/闪烁段 ∩ 车辆静止区间 ∩ 压线区间(max_overlap≥preset) 且交集时长≥duration。
灯态未知+遮挡段 ∩ 静止 ∩ 压线 -> review(D1)。**overlap 阈值在本层施加**(② 只按 base_thr 过噪)。
纯函数: 输入 intermediate_state dict, 输出 events; 不含流式状态机(替代 violation_engine 的批处理版)。
"""
from .temporal_fusion import interval_intersect


def _go_intervals(segs):
    return [[s["start_s"], s["end_s"]] for s in segs if s["state"] in ("green", "flashing")]


def _review_light_intervals(segs):
    # unknown 且 灯存在但看不到(遮挡)/上下文推断 -> review 侧
    return [[s["start_s"], s["end_s"]] for s in segs
            if s["state"] == "unknown" and s.get("evidence") in ("occluded", "inferred")]


def _state_at(segs, t):
    for s in segs:
        if s["start_s"] <= t < s["end_s"]:
            return s["state"]
    return segs[-1]["state"] if segs else "unknown"


def _peak_overlap(track, s, e):
    peak = 0.0
    for o in track.get("occupancy_intervals", []):
        if min(o["end_s"], e) > max(o["start_s"], s):     # 与 [s,e] 有交
            peak = max(peak, o["max_overlap"])
    return round(peak, 3)


def decide_violations(state, overlap_thr, min_duration_s):
    """intermediate_state -> [event]. event: {track_id,status,start_s,end_s,light_state,max_overlap}。"""
    segs = state.get("light_segments", [])
    go = _go_intervals(segs)
    review_light = _review_light_intervals(segs)
    events = []
    for tr in state.get("tracks", []):
        stat = tr.get("stationary_intervals", [])
        occ = [[o["start_s"], o["end_s"]] for o in tr.get("occupancy_intervals", [])
               if o["max_overlap"] >= overlap_thr]
        if not stat or not occ:
            continue
        base = interval_intersect(stat, occ)              # 静止 ∩ 压线(达阈)
        for light_ivs, status in ((go, "confirmed"), (review_light, "review")):
            for s, e in interval_intersect(light_ivs, base):
                if e - s >= min_duration_s:
                    events.append({
                        "track_id": tr["track_id"], "status": status,
                        "start_s": s, "end_s": e,
                        "light_state": _state_at(segs, s),
                        "max_overlap": _peak_overlap(tr, s, e),
                    })
    return events
