"""L5 判定层 (③, 架构 spec §2/§4 A-D3): 消费中间态, 区间交集代数出违规事件。

违规(E12) = 行人绿灯/闪烁段 ∩ 车辆静止区间 ∩ 压线区间(max_overlap≥preset) 且交集时长≥duration。
灯态未知+遮挡段 ∩ 静止 ∩ 压线 -> review(D1)。**overlap 阈值在本层施加**(② 只按 base_thr 过噪)。
纯函数: 输入 intermediate_state dict, 输出 events; 不含流式状态机(替代 violation_engine 的批处理版)。
"""
from .temporal_fusion import interval_intersect


def _go_intervals(segs, min_run=10.0):
    # #3 时序门控 (plan-gate #5, 6124470/a9c48a8): 仅保留「段内最长 raw 绿 run >= min_run」
    # 的持久绿 -> confirmed。瞬态绿(过路车/反光瞬态误绿, 段内最长 raw run < min_run)排除,
    # 改走 _transient_green_intervals -> review(非 confirmed)。缺省 max_raw_green_run_s=1e9
    # (旧段/未标注)视为持久, 不门控(向后兼容)。
    out = []
    for s in segs:
        if s["state"] in ("green", "flashing"):
            if s.get("max_raw_green_run_s", 1e9) >= min_run:
                out.append([s["start_s"], s["end_s"]])
    return out


def _transient_green_intervals(segs, min_run):
    """#3 时序门控: 绿/闪烁段但其段内最长 raw 绿 run < min_run(瞬态) -> review(非 confirmed)。
    瞬态=过路车/反光瞬态误绿; 降级 review 而非硬杀, 最坏进人工队列不静默丢绿。"""
    out = []
    for s in segs:
        if s["state"] in ("green", "flashing"):
            mr = s.get("max_raw_green_run_s", 1e9)
            if mr < min_run:
                out.append([s["start_s"], s["end_s"]])
    return out


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


def decide_violations(state, overlap_thr, min_duration_s, min_persistent_green_run_s=10.0):
    """intermediate_state -> [event]. event: {track_id,status,start_s,end_s,light_state,max_overlap}。

    时序门控 (#3, plan-gate #5): 绿段按段内最长 raw 绿 run 分桶 ——
      持久(>=min_persistent_green_run_s) -> confirmed; 瞬态(<T) -> review。
    瞬态降级 review 而非硬杀: 最坏进人工队列, 不静默丢绿(防 ped_signal.pt 屠真绿重演)。
    """
    segs = state.get("light_segments", [])
    go = _go_intervals(segs, min_persistent_green_run_s)
    review_light = _review_light_intervals(segs)
    transient_green = _transient_green_intervals(segs, min_persistent_green_run_s)
    events = []
    for tr in state.get("tracks", []):
        stat = tr.get("stationary_intervals", [])
        occ = [[o["start_s"], o["end_s"]] for o in tr.get("occupancy_intervals", [])
                if o["max_overlap"] >= overlap_thr]
        if not stat or not occ:
            continue
        base = interval_intersect(stat, occ)              # 静止 ∩ 压线(达阈)
        for light_ivs, status in ((go, "confirmed"), (review_light, "review"),
                                  (transient_green, "review")):
            for s, e in interval_intersect(light_ivs, base):
                if e - s >= min_duration_s:
                    events.append({
                        "track_id": tr["track_id"], "status": status,
                        "start_s": s, "end_s": e,
                        "light_state": _state_at(segs, s),
                        "max_overlap": _peak_overlap(tr, s, e),
                    })
    return events
