"""L5 中间态契约 (架构 spec §5): TemporalFusion 产出、判定层③/呈现④/评测消费。

本期只定义灯态部分(light_segments); tracks/occupancy/review_flags 待后续 TemporalFusion
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


def make_occupancy_interval(start_s, end_s, max_overlap, avg_overlap):
    """一段连续压线区间(overlap>0)。阈值由判定层③按 preset 施加, 本层不预筛(A-D3)。"""
    return {"start_s": start_s, "end_s": end_s,
            "max_overlap": max_overlap, "avg_overlap": avg_overlap}


def make_track(track_id, vehicle_class=None, plate=None,
               stationary_intervals=None, occupancy_intervals=None):
    """一条轨迹的中间态。

    时间不变: vehicle_class / plate(全局投票结果)。
    时变: stationary_intervals=[[s,e],...]、occupancy_intervals=[make_occupancy_interval...]。
    """
    return {"track_id": track_id, "vehicle_class": vehicle_class, "plate": plate,
            "stationary_intervals": list(stationary_intervals or []),
            "occupancy_intervals": list(occupancy_intervals or [])}
