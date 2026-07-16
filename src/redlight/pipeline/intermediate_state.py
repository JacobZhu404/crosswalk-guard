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


def enforce_transition_limit(segments, max_transitions=2, min_seg_dur=3.0,
                             conf_floor=0.85):
    """全局约束灯态段转换次数(强先验: ~2min视频最多2次红绿转换)。

    用高置信段兜底低置信/短碎段, 消除单帧识别抖动产生的 flashing/unknown 碎段。
    规则(迭代):
      1) unknown 段: 前后同色高置信 -> 填成该色(高置信兜底); 否则保留(真遮挡=review)。
      2) 短低置信段(时长<min_seg_dur 且 conf<conf_floor): 并入相邻更强段(时长×conf大者)。
      3) 若转换数仍 > max_transitions: 反复吸收"最弱段"(时长×conf最小)进邻段, 直到达标。
    高置信长段(如 red/green conf=1.0)是锚, 不会被吸收; 真实的长 flashing/长遮挡(时长
    ≥min_seg_dur)也因此保留, 仅短碎段(抖动)被折叠。

    输入/输出: light_segments(按时间有序)。纯函数。
    """
    if not segments:
        return segments
    segs = [dict(s) for s in segments]

    def _seg_strength(s):
        return (s["end_s"] - s["start_s"]) * max(s.get("conf", 0.0), 0.01)

    def _absorb(idx):
        """把 segs[idx] 吸收进相邻更强段(状态改为邻段状态), 然后合并同色。"""
        left = segs[idx - 1] if idx > 0 else None
        right = segs[idx + 1] if idx + 1 < len(segs) else None
        if left is None and right is None:
            return
        # 选更强的邻段的状态
        if left is None:
            tgt = right["state"]
        elif right is None:
            tgt = left["state"]
        else:
            tgt = left["state"] if _seg_strength(left) >= _seg_strength(right) else right["state"]
        segs[idx]["state"] = tgt

    # 步骤1: unknown 被前后同色高置信兜底
    for i, s in enumerate(segs):
        if s["state"] != "unknown":
            continue
        left = segs[i - 1] if i > 0 else None
        right = segs[i + 1] if i + 1 < len(segs) else None
        lc = left["state"] if left and left.get("conf", 0) >= conf_floor else None
        rc = right["state"] if right and right.get("conf", 0) >= conf_floor else None
        if lc and lc == rc and lc in ("green", "red", "flashing"):
            s["state"] = lc  # 前后同色高置信 -> 填充(如 red-unknown-red -> red)

    def _rebuild():
        return merge_adjacent_segments(segs)

    segs = _rebuild()

    # 步骤2: 吸收短低置信段
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(segs):
            dur = s["end_s"] - s["start_s"]
            if dur < min_seg_dur and s.get("conf", 0) < conf_floor and len(segs) > 1:
                _absorb(i)
                segs = merge_adjacent_segments(segs)
                changed = True
                break

    # 步骤3: 转换数仍超限 -> 反复吸收最弱段
    while len(segs) - 1 > max_transitions and len(segs) > 1:
        # 找最弱段(端点段只能被单邻吸收, 中间段选最弱)
        weakest = min(range(len(segs)), key=lambda i: _seg_strength(segs[i]))
        _absorb(weakest)
        segs = merge_adjacent_segments(segs)

    return segs


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
