"""BatchViolationEngine 端到端(②③ 接线): accumulate -> decide。纯 numpy, 无需 cv2。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.pipeline.violation_engine import BatchViolationEngine


def _states(tid, box, stationary=True):
    return {tid: {"active": True, "stationary": stationary, "box": list(box),
                  "cls": "car", "conf": 0.9}}


def _mask(occluded=False):
    m = np.zeros((400, 400), np.uint8)
    if occluded:
        m[100:400, 100:300] = 255   # 触底边 -> 遮挡
    else:
        m[100:200, 100:300] = 255
    return m


def test_batch_green_stationary_confirmed():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    for i in range(16):
        eng.accumulate(_states(1, (120, 100, 220, 200)), _mask(False),
                       {"obs": "green", "conf": 0.9}, i * 0.5)
    evs = eng.decide()
    conf = [e for e in evs if e["status"] == "confirmed"]
    assert len(conf) == 1 and conf[0]["track_id"] == 1 and conf[0]["light_state"] == "green"


def test_batch_unknown_occluded_review():
    """灯 unknown + 斑马线遮挡 + 车静止压线 -> review (D1, evidence 打标恢复)。"""
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    for i in range(16):
        eng.accumulate(_states(1, (120, 100, 220, 400)), _mask(occluded=True),
                       {"obs": "off", "conf": 0.0}, i * 0.5)   # obs off -> unknown 段
    evs = eng.decide()
    assert [e for e in evs if e["status"] == "review"]          # review 复活
    assert not [e for e in evs if e["status"] == "confirmed"]   # 未知不确认


def test_batch_red_no_event():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    for i in range(16):
        eng.accumulate(_states(1, (120, 100, 220, 200)), _mask(False),
                       {"obs": "red", "conf": 0.9}, i * 0.5)
    assert eng.decide() == []


def _states_multi(specs, stationary=True):
    """多 track 同帧: specs = {tid: box}。"""
    return {tid: {"active": True, "stationary": stationary, "box": list(box),
                  "cls": "car", "conf": 0.9} for tid, box in specs.items()}


def test_batch_fragmented_tracks_merge_to_one_episode():
    """碎片化: 同一违章窗内两个 track_id(物理同车被切开)都绿灯+静止+压线,
    应合并为**一个** confirmed episode(输出粒度=违章时间窗, 对齐 events.csv GT),
    而非各自成事件造成 FP。
    b2(2026-08-19) 后: member_tracks 收窄为**车组代表**(物理同车碎片 -> 1 个代表),
    原始全 member 保留在 member_tracks_all(车牌回填线用)。"""
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    for i in range(16):
        eng.accumulate(
            _states_multi({1: (120, 100, 220, 200), 2: (122, 101, 222, 201)}),
            _mask(False), {"obs": "green", "conf": 0.9}, i * 0.5)
    conf = [e for e in eng.decide() if e["status"] == "confirmed"]
    assert len(conf) == 1, f"跨 track 重叠事件应合并为1个, 实得 {len(conf)}"
    # b2 收窄: 同车碎片归一个车组 -> member_tracks 只剩代表; 全量仍保全
    assert len(conf[0]["member_tracks"]) == 1
    assert conf[0]["member_tracks"][0] in (1, 2)          # 代表 = 组内 max overlap 车
    assert set(conf[0]["member_tracks_all"]) == {1, 2}    # 车牌线原始全 member 不丢


# ---- _dedup 全局时序合并(直接测被改单元, 不走 fuse 平滑) ----

def _raw(tid, s, e, status="confirmed", ov=0.5, light="green"):
    return {"track_id": tid, "status": status, "start_s": s, "end_s": e,
            "light_state": light, "max_overlap": ov}


def test_dedup_merges_overlapping_cross_track():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)  # gap=5s 默认
    out = eng._dedup([_raw(1, 91.5, 110.9), _raw(2, 91.5, 97.0),
                      _raw(3, 94.0, 109.9), _raw(4, 110.6, 121.2)])
    assert len(out) == 1
    assert out[0]["start_s"] == 91.5 and out[0]["end_s"] == 121.2
    assert set(out[0]["member_tracks"]) == {1, 2, 3, 4}


def test_dedup_keeps_far_apart_events():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    # 间隔 >gap(5s): 8s 结束, 下一个 20s 开始 -> 不合并
    out = eng._dedup([_raw(1, 0.0, 8.0), _raw(2, 20.0, 30.0)])
    assert len(out) == 2


def test_dedup_representative_track_is_max_overlap():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    out = eng._dedup([_raw(1, 10, 20, ov=0.3), _raw(2, 11, 25, ov=0.8)])
    assert len(out) == 1
    assert out[0]["track_id"] == 2                 # 代表 track = 压线比例最大者(供车牌/证据)
    assert out[0]["max_overlap"] == 0.8


def test_dedup_review_priority_on_merge():
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    out = eng._dedup([_raw(1, 10, 20, status="confirmed"),
                      _raw(2, 12, 22, status="review", light="unknown")])
    assert len(out) == 1 and out[0]["status"] == "review"   # 安全侧: review 优先


# ================= b2 脏袋收窄: _narrow_members(2026-08-19, cc plan-gate PASS) =================

def _engine_map(track_map, b2_centroid_d=200.0, b2_gap_merge=3.0):
    """构造引擎并直接注入 track_samples(跳过 accumulate), 供 _narrow_members 单测。

    track_map: tid -> (box_x_center, 首帧ts, 末帧ts, sta_ratio, max_ov)
    samples: 每 tid 在 [t0,t1] 内每 0.5s 一帧, box 以 cx 为中心固定 100px 宽。
    """
    eng = BatchViolationEngine("balanced", sample_fps=8.0)
    eng.b2_centroid_d = b2_centroid_d
    eng.b2_gap_merge = b2_gap_merge
    eng._track_samples = {}
    for tid, (cx, t0, t1, sta, ov) in track_map.items():
        n = int(round((t1 - t0) / 0.5)) + 1
        eng._track_samples[tid] = [{
            "ts": t0 + 0.5 * i, "stationary": (i < int(round(sta * n))),
            "box": [cx - 50, 100, cx + 50, 200], "overlap": ov,
            "cls": "car", "conf": 0.9} for i in range(n)]
    return eng


def _ep(rep, members, w0=0.0, w1=20.0, status="confirmed", max_overlap=0.8):
    return {"track_id": rep, "member_tracks": list(members), "start_s": w0,
            "end_s": w1, "status": status, "max_overlap": max_overlap}


def test_narrow_same_car_fragments_merge_to_rep():
    """同车碎片(时序连续 + 空间相邻) -> 车组 -> member_tracks 只留组代表;
    member_tracks_all 保全原 member(车牌线)。代表 = 组内 max overlap 最高车。"""
    # tid1 [0,10] cx=400, tid11 [10.5,20] cx=405 (gap0.5<3s, 5px<200px): 物理同车
    eng = _engine_map({1: (400, 0.0, 10.0, 1.0, 0.5),
                       11: (405, 10.5, 20.0, 1.0, 0.9)})
    out = eng._narrow_members([_ep(11, [1, 11])])
    assert out[0]["track_id"] == 11                          # 窗口/代表不变
    assert out[0]["member_tracks"] == [11]                   # 收窄: 1 车组 1 代表(max ov=0.9)
    assert out[0]["member_tracks_all"] == [1, 11]            # 全量保全


def test_narrow_parallel_passing_group_excluded():
    """时序并行的空间远离车组(窗口内 stationary<0.6 的过路车) -> 移除。
    tid2 与 tid1 时序完全并行但 cx 差 600px(>200), 且 sta=0.3 -> 不入 member_tracks。"""
    eng = _engine_map({1: (400, 0.0, 20.0, 1.0, 0.8),
                       2: (1000, 0.0, 20.0, 0.3, 0.7)})
    out = eng._narrow_members([_ep(1, [1, 2])])
    assert out[0]["member_tracks"] == [1]
    assert out[0]["member_tracks_all"] == [1, 2]             # 车牌线仍需全量(span 绕行判断)
    assert out[0]["status"] == "confirmed"                   # 状态/窗口不变(Fix A 语义)


def test_parallel_stationary_multicar_kept():
    """真实多车违章(08 两白车/09 两车): 两个空间分离的**静止违章车组**都保留(双代表),
    不拆 episode(member_tracks=2 个代表, 窗口不变)。"""
    eng = _engine_map({1: (400, 0.0, 20.0, 1.0, 0.8),
                       2: (1500, 0.0, 20.0, 1.0, 0.6)})
    out = eng._narrow_members([_ep(1, [1, 2])])
    assert out[0]["member_tracks"] == [1, 2]     # 双违章车组均保留(按 overlap 降序)
    assert out[0]["end_s"] == 20.0                # 窗口未拆碎(硬条件③)


def test_narrow_transitive_chain_single_group():
    """时序连续传递闭包(union-find): tid1[0,5]→tid2[7,12](gap2<3) 并入一组,
    tid3[20,25](gap8>3) 拆出 -> 2 个代表, 不出现 3 代表(长链吞并)。"""
    eng = _engine_map({1: (400, 0.0, 5.0, 1.0, 0.5),
                       2: (410, 7.0, 12.0, 1.0, 0.6),
                       3: (420, 20.0, 25.0, 1.0, 0.7)})
    out = eng._narrow_members([_ep(2, [1, 2, 3], w0=0.0, w1=25.0)])
    # 组{1,2} 代表=2(max ov 0.6), 组{3} 代表=3 -> 2 个代表
    assert set(out[0]["member_tracks"]) == {2, 3}
    assert set(out[0]["member_tracks_all"]) == {1, 2, 3}   # 全量保全(车牌线)


def test_narrow_rep_always_present():
    """代表车组即使不满足 sta/ov 候选也强制保留(代表车=最显著违规车)。"""
    eng = _engine_map({1: (400, 0.0, 20.0, 0.2, 0.5),
                       2: (1000, 0.0, 20.0, 0.9, 0.6)})
    out = eng._narrow_members([_ep(1, [1, 2])])    # rep=1 低静止, tid2 高静止远车
    assert 1 in out[0]["member_tracks"]            # 代表恒在
    assert 2 in out[0]["member_tracks"]            # tid2 稳守组也留(候选车组)


def test_narrow_single_member_untouched():
    eng = _engine_map({1: (400, 0.0, 20.0, 1.0, 0.8)})
    out = eng._narrow_members([_ep(1, [1])])
    assert out[0]["member_tracks"] == [1]
    assert out[0]["member_tracks_all"] == [1]
