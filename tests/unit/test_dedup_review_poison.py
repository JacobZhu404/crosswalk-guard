"""Plan A (_dedup review 毒化修复) 语义单测.

验证 BatchViolationEngine._dedup / _absorb 在 #3 引入 transient_green review 后,
不再毒化同 episode 内的 confirmed 核; 同时保留 D1(light_uncertain) review 的
review 优先安全语义.

依据: cc brief 4329e78 + cc ruling cf2e94e(授权 wb 落 Fix A, 保守标签版待 Jacob).
验收锚: 09 confirmed 命中[11-72] / 01 仍 0 FP / 全11 F1>=0.941/P=1.000.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from redlight.pipeline.violation_engine import BatchViolationEngine


def _ev(track_id, status, start_s, end_s, review_reason=None, max_overlap=1.0):
    ev = {
        "track_id": track_id,
        "status": status,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "light_state": "green",
        "max_overlap": max_overlap,
    }
    if review_reason is not None:
        ev["review_reason"] = review_reason
    return ev


def _dedup(events, gap=5.0):
    eng = BatchViolationEngine(min_event_gap_sec=gap)
    return eng._dedup(events)


def test_09_confirmed_absorbs_transient_green_review():
    # 09 真况: 1 瞬态绿 review(tid5 [0.67,4.18]) + 多 confirmed(压 55.9s 铁绿),
    # gap=3.23<5 合并 -> 整 episode 应为 confirmed 且跨含 [11-72].
    events = [
        _ev(5, "review", 0.67, 4.18, review_reason="transient_green"),
        _ev(19, "confirmed", 7.41, 30.0),
        _ev(19, "confirmed", 31.0, 60.0),
        _ev(7, "confirmed", 61.0, 106.26),
    ]
    eps = _dedup(events)
    assert len(eps) == 1, eps
    assert eps[0]["status"] == "confirmed", eps[0]
    assert eps[0]["start_s"] == 0.67 and eps[0]["end_s"] == 106.26
    assert eps[0]["end_s"] >= 72.0  # 跨含 GT[11-72]


def test_01_pure_transient_green_review_stays_review():
    # 01 硬约束: 纯瞬态绿 review, 无 confirmed 核 -> 仍 review(01 FP 不复活).
    events = [_ev(3, "review", 48.36, 61.29, review_reason="transient_green")]
    eps = _dedup(events)
    assert len(eps) == 1
    assert eps[0]["status"] == "review"


def test_light_uncertain_review_poisons_confirmed():
    # D1 安全语义: light_uncertain review + confirmed -> 仍 review(保留旧优先级).
    events = [
        _ev(9, "confirmed", 10.0, 40.0),
        _ev(9, "review", 41.0, 45.0, review_reason="light_uncertain"),
    ]
    eps = _dedup(events)
    assert len(eps) == 1
    assert eps[0]["status"] == "review", "D1 review 必须保留 review 优先"


def test_legacy_untagged_review_poisons():
    # 向后兼容: 未打 review_reason 标签的 review 仍视为毒化(review 优先).
    events = [
        _ev(9, "confirmed", 10.0, 40.0),
        _ev(9, "review", 41.0, 45.0),  # 无 review_reason
    ]
    eps = _dedup(events)
    assert eps[0]["status"] == "review"


def test_order_independent_confirmed_wins_over_transient():
    # confirmed 先于 transient review: 仍 confirmed(核胜出不依赖顺序).
    events = [
        _ev(19, "confirmed", 7.41, 60.0),
        _ev(5, "review", 61.0, 63.0, review_reason="transient_green"),
    ]
    eps = _dedup(events)
    assert len(eps) == 1
    assert eps[0]["status"] == "confirmed"


def test_all_transient_review_no_confirmed_core_stays_review():
    # 全为瞬态绿 review(无 confirmed 成员) -> episode 保持 review(默认, 不误升).
    events = [
        _ev(3, "review", 48.36, 61.29, review_reason="transient_green"),
        _ev(4, "review", 62.0, 64.0, review_reason="transient_green"),
    ]
    eps = _dedup(events)
    assert len(eps) == 1
    assert eps[0]["status"] == "review"


# ---------------------------------------------------------------------------
# 端到端决策层接线(cicd 无需视频/权重): decide_violations(review_reason) -> _dedup
# 复刻 decide() 的 决策层 部分(accumulate/temporal_fusion 在决策层之前),
# 证明 review_reason 标签从生成到合并消费整条链路打通。
# ---------------------------------------------------------------------------
def _track(track_id, stat, occ, max_overlap=1.0):
    return {
        "track_id": track_id,
        "vehicle_class": "car",
        "stationary_intervals": list(stat),
        "occupancy_intervals": [{"start_s": s, "end_s": e, "max_overlap": max_overlap}
                                 for (s, e) in occ],
    }


def _e2e(light_segments, tracks, min_run=6.0):
    from redlight.pipeline.decision import decide_violations
    state = {"light_segments": light_segments, "tracks": tracks}
    raw = decide_violations(state, overlap_thr=0.5, min_duration_s=0.5,
                            min_persistent_green_run_s=min_run)
    return BatchViolationEngine(min_event_gap_sec=5)._dedup(raw)


def test_e2e_09_recovered_to_confirmed():
    # 09 真况: [0,4.18] 瞬态绿(0.94s) + [7.41,106.26] 55.9s 持久绿(GT[11-72]在内)
    light_segments = [
        {"start_s": 0.0, "end_s": 4.18, "state": "green", "max_raw_green_run_s": 0.943, "evidence": "visible"},
        {"start_s": 4.18, "end_s": 7.41, "state": "red", "max_raw_green_run_s": 0.0, "evidence": "visible"},
        {"start_s": 7.41, "end_s": 106.26, "state": "green", "max_raw_green_run_s": 55.892, "evidence": "visible"},
    ]
    tracks = [
        _track(19, [(7.41, 106.26)], [(0.0, 106.26)]),   # 持久绿上停车占道 -> confirmed
        _track(5, [(0.0, 4.18)], [(0.0, 4.18)]),          # 瞬态绿上停车 -> transient_green review
    ]
    eps = _e2e(light_segments, tracks)
    assert len(eps) == 1, eps
    assert eps[0]["status"] == "confirmed"
    assert eps[0]["start_s"] <= 11.0 and eps[0]["end_s"] >= 72.0  # 跨含 GT[11-72]


def test_e2e_01_stays_review_zero_fp():
    # 01 硬约束: 仅 3.10s 瞬态绿, 无持久绿 -> 纯 transient_green review, 0 confirmed
    light_segments = [
        {"start_s": 0.0, "end_s": 48.36, "state": "red", "max_raw_green_run_s": 0.0, "evidence": "visible"},
        {"start_s": 48.36, "end_s": 61.29, "state": "green", "max_raw_green_run_s": 3.10, "evidence": "visible"},
        {"start_s": 61.29, "end_s": 106.0, "state": "red", "max_raw_green_run_s": 0.0, "evidence": "visible"},
    ]
    tracks = [_track(3, [(48.36, 61.29)], [(48.36, 61.29)])]
    eps = _e2e(light_segments, tracks)
    assert len(eps) == 1
    assert eps[0]["status"] == "review"
    assert not any(e["status"] == "confirmed" for e in eps)  # 01 FP 不复活


def test_e2e_review_reason_tagged():
    # 决策层确实在 review 事件上打了 review_reason 标签
    light_segments = [
        {"start_s": 0.0, "end_s": 4.18, "state": "green", "max_raw_green_run_s": 0.943, "evidence": "visible"},
    ]
    tracks = [_track(5, [(0.0, 4.18)], [(0.0, 4.18)])]
    from redlight.pipeline.decision import decide_violations
    raw = decide_violations({"light_segments": light_segments, "tracks": tracks},
                            overlap_thr=0.5, min_duration_s=0.5, min_persistent_green_run_s=6.0)
    assert any(r.get("review_reason") == "transient_green" for r in raw if r["status"] == "review")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
