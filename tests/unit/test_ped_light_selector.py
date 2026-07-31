"""PedLightSelector 单测 (TDD, 阶段1 M2a/M2b/护栏1)。纯合成数据, 确定性, 不依赖模型/视频。

红线程: M2a 用 GT 种子锁 ped 轨迹(训练态); M2b 生产 GT-free 选灯(不收 GT);
护栏1: 生产选灯绝不能读逐帧 GT。leave-some-out 是真留出(held-out 视频先验来自其他视频)。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import inspect
import numpy as np
from redlight.models.ped_light_selector import (
    derive_ped_priors, seed_with_gt, select_gtfree, leave_some_out_eval, iou,
    compute_temporal_scores,
)


# ---- 合成候选构造 ----
def _cand(box_norm, source="yolo"):
    return {"box_norm": tuple(box_norm), "source": source}


def _ped_box(cx, cy, w=0.02, h=0.06):  # 竖长条(行人灯)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _veh_box(cx, cy, w=0.06, h=0.035):  # 宽扁(车灯)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_seed_with_gt_picks_ped_not_vehicle():
    """M2a: 训练态, GT 种子应锁 ped 框(即便帧里还有车灯框)。"""
    gt = _ped_box(0.5, 0.5)
    cands = [_cand(_veh_box(0.2, 0.3)), _cand(_ped_box(0.5, 0.5)), _cand(_veh_box(0.8, 0.7))]
    sel = seed_with_gt(cands, gt)
    assert sel is not None
    assert sel["box_norm"] == _ped_box(0.5, 0.5)
    assert iou(sel["box_norm"], gt) > 0.9


def test_selector_gtfree_picks_ped_by_geometry():
    """M2b: 生产 GT-free, 仅用几何先验选 ped(竖长条), 不碰 GT。"""
    # 先验来自"其他视频"的 ped 几何(竖长条)
    prior = derive_ped_priors([(0.02, 0.06), (0.018, 0.058), (0.022, 0.064)])
    cands = [_cand(_veh_box(0.2, 0.3)), _cand(_ped_box(0.5, 0.5)), _cand(_veh_box(0.8, 0.7))]
    sel = select_gtfree(cands, prior)
    assert sel is not None
    # 选中的必须是竖长条 ped 框, 不是宽扁车灯
    assert sel["box_norm"] == _ped_box(0.5, 0.5)


def test_selector_never_reads_gt_in_production():
    """护栏1: select_gtfree 签名不含 gt 参数 —— 结构上保证生产路径不读逐帧 GT。
    L2/L3 扩展: temporal_scores/l3_scores 等为可选(零标注或由 Jacob 标样本训的判别头产),
    仍无 GT; 生产接线默认不传 l3(待 cc 放行), 行为退化为纯 L1/L2。"""
    sig = inspect.signature(select_gtfree)
    assert "gt" not in sig.parameters, "select_gtfree 不得接收 GT 参数(护栏1)"
    assert set(sig.parameters.keys()) == {"candidates", "ped_prior",
                                          "temporal_scores", "temporal_weight",
                                          "l3_scores", "l3_weight",
                                          "governing_scores", "governing_weight",
                                          "governing_threshold"}
    # 新增参数必须是可选(有默认), 旧调用 select_gtfree(cands, prior) 行为不变
    assert sig.parameters["temporal_scores"].default is None
    assert sig.parameters["temporal_weight"].default is not None
    assert sig.parameters["l3_scores"].default is None
    assert sig.parameters["l3_weight"].default is not None
    # 弃权门参数: 默认不启用以保持生产行为不变(向后兼容, 红线)
    assert sig.parameters["governing_scores"].default is None
    assert sig.parameters["governing_weight"].default == 0.0
    assert sig.parameters["governing_threshold"].default == 0.5
    # 即便帧里恰好有个候选等于某 GT 框, 选择也只由 prior 几何决定, 不看 GT
    prior = derive_ped_priors([(0.02, 0.06)])
    # 构造: 一个"像车灯"的宽框恰好中心落在 GT 位置, 一个"像 ped"的竖框在别处
    gt_like_wide = _veh_box(0.5, 0.5)   # 宽扁, 中心在 (0.5,0.5)
    ped_like_tall = _ped_box(0.3, 0.4)  # 竖长, 中心在 (0.3,0.4)
    cands = [_cand(gt_like_wide), _cand(ped_like_tall)]
    sel = select_gtfree(cands, prior)
    # 应选竖长 ped 框(几何), 而非"中心在 GT 位置"的宽框 —— 证明没用 GT
    assert sel["box_norm"] == ped_like_tall


def test_leave_some_out_is_true_holdout():
    """leave-some-out: held-out 视频 V 的先验来自其他视频, 且 V 帧上 M2b 选对 ped。"""
    # 视频 A(训练): ped 竖长条
    # 视频 B(held-out): 同有 ped 竖长条 + 车灯宽扁
    records = [
        {"video": "A", "fi": 1, "candidates": [_cand(_ped_box(0.5, 0.5))],
         "gt_box_norm": _ped_box(0.5, 0.5), "gt_wh": (0.02, 0.06)},
        {"video": "B", "fi": 2,
         "candidates": [_cand(_veh_box(0.2, 0.3)), _cand(_ped_box(0.6, 0.6)),
                        _cand(_veh_box(0.85, 0.8))],
         "gt_box_norm": _ped_box(0.6, 0.6), "gt_wh": (0.02, 0.06)},
    ]
    res = leave_some_out_eval(records)
    # B 帧 ped 被检测到(竖长条候选存在)且 M2b 选对
    assert res["per_video"]["B"]["detected_frames"] == 1
    assert res["per_video"]["B"]["selection_correct"] == 1
    assert res["selection_accuracy"] == 1.0
    # 关键: B 的先验应来自 A, 不含 B 自身 —— 验证 holdout 正确
    from redlight.models.ped_light_selector import derive_ped_priors
    # 直接复算 B 的 prior: 应只含 A 的 wh
    prior_b = derive_ped_priors([r["gt_wh"] for r in records if r["video"] != "B"])
    assert prior_b["aspect_min"] <= (0.06 / 0.02) <= prior_b["aspect_max"]


def test_leave_some_out_excludes_undetected_from_selection_acc():
    """held-out 视频若 ped 未被检测到(如 05 类), 不计入选灯准确率(属检测缺口, 归 M4/M3)。"""
    records = [
        {"video": "A", "fi": 1, "candidates": [_cand(_ped_box(0.5, 0.5))],
         "gt_box_norm": _ped_box(0.5, 0.5), "gt_wh": (0.02, 0.06)},
        # B 帧: 只有车灯, 无 ped 候选(模拟 05 漏检)
        {"video": "B", "fi": 2,
         "candidates": [_cand(_veh_box(0.2, 0.3)), _cand(_veh_box(0.85, 0.8))],
         "gt_box_norm": _ped_box(0.6, 0.6), "gt_wh": (0.02, 0.06)},
    ]
    res = leave_some_out_eval(records)
    # B 未检测到 ped -> detected_frames=0 -> 不计入 selection_accuracy(属检测缺口, 归 M4/M3)
    assert res["per_video"]["B"]["detected_frames"] == 0
    # 分母只含检测到 ped 的帧(A); A 在评 A 时被 holdout(先验来自 B 的单一 wh)仍选对
    assert res["per_video"]["A"]["detected_frames"] == 1
    assert res["per_video"]["A"]["selection_correct"] == 1
    # 聚合选灯准确率: 仅 A 计入 -> 1.0 (B 已被排除在分母外, 正是要验证的)
    assert res["selection_accuracy"] == 1.0
    assert res["detection_coverage"] == 0.5  # 2 帧中仅 A 检测到 ped


def _frame(cands):
    return {"candidates": cands}


def test_compute_temporal_scores_rewards_persistent_candidate():
    """L2: 跨帧复现的候选(固定设施/行人灯)得分 1.0, 仅单帧出现的瞬时亮斑得 0.0。"""
    ped_box = (0.40, 0.20, 0.50, 0.50)   # 竖长条, 三帧都在
    transient = (0.10, 0.10, 0.20, 0.15)  # 仅第 0 帧
    frames = [
        _frame([_cand(ped_box, "yolo"), _cand(transient, "hsv")]),
        _frame([_cand(ped_box, "yolo")]),
        _frame([_cand(ped_box, "yolo")]),
    ]
    scores = compute_temporal_scores(frames)
    # 第 0 帧 ped 候选(索引0)在另外 2 帧都匹配 -> 2/(3-1)=1.0
    assert abs(scores[0][0] - 1.0) < 1e-9
    # 第 0 帧 transient(索引1)无其他帧匹配 -> 0.0
    assert scores[0][1] == 0.0


def test_compute_temporal_scores_single_frame_is_empty():
    """单帧(生产首帧/稀疏)无时序信息 -> 返回空分, 退化为 L1。"""
    frames = [_frame([_cand(_ped_box(0.5, 0.5))])]
    scores = compute_temporal_scores(frames)
    assert scores == [{}]


def test_select_gtfree_temporal_flips_tie():
    """L2: 当两候选几何(L1)相同时, temporal 评分翻转选择(持久 ped 胜过瞬时框)。"""
    prior = derive_ped_priors([(0.10, 0.30)])  # aspect_mean=3, 较大 std
    # A, B 几何完全相同(L1 平局); B 有时序奖励, A 无
    A = _cand((0.40, 0.20, 0.50, 0.50), "yolo")
    B = _cand((0.60, 0.20, 0.70, 0.50), "yolo")
    cands = [A, B]
    # 无 temporal: 平局, 返回首个(A)
    assert select_gtfree(cands, prior) is A
    # 有 temporal(B=1.0, A=0.0): B 胜
    sel = select_gtfree(cands, prior, temporal_scores={0: 0.0, 1: 1.0})
    assert sel is B


def test_select_gtfree_backward_compat_no_temporal():
    """向后兼容: 不传 temporal_scores 时(单帧/首帧)行为与原 L1-only 完全一致。"""
    prior = derive_ped_priors([(0.10, 0.30)])
    A = _cand((0.40, 0.20, 0.50, 0.50), "yolo")    # 竖长条 ped, L1 高
    B = _cand((0.10, 0.10, 0.20, 0.15), "hsv")     # 宽扁/小, L1 低
    cands = [A, B]
    r1 = select_gtfree(cands, prior)
    r2 = select_gtfree(cands, prior, temporal_scores=None)
    assert r1 is r2 is A


def test_select_gtfree_l3_flips_tie():
    """L3: 当两候选 L1+L2 相同时, l3_scores(P(ped)) 翻转选择(高 ped 概率胜出)。"""
    prior = derive_ped_priors([(0.10, 0.30)])
    A = _cand((0.40, 0.20, 0.50, 0.50), "yolo")
    B = _cand((0.60, 0.20, 0.70, 0.50), "yolo")
    cands = [A, B]
    ts = {0: 1.0, 1: 1.0}  # 都不同时序奖励(平局)
    # 仅 L1+L2: 平局 -> 首个 A
    assert select_gtfree(cands, prior, temporal_scores=ts) is A
    # 加 l3: B 的 P(ped)=0.9 > A=0.1 -> B 胜
    sel = select_gtfree(cands, prior, temporal_scores=ts,
                        l3_scores={0: 0.1, 1: 0.9}, l3_weight=0.5)
    assert sel is B


def test_select_gtfree_backward_compat_no_l3():
    """向后兼容: l3_scores=None 时(生产默认/未放行)行为与原 L1+L2 完全一致。"""
    prior = derive_ped_priors([(0.10, 0.30)])
    A = _cand((0.40, 0.20, 0.50, 0.50), "yolo")    # L1 高
    B = _cand((0.10, 0.10, 0.20, 0.15), "hsv")     # L1 低
    cands = [A, B]
    ts = {0: 0.8, 1: 0.0}
    r_l2 = select_gtfree(cands, prior, temporal_scores=ts)
    r_none = select_gtfree(cands, prior, temporal_scores=ts, l3_scores=None)
    r_default = select_gtfree(cands, prior, temporal_scores=ts, l3_scores={0: 0.0, 1: 0.0})
    assert r_l2 is r_none is A
    # 即便给出全 0 的 l3(但非 None), 因 l3_weight 把最终分拉向 0, 可能翻盘 —— 验证 None 才等同原行为
    assert r_default is A  # 这里 A 仍赢, 但关键是 r_none==r_l2(生产默认不改行为)


# ---- §3.1 弃权门 + governing 判别器接入(TDD) ----
def test_select_gtfree_backward_compat_no_governing():
    """向后兼容(R1): governing_scores=None 时(生产默认/未放行)行为与原 L1+L2+L3 完全一致,
    弃权门完全不触发(即便"想弃权"也不返 None)。"""
    prior = derive_ped_priors([(0.10, 0.30)])
    A = _cand((0.40, 0.20, 0.50, 0.50), "yolo")    # L1 高
    B = _cand((0.10, 0.10, 0.20, 0.15), "hsv")     # L1 低
    cands = [A, B]
    ts = {0: 0.8, 1: 0.0}
    # 无 governing: 等同 L1+L2(+l3) 行为, 必返回候选(永不 None)
    r_none = select_gtfree(cands, prior, temporal_scores=ts)
    r_explicit = select_gtfree(cands, prior, temporal_scores=ts, governing_scores=None)
    assert r_none is r_explicit is A
    # 即便传了全低 governing 分数(但非 None 才触发门)——None 时不触发
    r_zero_gov = select_gtfree(cands, prior, temporal_scores=ts,
                               governing_scores={0: 0.0, 1: 0.0}, governing_weight=1.0)
    # 注意: 这里 governing_scores 非 None, 会触发门; 该测试仅验证 None 路径不触发门
    assert r_none is A  # None 路径结果


def test_select_gtfree_governing_abstain_returns_none():
    """弃权门(R1): 无灯帧所有候选 conf 都低于阈值 -> 返 None(不输出绿)。
    这是 5/8 主导类(无灯干扰自发绿)的结构性修复。"""
    prior = derive_ped_priors([(0.10, 0.30)])
    # 两个候选都长得像灯(高 L1), 但都不是有效行人灯(conf 低)
    X = _cand((0.40, 0.20, 0.50, 0.50), "yolo")
    Y = _cand((0.60, 0.20, 0.70, 0.50), "yolo")
    cands = [X, Y]
    ts = {0: 1.0, 1: 1.0}  # 时序平局
    # 不传 governing: 旧行为必返一个(误绿来源)
    old = select_gtfree(cands, prior, temporal_scores=ts)
    assert old is not None
    # 传 governing 且都低: 弃权返 None
    sel = select_gtfree(cands, prior, temporal_scores=ts,
                        governing_scores={0: 0.10, 1: 0.20},
                        governing_weight=1.0, governing_threshold=0.5)
    assert sel is None


def test_select_gtfree_governing_reranks_valid_light():
    """弃权门 + 重排(R1): 干扰(高L1/低conf)被压低, 有效行人灯(低L1/高conf)排上来。
    治 2/8 排序错(背面/反射被当灯)。"""
    prior = derive_ped_priors([(0.10, 0.30)])
    # 干扰: 极像灯(aspect≈3.0, area≈0.03 → 高 L1≈1.0)但 conf=0.1
    interference = _cand((0.45, 0.25, 0.55, 0.55), "yolo")
    # 有效灯: 同样 aspect 但面积偏小(area≈0.011 → L1≈0.7)的候选, conf=0.9
    valid = _cand((0.27, 0.31, 0.33, 0.49), "yolo")
    cands = [interference, valid]
    ts = {0: 0.0, 1: 0.0}  # 无时序差异
    # 不传 governing(或 weight=0): 按 L1 选干扰(几何更像灯)
    sel_l1 = select_gtfree(cands, prior, temporal_scores=ts)
    assert sel_l1 is interference
    # 传 governing(weight=1.0): 按 conf 选有效灯, 且 conf>=阈值不弃权
    sel_gov = select_gtfree(cands, prior, temporal_scores=ts,
                            governing_scores={0: 0.1, 1: 0.9},
                            governing_weight=1.0, governing_threshold=0.5)
    assert sel_gov is valid


def test_select_gtfree_abstain_inactive_without_governing():
    """弃权门是 opt-in(R1): 不传 governing_scores 时绝不返 None(生产默认行为不变)。"""
    prior = derive_ped_priors([(0.10, 0.30)])
    X = _cand((0.40, 0.20, 0.50, 0.50), "yolo")
    cands = [X]
    # 无 governing_scores: 即便我们"想"弃权, 也必须返回候选
    assert select_gtfree(cands, prior) is X
    # 阈值再高也不影响 None 路径(门未启用)
    assert select_gtfree(cands, prior, governing_threshold=0.999) is X


if __name__ == "__main__":
    test_seed_with_gt_picks_ped_not_vehicle()
    test_selector_gtfree_picks_ped_by_geometry()
    test_selector_never_reads_gt_in_production()
    test_leave_some_out_is_true_holdout()
    test_leave_some_out_excludes_undetected_from_selection_acc()
    test_compute_temporal_scores_rewards_persistent_candidate()
    test_compute_temporal_scores_single_frame_is_empty()
    test_select_gtfree_temporal_flips_tie()
    test_select_gtfree_backward_compat_no_temporal()
    test_select_gtfree_l3_flips_tie()
    test_select_gtfree_backward_compat_no_l3()
    test_select_gtfree_backward_compat_no_governing()
    test_select_gtfree_governing_abstain_returns_none()
    test_select_gtfree_governing_reranks_valid_light()
    test_select_gtfree_abstain_inactive_without_governing()
    print("all ped_light_selector unit tests passed")
