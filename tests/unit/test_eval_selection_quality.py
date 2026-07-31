"""eval_selection_quality 指标单测(TDD, 纯合成, 不跑 YOLO/视频)。

验证 §3.3 核心指标: 选灯精度(IoU≥0.3) / 正确弃权(无灯帧 conf<τ) / 漏绿(硬约束) / τ 敏感性。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # 仓库根(使 scripts 可导入)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
from scripts.eval_selection_quality import metrics_for_tau, color_state
import cv2


def _row(video, gov_boxes, best_box, best_conf, best_color, no_light=False, gt_green=False):
    best_cand = {"box_norm": tuple(best_box), "source": "yolo"} if best_box else None
    return {"video": video, "gov_boxes": [tuple(b) for b in gov_boxes],
            "best_cand": best_cand, "best_conf": best_conf, "best_color": best_color,
            "no_light": no_light, "gt_green": gt_green}


def test_selection_precision_iou_threshold():
    """governing 帧, best 候选 IoU≥0.3 → 选灯精度计数。"""
    rows = [_row("v", gov_boxes=[(0.4, 0.2, 0.5, 0.5)], best_box=(0.4, 0.2, 0.5, 0.5),
                best_conf=0.9, best_color="green")]
    m = metrics_for_tau(rows, tau=0.5)
    assert abs(m["sel_prec"] - 1.0) < 1e-9


def test_selection_precision_iou_below_threshold():
    """governing 帧, best 候选 IoU<0.3(选错灯) → 选灯精度不计。"""
    rows = [_row("v", gov_boxes=[(0.4, 0.2, 0.5, 0.5)], best_box=(0.1, 0.1, 0.15, 0.2),
                best_conf=0.9, best_color="green")]
    m = metrics_for_tau(rows, tau=0.5)
    assert m["sel_prec"] == 0.0


def test_correct_rejection_no_light():
    """无灯帧, best conf<τ → 正确弃权(rej_rate=1.0)。"""
    rows = [_row("v", gov_boxes=[], best_box=(0.2, 0.2, 0.3, 0.4),
                best_conf=0.1, best_color="green", no_light=True)]
    m = metrics_for_tau(rows, tau=0.5)
    assert abs(m["rej_rate"] - 1.0) < 1e-9


def test_no_light_not_abstained_is_not_rejection():
    """无灯帧, best conf>=τ(误判有效灯) → 未弃权(rej_rate=0)。"""
    rows = [_row("v", gov_boxes=[], best_box=(0.2, 0.2, 0.3, 0.4),
                best_conf=0.9, best_color="green", no_light=True)]
    m = metrics_for_tau(rows, tau=0.5)
    assert m["rej_rate"] == 0.0


def test_miss_green_hard_constraint():
    """真绿帧若 conf<τ 被弃权 → 漏绿计数(R4 硬约束)。"""
    rows_green_sel = [_row("v", gov_boxes=[(0.1, 0.1, 0.2, 0.3)], best_box=(0.1, 0.1, 0.2, 0.3),
                           best_conf=0.9, best_color="green", gt_green=True)]
    m1 = metrics_for_tau(rows_green_sel, tau=0.5)
    assert m1["miss"] == 0  # 选中且绿 → 不漏
    rows_green_abstain = [_row("v", gov_boxes=[(0.1, 0.1, 0.2, 0.3)], best_box=(0.1, 0.1, 0.2, 0.3),
                                best_conf=0.9, best_color="green", gt_green=True)]
    m2 = metrics_for_tau(rows_green_abstain, tau=0.95)  # conf 0.9<0.95 → 弃权
    assert m2["miss"] == 1  # 漏绿 +1


def test_false_green_excl_05():
    """误绿(扣05): 05 帧不计入主标尺。"""
    rows = [
        _row("违章05", gov_boxes=[], best_box=(0.2, 0.2, 0.3, 0.4), best_conf=0.9,
             best_color="green", no_light=True),  # 05 误绿
        _row("违章03", gov_boxes=[], best_box=(0.2, 0.2, 0.3, 0.4), best_conf=0.9,
             best_color="green", no_light=True),  # 03 误绿
    ]
    m_excl = metrics_for_tau(rows, tau=0.5, exclude_video="违章05")
    assert m_excl["fg"] == 1 and m_excl["n_eval"] == 1  # 仅 03 计入
    m_all = metrics_for_tau(rows, tau=0.5, exclude_video=None)
    assert m_all["fg"] == 2  # 全量计入 2


def test_tau_sensitivity_abstain_rate():
    """τ 越高, 弃权越多 → 漏绿随 τ 单调不减(阈值越高越易弃绿帧)。"""
    rows = [_row("v", gov_boxes=[(0.1, 0.1, 0.2, 0.3)], best_box=(0.1, 0.1, 0.2, 0.3),
                best_conf=0.6, best_color="green", gt_green=True)]
    miss_low = metrics_for_tau(rows, tau=0.3)["miss"]   # 0.6>=0.3 → 选中
    miss_high = metrics_for_tau(rows, tau=0.9)["miss"]  # 0.6<0.9 → 弃权
    assert miss_low == 0 and miss_high == 1


def test_color_state_basic():
    """color_state 基本判定(绿/红/无)。"""
    green = np.zeros((10, 10, 3), dtype=np.uint8); green[:, :, 1] = 200  # 纯绿
    assert color_state(green) == "green"
    none = np.zeros((10, 10, 3), dtype=np.uint8)
    assert color_state(none) is None


if __name__ == "__main__":
    test_selection_precision_iou_threshold()
    test_selection_precision_iou_below_threshold()
    test_correct_rejection_no_light()
    test_no_light_not_abstained_is_not_rejection()
    test_miss_green_hard_constraint()
    test_false_green_excl_05()
    test_tau_sensitivity_abstain_rate()
    test_color_state_basic()
    print("all eval_selection_quality unit tests passed")
