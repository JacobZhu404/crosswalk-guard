"""TDD for CrosswalkDetectorV2 (Plan v6, Phase 1 train-free probe).

验证:
  1) 干净合成梯形帧 -> 检测器输出多边形掩膜, 分行 x 延展(非全宽水平带),
     y-span 误差小, 与理想白条掩膜 IoU 高。
  2) 时间聚合抗遮挡: 帧A完整 + 帧B中段被车矩形遮挡, 顺序 detect(A),detect(B)
     的聚合掩膜 y-span 覆盖 > 单帧B检测, 证明恢复了被遮挡行(改定 2 核心)。

注: 本测试用合成帧验证检测器"几何能力", 不依赖真实 GT; 真实 mask-IoU 由
eval_crosswalk_mask.py 在 35 帧 GT 上度量。
"""
import os
import sys
import types

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.models.crosswalk_v2 import CrosswalkDetectorV2


def _make_cfg():
    cfg = types.SimpleNamespace()
    cfg.crosswalk = types.SimpleNamespace(method="cv", cv_min_area=500)
    return cfg


def _build_frame(h=480, w=640, occ_top=None, occ_bottom=None):
    """画一个透视梯形斑马线(白条+暗背景), 可选在中段画遮挡矩形。

    梯形: 顶 y=150 x∈[250,390] (窄), 底 y=400 x∈[120,520] (宽)。
    白条: 梯形内每 20px 一列交替白(230)/暗(90)。
    """
    frame = np.full((h, w, 3), 90, dtype=np.uint8)  # 沥青灰
    # 梯形顶点
    top_y, bot_y = 150, 400
    trap = np.array([[250, top_y], [390, top_y],
                     [520, bot_y], [120, bot_y]], dtype=np.int32)
    # 梯形遮罩
    trap_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(trap_mask, [trap], 255)
    # 白条: 梯形内 (x//20)%2==0 的列为白
    for y in range(top_y, bot_y):
        for x in range(w):
            if trap_mask[y, x] and (x // 20) % 2 == 0:
                frame[y, x] = (230, 230, 230)
    # 遮挡(车): 中段画暗矩形
    if occ_top is not None:
        cv2.rectangle(frame, (100, occ_top), (540, occ_bottom), (70, 70, 70), cv2.FILLED)
    return frame, trap_mask, (top_y, bot_y)


def _yspan(mask):
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        return (0, 0)
    return (int(ys.min()), int(ys.max()))


def _iou(a, b):
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _ideal_white_bars(trap_mask):
    """理想白条掩膜: 梯形内白列(与 _build_frame 同规则)。"""
    h, w = trap_mask.shape
    ideal = np.zeros_like(trap_mask)
    for y in range(h):
        for x in range(w):
            if trap_mask[y, x] and (x // 20) % 2 == 0:
                ideal[y, x] = 255
    return ideal


def test_clean_frame_quadrilateral_fit():
    """干净帧: 检测器输出多边形掩膜, y-span 误差小, 与理想白条 IoU 高。"""
    frame, trap_mask, (gt_y1, gt_y2) = _build_frame()
    det = CrosswalkDetectorV2(_make_cfg(), verbose=False)
    mask = det.detect(frame)

    assert mask.sum() > 0, "掩膜不应为空"
    # 不是全宽水平带: 掩膜左右应有空白(即非整帧宽)
    ys, xs = np.where(mask > 0)
    x_extent = xs.max() - xs.min()
    assert x_extent < frame.shape[1] - 20, "应输出梯形(左右有边界), 而非全宽带"

    # y-span 误差 ≤ 40px
    y1, y2 = _yspan(mask)
    assert abs(y1 - gt_y1) <= 40, f"检测上沿 {y1} vs GT {gt_y1}"
    assert abs(y2 - gt_y2) <= 40, f"检测下沿 {y2} vs GT {gt_y2}"

    # 与整梯形 GT 区域 IoU ≥ 0.7 (检测器应恢复整块斑马线区域, 非仅白条)
    assert _iou(mask, trap_mask) >= 0.7, f"mask-IoU={_iou(mask, trap_mask):.3f} < 0.7"


def test_temporal_aggregation_recovers_occluded_rows():
    """时间聚合抗遮挡(改定 2): 帧A完整+帧B中段被车挡, 聚合掩膜恢复被挡行。"""
    frame_a, _, _ = _build_frame()                       # 完整
    occ_top, occ_bottom = 240, 310
    frame_b, _, _ = _build_frame(occ_top=occ_top, occ_bottom=occ_bottom)  # 中段被挡

    # 单帧 B(无聚合): 中段缺失
    det_single = CrosswalkDetectorV2(_make_cfg(), verbose=False)
    mask_single_b = det_single.detect(frame_b)

    # 顺序 detect(A) 再 detect(B): 聚合应恢复中段
    det_agg = CrosswalkDetectorV2(_make_cfg(), verbose=False)
    det_agg.detect(frame_a)
    mask_agg = det_agg.detect(frame_b)

    covered_single = int(mask_single_b.any(axis=1).sum())
    covered_agg = int(mask_agg.any(axis=1).sum())

    # 聚合恢复了被遮挡的中段 -> 覆盖行数更多(≥30 行余量)
    assert covered_agg >= covered_single + 30, \
        f"聚合覆盖行数={covered_agg} 未超过单帧 {covered_single} (差<30)"

    # 聚合掩膜在被遮挡行(240-310)有像素, 单帧基本没有
    occ_band_agg = mask_agg[occ_top:occ_bottom, :].sum()
    occ_band_single = mask_single_b[occ_top:occ_bottom, :].sum()
    assert occ_band_agg > occ_band_single + 100, \
        "聚合应恢复被遮挡中段像素"


def test_state_resets_per_video_no_leak_cross_video():
    """检测器每视频新建(状态隔离): 不同实例互不影响。"""
    frame_a, _, _ = _build_frame()
    d1 = CrosswalkDetectorV2(_make_cfg(), verbose=False)
    m1 = d1.detect(frame_a)
    d2 = CrosswalkDetectorV2(_make_cfg(), verbose=False)
    m2 = d2.detect(frame_a)
    assert np.array_equal(m1, m2), "同输入不同实例应产出相同掩膜"


if __name__ == "__main__":
    test_clean_frame_quadrilateral_fit()
    test_temporal_aggregation_recovers_occluded_rows()
    test_state_resets_per_video_no_leak_cross_video()
    print("ALL crosswalk_v2 TDD passed")
