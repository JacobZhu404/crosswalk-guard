"""Part B 模块级评测的纯函数: 斑马线 band-IoU + 跟踪归属/静止准确率。

纯函数(便于单测), 不碰管线逻辑。评测脚本 scripts/eval_crosswalk_mask.py /
scripts/eval_tracking.py 调用本模块; GT 标注由 Jacob 提供(wb 只造骨架)。
"""
import numpy as np


def band_iou(pred_band, gt_band):
    """两竖直区间 [y0, y1] 的 interval IoU。任一为空/无效 -> 0.0。

    斑马线掩膜在 v11 下是全宽矩形, 其"带"由竖直范围定义, 故用区间 IoU 近似掩膜 IoU。
    """
    if pred_band is None or gt_band is None:
        return 0.0
    p0, p1 = pred_band
    g0, g1 = gt_band
    if p1 <= p0 or g1 <= g0:
        return 0.0
    lo, hi = max(p0, g0), min(p1, g1)
    ov = max(0.0, hi - lo)
    union = (p1 - p0) + (g1 - g0) - ov
    return ov / union if union > 0 else 0.0


def mask_band(mask):
    """从二值掩膜取非零行区间 [y0, y1]; 全空 -> None。"""
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    return (int(rows[0]), int(rows[-1]))


def poly_to_mask(poly, h, w):
    """把多边形顶点 [[x,y],...] 光栅化成 (h,w) 二值掩膜(255 前景)。

    支持任意四边形/多边形(≥3 点), 用于斜视角斑马线 GT。空/不足 3 点 -> 全零。
    """
    import cv2
    m = np.zeros((h, w), dtype=np.uint8)
    if not poly or len(poly) < 3:
        return m
    pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m


def mask_iou(pred_mask, gt_mask):
    """两二值掩膜的 2D IoU(交/并); 任一全空且另一非空 -> 0.0, 皆空 -> 0.0。

    通用于任意形状 GT(多边形/框/带), 取代仅竖直的 band_iou 用于斜马线。
    """
    if pred_mask is None or gt_mask is None:
        return 0.0
    p = pred_mask > 0
    g = gt_mask > 0
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    return inter / union if union > 0 else 0.0


def iou_box(a, b):
    """两框 [x1, y1, x2, y2] 的 IoU; 任一无效 -> 0.0。"""
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def attribution_union(track_samples, anchor_boxes, window, T=0.5):
    """窗内归属该违章车的所有 track_id 之并集(理想大小=1, 碎片化时>1)。

    track_samples: tid -> [{ts, stationary, box, ...}, ...]
    anchor_boxes:  同一违章车的锚框列表 [box, ...] (不同锚帧的框)
    window:        (start_s, end_s)
    T:             box IoU 归属阈值(默认 0.5, 与 track 静止判定 IoU 量级一致)

    归属判据(已写死, cc 裁定): 窗内任一帧中, 若某 track 的框与任一锚框 IoU >= T,
    则该 track 归入此车。取**并集**而非"只取 IoU 最高那个"——否则碎片化数会低估。
    """
    s, e = window
    attr = set()
    for tid, samples in track_samples.items():
        attributed = False
        for sm in samples:
            if not (s <= sm["ts"] <= e):
                continue
            box = sm.get("box")
            if not box:
                continue
            if any(iou_box(box, ab) >= T for ab in anchor_boxes):
                attributed = True
                break
        if attributed:
            attr.add(tid)
    return attr


def stationary_accuracy(track_samples, tids, window):
    """窗内给定 track 集合的 stationary 帧占比(违章车定义上应静止)。"""
    s, e = window
    tot = stat = 0
    for tid in tids:
        for sm in track_samples.get(tid, []):
            if not (s <= sm["ts"] <= e):
                continue
            tot += 1
            if sm.get("stationary"):
                stat += 1
    return (stat / tot) if tot > 0 else 0.0
