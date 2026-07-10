"""L1 基础设施: 几何计算工具 (纯函数, 易测)。"""
import cv2
import numpy as np


def iou(box_a, box_b):
    """两个 [x1,y1,x2,y2] 框的 IoU。无交集返回 0.0。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_overlap_ratio(box, mask):
    """检测框内落入 mask(前景) 的面积占比, 返回 0~1。"""
    if mask is None:
        return 0.0
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = mask[y1:y2, x1:x2]
    box_area = (x2 - x1) * (y2 - y1)
    if box_area <= 0:
        return 0.0
    inside = int(np.count_nonzero(sub > 0))
    return inside / box_area


def mask_to_contour(mask, min_area=500):
    """从 mask 取最大连通域轮廓, 用于可视化。无则返回 None。"""
    if mask is None:
        return None
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return c
