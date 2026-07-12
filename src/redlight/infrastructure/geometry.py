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


def compute_overlap_ratio(box, mask, footprint=1.0, denom="box"):
    """检测框与 mask(斑马线前景) 的面积交并比类指标, 返回 0~1。

    footprint: 车框收紧比例(取车体下半部)。默认 1.0 = 整车框。
    设 0.5 = 只用车体下半部(车轮/底盘)作为"占道足迹", 去除 YOLO 大框
    对车顶/天空的稀释(E15/E17 实证: 真实占道仅 0.037 失真)。

    denom: 分母选择 (D2, 2026-07-12):
      - "box"  (默认, 旧行为): inside / 车框面积  = "车有多少压在线上"
      - "mask" (推荐):       inside / mask面积 = "斑马线被车覆盖的比例"
         COT"占据了斑马线 30%" 即此定义; 违规判定与 COT 统一用 mask 分母。
    """
    if mask is None:
        return 0.0
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    # 收紧为下半部足迹
    if 0.0 < footprint < 1.0:
        fh = max(1, int(round((y2 - y1) * footprint)))
        y1 = y2 - fh
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = mask[y1:y2, x1:x2]
    inside = int(np.count_nonzero(sub > 0))
    if denom == "mask":
        mask_area = int(np.count_nonzero(mask > 0))
        return inside / mask_area if mask_area > 0 else 0.0
    box_area = (x2 - x1) * (y2 - y1)
    return inside / box_area if box_area > 0 else 0.0


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
