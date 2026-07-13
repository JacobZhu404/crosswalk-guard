"""L3 行人信号灯候选并集 (M1 spec M1-D2): YOLO traffic-light 框 ∪ HSV 亮斑候选, 去重。

纯几何, 无 cv2/torch 依赖, 便于单测。输出归一化中心便于下游锚点/聚类复用。

去重用 **IoMin = 交集/较小框面积** 而非 IoU: 一个 HSV 亮斑常被更大的 YOLO 灯框
完全包含(IoU 很低但语义上是同一个信号), IoMin 能正确判定包含关系并去重。
"""


def build_candidates(yolo_boxes, hsv_boxes, frame_w, frame_h, iou_thr=0.5):
    """合并两路候选框, 重叠(IoMin>=iou_thr)者保留面积大者。

    yolo_boxes / hsv_boxes: list of (x1,y1,x2,y2) 像素。
    iou_thr: IoMin(交集/较小框面积) 阈值; 名称沿用 iou_thr 便于调用方理解。
    返回: list of {"box":(x1,y1,x2,y2), "source":"yolo"|"hsv", "area":int, "cx":float, "cy":float}
    """
    tagged = [(b, "yolo") for b in yolo_boxes] + [(b, "hsv") for b in hsv_boxes]
    # 面积降序: 先放大框, 后来的小框若与已保留框重叠(IoMin)则丢弃
    tagged.sort(key=lambda t: -_area(t[0]))
    kept = []
    for box, src in tagged:
        if any(_iomin(box, k["box"]) >= iou_thr for k in kept):
            continue
        kept.append({
            "box": tuple(int(v) for v in box),
            "source": src,
            "area": int(_area(box)),
            "cx": ((box[0] + box[2]) / 2.0) / frame_w if frame_w else 0.0,
            "cy": ((box[1] + box[3]) / 2.0) / frame_h if frame_h else 0.0,
        })
    return kept


def _area(b):
    return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))


def _iomin(a, b):
    """交集 / 较小框面积 (0~1)。用于包含关系去重(小框在大框内 -> 1.0)。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    m = min(_area(a), _area(b))
    return inter / m if m > 0 else 0.0
