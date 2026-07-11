"""L4c 评测指标 (纯函数, 易测, 不依赖模型)。

四类指标:
  1. 目标检测: Precision/Recall/F1/mAP@IoU
  2. OCR(车牌): 字符准确率 / 整牌准确率 / 编辑距离 / 省份准确率
  3. 事件(违规): 基于 track_id + Temporal-IoU 的事件级 P/R/F1
  4. 区间: Temporal-IoU
"""
from ..infrastructure.geometry import iou  # 复用基础设施几何


# ---------------------------------------------------------------------------
# 1. 目标检测指标
# ---------------------------------------------------------------------------
def precision_recall_f1(tp, fp, fn):
    """由 TP/FP/FN 计算 P/R/F1。除零安全。"""
    tp, fp, fn = int(tp), int(fp), int(fn)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def match_detections(pred_boxes, gt_boxes, iou_thr=0.5):
    """贪心匹配 pred 与 gt (按 IoU 降序)。

    返回 (tp, fp, fn, matches)
      matches: list of (pred_idx, gt_idx, iou)
      tp = 成功匹配数, fp = 未匹配 pred 数, fn = 未匹配 gt 数
    """
    pred_boxes = list(pred_boxes)
    gt_boxes = list(gt_boxes)
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            i = iou(pb, gb)
            if i >= iou_thr:
                pairs.append((i, pi, gi))
    pairs.sort(reverse=True)  # IoU 降序
    used_pred, used_gt = set(), set()
    matches = []
    for i, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, i))
    tp = len(matches)
    fp = len(pred_boxes) - len(used_pred)
    fn = len(gt_boxes) - len(used_gt)
    return tp, fp, fn, matches


def detection_metrics(pred_boxes, gt_boxes, iou_thr=0.5):
    """返回检测 P/R/F1 与计数。"""
    tp, fp, fn, matches = match_detections(pred_boxes, gt_boxes, iou_thr)
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "precision": p, "recall": r, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": len(pred_boxes), "n_gt": len(gt_boxes),
        "iou_thr": iou_thr,
    }


def compute_map(preds, gts, iou_thr=0.5, classes=None):
    """简化 VOC-mAP: 按类别对每个图像 pred/gt 做贪心匹配,
    以 pred 的 score 降序构建 P-R 曲线, 11-point 插值求 AP, 再对类别平均。

    preds / gts 结构: list(按图像) of list(按目标) of dict:
        pred: {"box":[x1,y1,x2,y2], "score":float, "cls":str}
        gt:   {"box":[x1,y1,x2,y2], "cls":str}
    返回 {"mAP":float, "per_class":{cls:ap}}
    """
    if classes is None:
        cls_set = set()
        for imgs in (preds, gts):
            for img in imgs:
                for o in img:
                    cls_set.add(o["cls"])
        classes = sorted(cls_set)
    per_class = {}
    for cls in classes:
        y_true, y_score = [], []
        for pimg, gimg in zip(preds, gts):
            pboxes = [(o["box"], float(o["score"])) for o in pimg if o["cls"] == cls]
            gboxes = [o["box"] for o in gimg if o["cls"] == cls]
            # 本图像内贪心匹配 (同 cls)
            pairs = []
            for pi, (pb, sc) in enumerate(pboxes):
                for gi, gb in enumerate(gboxes):
                    i = iou(pb, gb)
                    if i >= iou_thr:
                        pairs.append((i, pi, gi, sc))
            pairs.sort(reverse=True)
            used_p, used_g = set(), set()
            matched_scores = []
            for i, pi, gi, sc in pairs:
                if pi in used_p or gi in used_g:
                    continue
                used_p.add(pi); used_g.add(gi)
                matched_scores.append(sc)
            n_gt = len(gboxes)
            # 正样本(pred matched): score; 负样本(unmatched pred): score
            for pi, (pb, sc) in enumerate(pboxes):
                if pi in used_p:
                    y_true.append(1); y_score.append(sc)
                else:
                    y_true.append(0); y_score.append(sc)
            # 没有 gt 的图像里所有 pred 都是 FP, 上面已处理
            _ = n_gt
        ap = _ap_11_point(y_true, y_score)
        per_class[cls] = ap
    mAP = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return {"mAP": mAP, "per_class": per_class}


def _ap_11_point(y_true, y_score):
    """11-point 插值 AP。y_true: 0/1 列表; y_score: 对应分数。"""
    if not y_score:
        return 0.0
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    y_sorted = [y_true[i] for i in order]
    n_pos = sum(y_sorted)
    if n_pos == 0:
        return 0.0
    tp_cum, fp_cum = 0, 0
    precisions = []
    for y in y_sorted:
        if y == 1:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
    # 11-point: recall = [0.0,0.1,...,1.0]
    recalls = [tp_cum / n_pos for tp_cum in range(1, n_pos + 1)]
    # 构造 (recall, precision) 点
    points = list(zip(recalls, precisions))
    interp = 0.0
    for t in [i / 10.0 for i in range(11)]:
        # 取 recall >= t 的最大 precision
        vals = [p for (r, p) in points if r >= t]
        interp += max(vals) if vals else 0.0
    return interp / 11.0


# ---------------------------------------------------------------------------
# 2. OCR 指标 (车牌)
# ---------------------------------------------------------------------------
def levenshtein(a, b):
    """标准编辑距离 (Levenshtein)。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def edit_distance(a, b):
    return levenshtein(a, b)


def edit_similarity(a, b):
    """1 - dist / max(len)。完全相同为 1.0。"""
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / m


def char_accuracy(pred, gt):
    """位置对齐的字符级准确率 = 匹配字符数 / len(gt)。"""
    if not gt:
        return 1.0 if not pred else 0.0
    n = min(len(pred), len(gt))
    hit = sum(1 for i in range(n) if pred[i] == gt[i])
    return hit / len(gt)


def plate_accuracy(preds, gts):
    """整牌精确匹配率 = 完全一致对数 / 总对数 (逐对比较)。"""
    assert len(preds) == len(gts), "preds/gts 必须逐对"
    if not gts:
        return 0.0
    hit = sum(1 for p, g in zip(preds, gts) if p == g)
    return hit / len(gts)


def province_accuracy(preds, gts):
    """省份(首字符)识别准确率。"""
    assert len(preds) == len(gts)
    if not gts:
        return 0.0
    hit = sum(1 for p, g in zip(preds, gts) if p and g and p[0] == g[0])
    return hit / len(gts)


# ---------------------------------------------------------------------------
# 3. 区间 / 事件指标
# ---------------------------------------------------------------------------
def temporal_iou(span_a, span_b):
    """两区间 [start,end] 的 Temporal-IoU。无交返回 0.0。"""
    s1, e1 = span_a
    s2, e2 = span_b
    if e1 < s1 or e2 < s2:
        return 0.0
    inter_s, inter_e = max(s1, s2), min(e1, e2)
    inter = max(0.0, inter_e - inter_s)
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def event_metrics(pred_events, gt_events, tiou_thr=0.5, match_by_track=True):
    """事件级 P/R/F1。

    event: {"track_id":int, "start_ts":float, "end_ts":float, "status":str}
    匹配: 同 track_id(可选) 且 temporal_iou >= tiou_thr 即为一对 TP。
    返回 {"precision","recall","f1","tp","fp","fn"}。
    """
    preds = list(pred_events)
    gts = list(gt_events)
    pairs = []
    for pi, pe in enumerate(preds):
        for gi, ge in enumerate(gts):
            if match_by_track and pe.get("track_id") != ge.get("track_id"):
                continue
            tiou = temporal_iou((pe["start_ts"], pe["end_ts"]),
                                (ge["start_ts"], ge["end_ts"]))
            if tiou >= tiou_thr:
                pairs.append((tiou, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    for tiou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
    tp = len(used_p)
    fp = len(preds) - len(used_p)
    fn = len(gts) - len(used_g)
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# 4. 信号灯状态分类指标 (要求#6: 量化红绿灯检测器 Precision/Recall)
# ---------------------------------------------------------------------------
LIGHT_CLASSES = ["red", "green", "flashing", "unknown"]


def light_state_metrics(pred_states, gt_states, classes=None):
    """逐帧信号灯状态分类 P/R/F1 (micro 准确率 + 每类 P/R/F1)。

    识别对象 = 斑马线行人信号灯 (§5.3.1)。状态枚举见 LIGHT_CLASSES。
    pred_states / gt_states: 等长列表, 元素为状态字符串。
    返回:
      accuracy : 整体帧准确率 (正确预测数 / 总数)
      macro_f1 : 各类 f1 的宏平均
      per_class: {cls: {precision, recall, f1, support}}
      n        : 样本数
    """
    assert len(pred_states) == len(gt_states), "pred/gt 必须逐帧等长"
    if classes is None:
        classes = list(LIGHT_CLASSES)
    n = len(gt_states)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "n": 0}
    correct = sum(1 for p, g in zip(pred_states, gt_states) if p == g)
    per_class = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(pred_states, gt_states) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred_states, gt_states) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred_states, gt_states) if p != cls and g == cls)
        pc, rc, f1c = precision_recall_f1(tp, fp, fn)
        per_class[cls] = {"precision": pc, "recall": rc, "f1": f1c, "support": fn + tp}
    macro_f1 = sum(d["f1"] for d in per_class.values()) / len(per_class)
    return {"accuracy": correct / n, "macro_f1": macro_f1,
            "per_class": per_class, "n": n}
