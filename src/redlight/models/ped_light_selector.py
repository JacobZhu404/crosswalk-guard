"""PedLightSelector — M2a (GT 种子选灯, 训练/评测态) + M2b (生产 GT-free 选灯, 命门 §1.5)。

设计边界(cc 红线, 计划中 §1.5):
- M2a `seed_with_gt` 仅在训练/评测态使用(挖 crop / 度量精度), **绝不进生产**。
- M2b `select_gtfree` 是生产唯一选灯路径, **不接收任何逐帧 GT**(护栏1);
  其判别先验(ped 几何/尺寸统计)从 *其他视频* 的 GT 派生 —— leave-some-out 保证生产态无泄露。
- L1 几何(竖长条 h/w + 尺寸区间) 为主, **不依赖分类器自身输出**(cc 校准#1);
  L2 轨迹时序为增强, 稀疏帧退化为 L1。
- 位置仅软特征, 不作硬锚(固定 prior 已证死路)。

输入约定: 候选/GT 框均为 **归一化 (x1,y1,x2,y2)** (0-1), 与 diag 公平测口径一致(去循环)。
"""
import math
import statistics
from typing import List, Dict, Optional, Tuple

Box = Tuple[float, float, float, float]


def _box_wh(box: Box):
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])


def _center_dist(a: Box, b: Box) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def derive_ped_priors(gt_wh_list: List[Tuple[float, float]],
                      aspect_slack: float = 0.25, size_slack: float = 0.5) -> Dict:
    """从一组 GT (w, h) 归一化框派生 ped 几何先验。
    gt_wh_list: list of (w_norm, h_norm)。
    返回峰值高斯所需 {aspect_mean, aspect_std, area_mean, area_std} + 参考 min/max。
    **必须由 *其他视频* 的 GT 派生**(leave-some-out 保证生产态无泄露)。"""
    if not gt_wh_list:
        return {"aspect_mean": 3.0, "aspect_std": 1.0, "area_mean": 0.005, "area_std": 0.003,
                "aspect_min": 0.1, "aspect_max": 12.0, "area_min": 0.0, "area_max": 1.0}
    aspects = [h / w if w > 0 else 99.0 for w, h in gt_wh_list]
    areas = [w * h for w, h in gt_wh_list]
    a_mean = statistics.mean(aspects)
    a_std = max(statistics.pstdev(aspects), 0.3)
    s_mean = statistics.mean(areas)
    s_std = max(statistics.pstdev(areas), 1e-3)
    a_min, a_max = min(aspects), max(aspects)
    s_min, s_max = min(areas), max(areas)
    aspect_span = max(a_max - a_min, 0.2)
    size_span = max(s_max - s_min, 1e-4)
    return {
        "aspect_mean": a_mean, "aspect_std": a_std,
        "area_mean": s_mean, "area_std": s_std,
        "aspect_min": max(0.1, a_min - aspect_slack * aspect_span),
        "aspect_max": a_max + aspect_slack * aspect_span,
        "area_min": max(0.0, s_min - size_slack * size_span),
        "area_max": s_max + size_slack * size_span,
    }


def _l1_geom_score(box: Box, prior: Dict) -> float:
    """L1 几何判别(峰值高斯, 非二元区间): ped 灯=竖长条(高/宽 比大) + 尺寸在典型值附近。
    返回 0~1, 越高越 ped-like。 **不依赖分类器输出**(cc 校准#1)。"""
    w, h = _box_wh(box)
    if w <= 0 or h <= 0:
        return 0.0
    aspect = h / w
    area = w * h
    a_score = math.exp(-((aspect - prior["aspect_mean"]) ** 2) / (2 * prior["aspect_std"] ** 2))
    s_score = math.exp(-((area - prior["area_mean"]) ** 2) / (2 * (prior["area_std"] + 1e-6) ** 2))
    # 几何为主(0.7), 尺寸次要(0.3, 尺寸区间可能重叠车灯)
    return 0.7 * a_score + 0.3 * s_score


def seed_with_gt(candidates: List[Dict], gt_box: Box) -> Optional[Dict]:
    """M2a: 训练/评测态。选与 GT 框 IoU 最高的候选(用 GT 种子锁 ped 轨迹)。
    candidates: list of {"box_norm": (x1,y1,x2,y2), ...}。返回选中候选或 None。
    **生产禁用** —— 见 `select_gtfree`。"""
    best, best_iou = None, -1.0
    for c in candidates:
        b = c.get("box_norm")
        if not b:
            continue
        v = iou(b, gt_box)
        if v > best_iou:
            best_iou, best = v, c
    return best


def select_gtfree(candidates: List[Dict], ped_prior: Dict,
                 temporal_scores: Optional[Dict[int, float]] = None,
                 temporal_weight: float = 0.4,
                 l3_scores: Optional[Dict[int, float]] = None,
                 l3_weight: float = 0.3,
                 governing_scores: Optional[Dict[int, float]] = None,
                 governing_weight: float = 0.0,
                 governing_threshold: float = 0.5) -> Optional[Dict]:
    """M2b: 生产 GT-free 选灯。**不接收任何逐帧 GT**(护栏1)。
    candidates: list of {"box_norm":..., "source":...}。返回选中候选或 None。
    L1 几何判别(峰值高斯)为主; **YOLO cls=9 候选优先**(检测器类别=交通灯, 合法信号,
    非状态分类器自身输出, 符合校准#1)——有 YOLO 候选时只在 YOLO 里选, HSV 仅作无 YOLO 兜底。
    L2 时序(零标注, compute_temporal_scores 产): 同视频跨帧复现的固定设施(行人灯)得高分,
    瞬时亮斑/车灯得低分, temporal_scores={候选索引: 0~1}。temporal_scores=None 时退化为纯 L1。
    L3 学习式判别(Jacob 标样本训的 ped-vs-vehicle 头, 可选): l3_scores={候选索引: P(ped)∈[0,1]}。
    组合: final = (1-l3_weight)*base + l3_weight*l3_ped; l3_scores=None 时完全忽略(向后兼容,
    旧行为/单帧生产兜底不变, 且生产接线默认不传 l3 直到 cc 放行)。
    **governing 判别器接入(§3.1, 弃权门, R1)**:
    - governing_scores={候选索引: P(有效行人灯)∈[0,1]}(新判别器 "有效行人灯 vs 干扰" 输出);
    - final = (1-governing_weight)*base + governing_weight*governing_conf
      (此路令 l3_weight=0 即不叠加有害 L3); governing_scores=None 时完全忽略(向后兼容)。
    - **弃权门**: 仅当启用 governing(governing_scores 非 None)且最佳候选 conf < governing_threshold
      → 返回 None(该帧不输出绿), 治 5/8 无灯干扰自发绿 + 2/8 排序错。governing_scores=None 时
      门不触发(生产默认行为不变, 红线)。"""
    yolo_idx = [i for i, c in enumerate(candidates) if c.get("source") == "yolo"]
    use_idx = yolo_idx if yolo_idx else list(range(len(candidates)))
    best, best_score, best_conf = None, -1e9, 0.0
    for i in use_idx:
        c = candidates[i]
        b = c.get("box_norm")
        if not b:
            continue
        s = _l1_geom_score(b, ped_prior)
        if c.get("source") == "yolo":
            s += 0.15
        if temporal_scores is not None:
            s += temporal_weight * temporal_scores.get(i, 0.0)
        if l3_scores is not None:
            s = (1.0 - l3_weight) * s + l3_weight * l3_scores.get(i, 0.0)
        gconf = 0.0
        if governing_scores is not None:
            gconf = governing_scores.get(i, 0.0)
            s = (1.0 - governing_weight) * s + governing_weight * gconf
        if s > best_score:
            best_score, best, best_conf = s, c, gconf
    # 弃权门(R1): 仅启用 governing 且最佳候选非"有效行人灯"时返 None
    if governing_scores is not None and best is not None:
        if best_conf < governing_threshold:
            return None
    return best


def _temporal_match(a: Dict, b: Dict, center_thr: float, iou_thr: float) -> bool:
    ba, bb = a.get("box_norm"), b.get("box_norm")
    if not ba or not bb:
        return False
    if iou(ba, bb) >= iou_thr:
        return True
    if _center_dist(ba, bb) < center_thr:
        wa, ha = _box_wh(ba)
        wb, hb = _box_wh(bb)
        if wa > 0 and wb > 0 and abs((wa / wb) - (ha / hb)) < 0.5:
            return True
    return False


def compute_temporal_scores(video_frames: List[Dict],
                            center_thr: float = 0.03, iou_thr: float = 0.5
                            ) -> List[Dict[int, float]]:
    """L2 时序(零标注): 同一视频内跨帧复现的候选(固定设施/行人灯)得高分,
    瞬时亮斑/车灯(仅偶现)得低分。生产态在线累积历史同理(离线条形近似)。
    输入 video_frames: list of frame records, 每个含 "candidates":[{box_norm,source,...}]。
    返回 list(per frame) of {候选索引: temporal_score ∈ [0,1]}。
    **只依赖候选检测历史, 不读任何 GT**(护栏1 保持)。"""
    n = len(video_frames)
    if n <= 1:
        return [{} for _ in video_frames]
    out = []
    for i, fr in enumerate(video_frames):
        scores = {}
        for ci, c in enumerate(fr["candidates"]):
            matches = 0
            for j in range(n):
                if j == i:
                    continue
                for oc in video_frames[j]["candidates"]:
                    if _temporal_match(c, oc, center_thr, iou_thr):
                        matches += 1
                        break
            scores[ci] = min(1.0, matches / max(1, n - 1))
        out.append(scores)
    return out


def leave_some_out_eval(records: List[Dict], center_hit: float = 0.06,
                         iou30: float = 0.3, iou50: float = 0.5) -> Dict:
    """leave-one-video-out: 每视频 V 用其余视频 GT 派生 prior, 在 V 帧上跑 M2b,
    度量:
      (1) 检测覆盖: V 帧中是否存在中心距<center_hit 的候选(ped 被检测到);
      (2) 选灯准确率: 在 ped 被检测到的帧里, M2b 是否选了那个 ped 候选(中心距<center_hit)。
    records: list of {"video","fi","candidates":[{"box_norm",...}],"gt_box_norm","gt_wh"}。
    返回 {检测覆盖, 选灯准确率, mean_selected_iou, per_video}。
    注: 这是"新视频泛化"的代理(cc 校准#2), 非生产保证。
    """
    videos = sorted({r["video"] for r in records})
    per_video = {}
    det_total = sel_total = 0
    sum_iou_selected = 0.0
    for V in videos:
        train_wh = [r["gt_wh"] for r in records if r["video"] != V]
        prior = derive_ped_priors(train_wh)
        det_v = sel_v = 0
        for r in records:
            if r["video"] != V:
                continue
            cands = r["candidates"]
            gt = r["gt_box_norm"]
            sel = select_gtfree(cands, prior)
            if sel is not None and sel.get("box_norm"):
                sum_iou_selected += iou(sel["box_norm"], gt)
            has_det = any(_center_dist(c.get("box_norm"), gt) < center_hit
                          for c in cands if c.get("box_norm"))
            if has_det:
                det_v += 1
                if sel is not None and _center_dist(sel.get("box_norm"), gt) < center_hit:
                    sel_v += 1
        per_video[V] = {"detected_frames": det_v, "selection_correct": sel_v,
                        "selection_acc": (sel_v / det_v) if det_v else None}
        det_total += det_v
        sel_total += sel_v
    n = len(records)
    return {
        "n_frames": n,
        "n_videos": len(videos),
        "detection_coverage": det_total / n if n else 0.0,
        "selection_accuracy": (sel_total / det_total) if det_total else 0.0,
        "mean_selected_iou": (sum_iou_selected / n) if n else 0.0,
        "per_video": per_video,
    }
