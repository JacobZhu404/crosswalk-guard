"""GT (Ground Truth) 加载与查询工具。

与 docs/plans/2026-07-12-gt-format-spec.md 和 scripts/validate_gt.py 对齐的权威 GT 解析器。
"""

import csv
import os
from typing import Callable, Dict, List, Tuple


PLATE_SPECIAL_TOKENS = {"?", "无牌"}


def _parse_plate_field(field: str) -> List[str]:
    """分号分隔 -> token 列表(去空去空格)，过滤 ?/无牌。"""
    return [t.strip() for t in (field or "").split(";")
            if t.strip() and t.strip() not in PLATE_SPECIAL_TOKENS]


def load_plate_gt(events_csv: str) -> Dict[str, List[str]]:
    """从 events.csv 加载车牌真值。

    合并 violating_plates 与 other_plates，按 ';' 分割，
    过滤空串、?、无牌，去重。

    Args:
        events_csv: events.csv 路径

    Returns:
        {video_name: [plate1, plate2, ...]}
    """
    gt: Dict[str, List[str]] = {}
    if not os.path.exists(events_csv):
        return gt
    with open(events_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            video = row.get("video", "").strip()
            if not video:
                continue
            plates: List[str] = []
            for field in ("violating_plates", "other_plates"):
                plates.extend(_parse_plate_field(row.get(field, "")))
            if plates:
                gt.setdefault(video, []).extend(plates)
    # 去重并保持顺序
    for video in gt:
        seen = set()
        deduped = []
        for p in gt[video]:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        gt[video] = deduped
    return gt


def load_light_segments(events_csv: str) -> Dict[str, List[Tuple[float, float, str, str]]]:
    """从 events.csv 加载灯态段级真值（含 light_evidence）。

    Args:
        events_csv: events.csv 路径

    Returns:
        {video_name: [(start_s, end_s, light_state, light_evidence), ...]}
        light_evidence 在 light_state=unknown 时为空字符串。
    """
    gt: Dict[str, List[Tuple[float, float, str, str]]] = {}
    if not os.path.exists(events_csv):
        return gt
    with open(events_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            video = row.get("video", "").strip()
            if not video:
                continue
            try:
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
            except (KeyError, ValueError):
                continue
            light_state = (row.get("light_state") or "").strip()
            light_evidence = (row.get("light_evidence") or "").strip()
            gt.setdefault(video, []).append((start_s, end_s, light_state, light_evidence))
    # 按 start_s 升序
    for video in gt:
        gt[video].sort(key=lambda x: x[0])
    return gt


def load_light_state_csv(light_states_csv: str) -> Dict[str, List[Tuple[float, float, str, str]]]:
    """从 light_states.csv 加载灯态段级真值（含 confidence）。

    Args:
        light_states_csv: light_states.csv 路径

    Returns:
        {video_name: [(start_s, end_s, state, confidence), ...]}
    """
    gt: Dict[str, List[Tuple[float, float, str, str]]] = {}
    if not os.path.exists(light_states_csv):
        return gt
    with open(light_states_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            video = row.get("video", "").strip()
            if not video:
                continue
            try:
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
            except (KeyError, ValueError):
                continue
            state = (row.get("state") or "").strip()
            confidence = (row.get("confidence") or "").strip()
            gt.setdefault(video, []).append((start_s, end_s, state, confidence))
    for video in gt:
        gt[video].sort(key=lambda x: x[0])
    return gt


def load_labeled_gt(gt_csv: str) -> Tuple[List[str], List[str]]:
    """从 *_gt.csv 加载人工标注的灯态真值。

    Args:
        gt_csv: 人工标注 CSV 路径

    Returns:
        (predicted_states, gt_states) — 仅返回已标注(gt_state 非空)的行。
    """
    preds, gts = [], []
    if not os.path.exists(gt_csv):
        return preds, gts
    with open(gt_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gt_state = (row.get("gt_state") or "").strip()
            if not gt_state:
                continue
            preds.append(row.get("predicted_state", "").strip())
            gts.append(gt_state)
    return preds, gts


def state_at(segments: List[Tuple], timestamp: float) -> Tuple[str, str]:
    """在段列表中查询给定时间戳的状态。

    区间匹配规则: start <= timestamp <= end（两端闭合）。
    未命中时兜底返回最近一段的 state 与 meta。

    Args:
        segments: [(start, end, state, meta), ...]，已按 start 升序
        timestamp: 查询时间点

    Returns:
        (state, meta) — 未命中且 segments 为空时返回 ("unknown", "confirmed")
    """
    for s, e, st, meta in segments:
        if s <= timestamp <= e:
            return st, meta
    # 兜底: 取最近段（按边界距离）
    if not segments:
        return "unknown", "confirmed"
    best_seg = segments[0]
    best_dist = min(abs(timestamp - segments[0][0]), abs(timestamp - segments[0][1]))
    for seg in segments[1:]:
        dist = min(abs(timestamp - seg[0]), abs(timestamp - seg[1]))
        if dist < best_dist:
            best_dist = dist
            best_seg = seg
    return best_seg[2], best_seg[3]


def expand_light_evidence(
    segments: List[Tuple[float, float, str, str]]
) -> Callable[[float], Tuple[str, str]]:
    """将段级灯态 GT 展开为逐帧查询闭包。

    按 gt-format-spec.md 的 evidence 语义:
    - visible 段: 期望检测器输出字面 light_state
    - inferred / occluded 段: 期望输出 unknown (灯不可见，进 review)
    - unknown 段(无 evidence): 期望 unknown

    Args:
        segments: [(start_s, end_s, light_state, light_evidence), ...]

    Returns:
        callable(ts: float) -> (gt_state_for_detector, evidence)
    """
    def fn(ts: float) -> Tuple[str, str]:
        for a, b, stt, ev in segments:
            if a <= ts <= b:
                if ev in ("inferred", "occluded"):
                    return "unknown", ev
                return stt, (ev or "visible")
        return "unknown", "n/a"
    return fn
