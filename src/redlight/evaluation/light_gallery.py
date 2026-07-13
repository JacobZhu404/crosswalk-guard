"""灯态复核画廊 (BaseGalleryBuilder 实现)。

用途: 用户滚动逐张判定 mismatch 帧是 "算法错 / 标注错 / 都错 / 其他"。
风格与 plate_gallery 保持一致, 便于统一标注体验。
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..infrastructure.config import load_config
from ..infrastructure.image_utils import conf_color_bgr, robust_imread
from ..models.traffic_light import TrafficLightDetector
from .gallery_builder import BaseGalleryBuilder

STATE_COLOR = {
    "green": "#22c55e",
    "red": "#ef4444",
    "unknown": "#9ca3af",
    "flashing": "#f59e0b",
}


def _load_priors(path: str) -> Dict[str, Tuple[float, float, int]]:
    """加载灯态先验 (cx, cy, roi)。"""
    import json

    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out: Dict[str, Tuple[float, float, int]] = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = (float(v[0]), float(v[1]), int(v[2]) if len(v) > 2 else 160)
        elif isinstance(v, dict):
            out[k] = (
                float(v["cx"]),
                float(v["cy"]),
                int(v.get("roi", 160)),
            )
    return out


def _compress_records(records: List[dict], key: str = "pred") -> List[list]:
    """把逐帧记录压缩为连续段 [(state, start, end), ...]。"""
    segs: List[list] = []
    for r in records:
        st = r[key]
        t = float(r["t_sec"])
        if segs and segs[-1][0] == st:
            segs[-1][2] = t
        else:
            segs.append([st, t, t])
    return segs


def _timeline_html(segs: List[list], total_dur: float, label: str) -> str:
    """生成一条时间线色带 HTML。"""
    parts = []
    for st, a, b in segs:
        dur = max(b - a, 0.01)
        w = dur / total_dur * 100
        c = STATE_COLOR.get(st, "#64748b")
        parts.append(
            f'<div style="width:{w:.2f}%;background:{c};" title="{st} {a:.1f}-{b:.1f}s"></div>'
        )
    return (
        f'<div style="font-size:12px;margin:2px 0;color:#475569;">{label}</div>'
        f'<div style="display:flex;height:22px;border-radius:4px;overflow:hidden;'
        f'border:1px solid #e2e8f0;">{"".join(parts)}</div>'
    )


def _detect_center(fr: np.ndarray, det: Optional[TrafficLightDetector]) -> Optional[Tuple[float, float]]:
    """跑检测器取算法认定的灯中心(归一化 cx,cy); 失败返回 None。"""
    if det is None:
        return None
    try:
        res = det.detect(fr)
        t = res.get("track") or res.get("anchor")
        if t and "cx" in t:
            return (float(t["cx"]), float(t["cy"]))
    except Exception:
        return None
    return None


class LightGalleryBuilder(BaseGalleryBuilder):
    """灯态识别误差确认画廊生成器。

    gt_map 期望: {video: [(start_s, end_s, state, confidence), ...]}
    video_items 期望: {video: [{"frame_idx": int, "t_sec": float, "pred": str,
                                 "gt": str, "gt_conf": str, "conf": float}, ...]}
    """

    def __init__(
        self,
        eval_dir: str,
        frames_dir: Optional[str] = None,
        feedback_path: Optional[str] = None,
        max_crops: int = 10,
        priors_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        super().__init__(eval_dir, frames_dir, feedback_path, max_crops)
        self.priors = _load_priors(priors_path) if priors_path else {}
        self.config_path = config_path
        self._det_cache: Dict[str, Optional[TrafficLightDetector]] = {}

    # ---- 子类必须实现 ----

    @property
    def title(self) -> str:
        return "灯态识别误差确认画廊"

    @property
    def toolbar_label(self) -> str:
        return "灯态误差确认画廊"

    def intro_html(self) -> str:
        return (
            "左=信号灯 ROI 特写("
            '<span style="color:#3b82f6;font-weight:700;">蓝框=搜索区</span>'
            ": 算法只在此框内找灯头; "
            '<span style="color:#eab308;font-weight:700;">黄圈=读取点</span>'
            ": 算法实际取色的中心点)。<b>点小图看原始整帧</b>(蓝框=搜索区, 黄圈=读取点)。"
            "请判定: <b>算法错</b> / <b>标注错</b> / <b>都错</b> / <b>其他</b>; "
            "原因按判定树选: <b>搜索区没罩住真信号</b>(蓝框没罩住) → "
            "<b>读取点没落在真灯上</b>(蓝框对但黄圈偏) → "
            "<b>颜色读错</b>(黄圈在真灯上但色被反射/白边读翻) / GT段边界标反 / 其他。"
        )

    def verdict_options(self) -> List[Tuple[str, str]]:
        return [
            ("algo_wrong", "算法错"),
            ("label_wrong", "标注错"),
            ("both_wrong", "都错"),
            ("other", "其他"),
        ]

    def reason_options(self) -> List[Tuple[str, str]]:
        return [
            ("search_area", "搜索区没罩住真信号(蓝框不对)"),
            ("reading_point", "读取点没落在真灯上(黄圈不对)"),
            ("color", "颜色读错(位置对但色错)"),
            ("gt_flipped", "GT段边界标反"),
            ("other", "其他"),
        ]

    def is_mismatch(self, item: dict, gt: Any) -> bool:
        """mismatch = pred != gt; 优先 confirmed GT, 无 confirmed 则退回全部。"""
        if item.get("pred") == item.get("gt"):
            return False
        # 如果 gt_conf 存在且不是 confirmed, 先不计入(但若无 confirmed 则全算)
        return True

    def annotate_crop(self, frame: np.ndarray, item: dict, gt: Any) -> np.ndarray:
        video = item.get("video", "")
        prior = self.priors.get(video)
        det = self._get_detector(video)
        label_txt = (
            f"t={float(item.get('t_sec', 0)):.1f}s "
            f"预{item.get('pred', '')}/GT{item.get('gt', '')}"
        )
        conf = float(item.get("conf", 0.0))
        return _annotate_crop_light(frame, prior, det, label_txt, conf)

    def annotate_full(self, frame: np.ndarray, item: dict, gt: Any) -> np.ndarray:
        video = item.get("video", "")
        prior = self.priors.get(video)
        det = self._get_detector(video)
        label_txt = (
            f"t={float(item.get('t_sec', 0)):.1f}s "
            f"预{item.get('pred', '')}/GT{item.get('gt', '')}"
        )
        conf = float(item.get("conf", 0.0))
        return _annotate_full_light(frame, prior, det, label_txt, conf)

    def item_meta_html(self, item: dict, gt: Any) -> str:
        conf = float(item.get("conf", 0.0))
        conf_color = "#16a34a" if conf >= 0.85 else ("#d97706" if conf >= 0.6 else "#dc2626")
        return (
            f'预测 <b class="p-{item.get("pred", "")}">{item.get("pred", "")}</b> / '
            f'GT <b class="p-{item.get("gt", "")}">{item.get("gt", "")}</b> '
            f'<span class="conf" style="background:{conf_color}">置信度 {conf:.2f}</span>'
        )

    def feedback_key(self, video: str, item: dict) -> Tuple[str, str]:
        return (video, f"{float(item.get('t_sec', 0)):.1f}")

    # ---- 可选覆写 ----

    def timeline_html(self, video: str, items: List[dict], gt: Any) -> str:
        if not items:
            return ""
        total_dur = float(items[-1]["t_sec"])
        pred_segs = _compress_records(items, "pred")
        tl = _timeline_html(pred_segs, total_dur, "预测时间线")
        if gt and isinstance(gt, list):
            gt_segs = _compress_records(
                [{"pred": s[2], "t_sec": s[0]} for s in gt]
                + [{"pred": gt[-1][2], "t_sec": total_dur}],
                "pred",
            )
            tl += _timeline_html(gt_segs, total_dur, "GT 时间线")
        return tl

    def extra_data_attrs(self, video: str, item: dict, gt: Any) -> Dict[str, str]:
        return {
            "pred": item.get("pred", ""),
            "gt": item.get("gt", ""),
        }

    # ---- 内部 ----

    def _get_detector(self, video: str) -> Optional[TrafficLightDetector]:
        if video in self._det_cache:
            return self._det_cache[video]
        pv = self.priors.get(video)
        if pv is None or not self.config_path:
            self._det_cache[video] = None
            return None
        try:
            cfg = load_config(self.config_path)
            det = TrafficLightDetector(cfg, verbose=False)
            det.signal_prior = (pv[0], pv[1])
            det.prior_roi_px = pv[2]
            self._det_cache[video] = det
            return det
        except Exception:
            self._det_cache[video] = None
            return None


# ---------- 灯态专用标注函数 ----------


def _annotate_crop_light(
    fr: np.ndarray,
    prior: Optional[Tuple[float, float, int]],
    det: Optional[TrafficLightDetector],
    label_txt: str,
    conf: float = 0.0,
) -> np.ndarray:
    """信号灯 ROI 特写(小图): 浅蓝框=搜索区边界, 黄圈=读取点。"""
    H, W = fr.shape[:2]
    if prior is not None:
        cx, cy, roi = prior
        ax, ay = int(cx * W), int(cy * H)
        x1, y1 = max(0, ax - roi), max(0, ay - roi)
        x2, y2 = min(W, ax + roi), min(H, ay + roi)
    else:
        x1, y1, x2, y2 = 0, 0, W, H
    crop = fr[y1:y2, x1:x2].copy()
    Hc, Wc = crop.shape[:2]
    if Hc < 4 or Wc < 4:
        return crop
    # 浅蓝框 = 搜索区(ROI)边界
    cv2.rectangle(crop, (6, 6), (Wc - 7, Hc - 7), (255, 178, 102), 3)
    # 黄圈 = 算法读取颜色的中心点
    c = _detect_center(fr, det)
    if c is None:
        c = (cx, cy) if prior is not None else (0.5, 0.5)
    ccx = max(0, min(Wc - 1, int(c[0] * W - x1)))
    ccy = max(0, min(Hc - 1, int(c[1] * H - y1)))
    cv2.circle(crop, (ccx, ccy), 12, (0, 255, 255), 2)
    cv2.drawMarker(crop, (ccx, ccy), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
    # 顶部黑条 + 文字 + 置信度色块
    cv2.rectangle(crop, (0, 0), (Wc - 1, 28), (0, 0, 0), -1)
    cv2.putText(crop, label_txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # 右上角置信度色块 + 数字
    cb = conf_color_bgr(conf)
    bar_w = 64
    cv2.rectangle(crop, (Wc - bar_w - 4, 4), (Wc - 4, 24), cb, -1)
    cv2.putText(
        crop,
        f"{conf:.2f}",
        (Wc - bar_w, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return crop


def _annotate_full_light(
    fr: np.ndarray,
    prior: Optional[Tuple[float, float, int]],
    det: Optional[TrafficLightDetector],
    label_txt: str,
    conf: float = 0.0,
) -> np.ndarray:
    """原始整帧(放大图): 蓝框=搜索区ROI, 黄圈=读取点。"""
    H, W = fr.shape[:2]
    out = fr.copy()
    cx = cy = None
    if prior is not None:
        cx, cy, roi = prior
        ax, ay = int(cx * W), int(cy * H)
        x1, y1 = max(0, ax - roi), max(0, ay - roi)
        x2, y2 = min(W, ax + roi), min(H, ay + roi)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 178, 102), 3)
    c = _detect_center(fr, det)
    if c is None:
        c = (cx, cy) if prior is not None else (0.5, 0.5)
    ccx = max(0, min(W - 1, int(c[0] * W)))
    ccy = max(0, min(H - 1, int(c[1] * H)))
    cv2.circle(out, (ccx, ccy), 18, (0, 255, 255), 3)
    cv2.drawMarker(out, (ccx, ccy), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
    cv2.rectangle(out, (0, 0), (W - 1, 30), (0, 0, 0), -1)
    cv2.putText(
        out,
        label_txt + "   蓝框=搜索区ROI   黄圈=读取点",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cb = conf_color_bgr(conf)
    cv2.rectangle(out, (W - 130, 4), (W - 8, 26), cb, -1)
    cv2.putText(
        out,
        f"置信度 {conf:.2f}",
        (W - 126, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return out
