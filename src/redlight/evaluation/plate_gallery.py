"""车牌复核画廊 (BaseGalleryBuilder 实现)。

用途: 用户滚动逐张判定 mismatch 帧是 "算法错 / 标注错 / 难度太大 / 其他"。
风格与 light_gallery 保持一致, 便于统一标注体验。
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..infrastructure.image_utils import conf_color_bgr
from .gallery_builder import BaseGalleryBuilder


class PlateGalleryBuilder(BaseGalleryBuilder):
    """车牌识别误差确认画廊生成器。

    gt_map 期望: {video: [plate_text, ...]}
    video_items 期望: {video: [{"frame_idx": int, "t_sec": float,
                                 "detected": str, "conf": float,
                                 "plate_info": dict, "all_plates": list}, ...]}
    """

    # ---- 子类必须实现 ----

    @property
    def title(self) -> str:
        return "车牌识别误差确认画廊"

    @property
    def toolbar_label(self) -> str:
        return "车牌误差确认画廊"

    def intro_html(self) -> str:
        return (
            "左=车牌区域特写("
            '<span style="color:#dc2626;font-weight:700;">红框=检测框</span>'
            ": 算法识别的车牌位置)。<b>点小图看原始整帧</b>(红框=所有检测到的车牌)。"
            "请判定: <b>算法错</b>(识别结果与真实不符) / <b>标注错</b>(GT标注有误) / "
            "<b>难度太大</b>(遮挡/角度/光线等无法识别) / <b>其他</b>; "
            "原因按判定树选: <b>遮挡</b>(被其他车辆/物体挡住) → "
            "<b>角度问题</b>(侧拍/俯拍太偏) → <b>光线问题</b>(反光/过曝/过暗) → "
            "<b>模糊</b>(运动模糊/失焦) → <b>部分遮挡</b>(只看到部分车牌) / GT标注错误 / 其他。"
        )

    def verdict_options(self) -> List[Tuple[str, str]]:
        return [
            ("algo_wrong", "算法错"),
            ("label_wrong", "标注错"),
            ("too_hard", "难度太大"),
            ("other", "其他"),
        ]

    def reason_options(self) -> List[Tuple[str, str]]:
        return [
            ("occlusion", "遮挡(被其他车辆/物体挡住)"),
            ("angle", "角度问题(侧拍/俯拍太偏)"),
            ("light", "光线问题(反光/过曝/过暗)"),
            ("blur", "模糊(运动模糊/失焦)"),
            ("partial", "部分遮挡(只看到部分车牌)"),
            ("gt_error", "GT标注错误"),
            ("other", "其他"),
        ]

    def is_mismatch(self, item: dict, gt: Any) -> bool:
        """mismatch = detected text 不在 GT plates 列表中。"""
        if not gt or not isinstance(gt, list):
            return True
        detected = item.get("detected", "")
        return detected not in gt

    def annotate_crop(self, frame: np.ndarray, item: dict, gt: Any) -> np.ndarray:
        plate_info = item.get("plate_info", {})
        label_txt = (
            f"t={float(item.get('t_sec', 0)):.1f}s "
            f"识{item.get('detected', '')}/GT{','.join(gt) if gt else '?' }"
        )
        conf = float(item.get("conf", 0.0))
        return _annotate_crop_plate(frame, plate_info, label_txt, conf)

    def annotate_full(self, frame: np.ndarray, item: dict, gt: Any) -> np.ndarray:
        all_plates = item.get("all_plates", [item.get("plate_info", {})])
        label_txt = (
            f"t={float(item.get('t_sec', 0)):.1f}s "
            f"识{item.get('detected', '')}/GT{','.join(gt) if gt else '?' }"
        )
        conf = float(item.get("conf", 0.0))
        return _annotate_full_plate(frame, all_plates, label_txt, conf)

    def item_meta_html(self, item: dict, gt: Any) -> str:
        conf = float(item.get("conf", 0.0))
        conf_color = "#16a34a" if conf >= 0.85 else ("#d97706" if conf >= 0.6 else "#dc2626")
        gt_str = ",".join(gt) if gt else "?"
        return (
            f'识别 <b class="p-detected">{item.get("detected", "")}</b> / '
            f'GT <b class="p-gt">{gt_str}</b> '
            f'<span class="conf" style="background:{conf_color}">置信度 {conf:.2f}</span>'
        )

    def feedback_key(self, video: str, item: dict) -> Tuple[str, str]:
        return (video, str(item.get("frame_idx", "")))

    # ---- 可选覆写 ----

    def video_header_html(self, video: str, items: List[dict], gt: Any) -> str:
        gt_count = len(gt) if gt else 0
        mismatch_count = len([it for it in items if self.is_mismatch(it, gt)])
        # 估算时长: 取最大 t_sec
        total_dur = max((float(it.get("t_sec", 0)) for it in items), default=0)
        gt_str = ", ".join(gt) if gt else "未知"
        return (
            f'<div style="font-size:12px;margin:2px 0;color:#475569;">'
            f'GT车牌: {gt_str} · 时长: {total_dur:.0f}s · 待确认: {mismatch_count}帧</div>'
        )

    def extra_data_attrs(self, video: str, item: dict, gt: Any) -> Dict[str, str]:
        return {
            "detected": item.get("detected", ""),
            "gt": ",".join(gt) if gt else "",
        }


# ---------- 车牌专用标注函数 ----------


def _annotate_crop_plate(
    fr: np.ndarray,
    plate_info: dict,
    label_txt: str,
    conf: float = 0.0,
) -> np.ndarray:
    """车牌区域特写(小图): 红框=检测框。"""
    xyxy = plate_info.get("xyxy", [0, 0, 0, 0])
    x1, y1, x2, y2 = [int(v) for v in xyxy]

    H, W = fr.shape[:2]
    roi = 40
    x1_crop = max(0, x1 - roi)
    y1_crop = max(0, y1 - roi)
    x2_crop = min(W, x2 + roi)
    y2_crop = min(H, y2 + roi)

    crop = fr[y1_crop:y2_crop, x1_crop:x2_crop].copy()
    Hc, Wc = crop.shape[:2]
    if Hc < 4 or Wc < 4:
        # 回退: 返回原图左上角小块, 避免空图
        return fr[:200, :300].copy() if len(fr.shape) == 3 else fr[:200, :300]

    # 在原 crop 坐标系中画检测框
    cv2.rectangle(
        crop,
        (max(0, x1 - x1_crop), max(0, y1 - y1_crop)),
        (min(Wc - 1, x2 - x1_crop), min(Hc - 1, y2 - y1_crop)),
        (0, 0, 255),
        2,
    )

    cv2.rectangle(crop, (0, 0), (Wc - 1, 30), (0, 0, 0), -1)
    cv2.putText(crop, label_txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cb = conf_color_bgr(conf)
    bar_w = 70
    cv2.rectangle(crop, (Wc - bar_w - 4, 4), (Wc - 4, 26), cb, -1)
    cv2.putText(
        crop,
        f"{conf:.2f}",
        (Wc - bar_w, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    return crop


def _annotate_full_plate(
    fr: np.ndarray,
    all_plates: List[dict],
    label_txt: str,
    conf: float = 0.0,
) -> np.ndarray:
    """原始整帧(放大图): 红框=所有检测到的车牌。"""
    out = fr.copy()
    H, W = out.shape[:2]

    for p in all_plates:
        if not isinstance(p, dict):
            continue
        xyxy = p.get("xyxy", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = p.get("text", "")
        p_conf = p.get("conf", 0.0)
        cv2.putText(
            out,
            f"{text} ({p_conf:.2f})",
            (x1, max(10, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    cv2.rectangle(out, (0, 0), (W - 1, 32), (0, 0, 0), -1)
    cv2.putText(
        out,
        label_txt + "   红框=车牌检测框",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cb = conf_color_bgr(conf)
    cv2.rectangle(out, (W - 130, 4), (W - 8, 28), cb, -1)
    cv2.putText(
        out,
        f"置信度 {conf:.2f}",
        (W - 126, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return out
