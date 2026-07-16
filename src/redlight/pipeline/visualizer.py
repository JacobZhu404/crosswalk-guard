"""L5 可视化: 在画面上绘制斑马线/车辆/红绿灯/车牌与违规高亮。

违规高亮语义 (§5.3.1): 仅在 斑马线行人绿灯/闪烁 且 车辆静止压线 时标红。
"""
import cv2
import numpy as np

from ..infrastructure.geometry import mask_to_contour, compute_overlap_ratio
from .tracker import SENSITIVITY_PRESETS


class Visualizer:
    def __init__(self, cfg, preset="balanced"):
        self.cfg = cfg
        # 与违规引擎保持一致: 用所选 preset 的 overlap 阈值(而非 cfg.crosswalk.overlap_ratio),
        # 否则标注视频里的红框/HUD 与 violations.csv 的判定对不上。
        p = SENSITIVITY_PRESETS.get(preset, SENSITIVITY_PRESETS["balanced"])
        self.overlap = p["overlap"]

    def draw(self, frame, dets, track_states, mask, light_state, plates=None, light_boxes=None):
        if isinstance(light_state, dict):
            light_state = light_state.get("state", "unknown")
        disp = frame.copy()
        if plates is None:
            plates = []
        if light_boxes is None:
            light_boxes = []

        # YOLO 信号灯框(红框) + 识别结果文字
        for b in light_boxes:
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(disp, f"TL:{str(light_state).upper()}",
                        (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if mask is not None:
            contour = mask_to_contour(mask)
            if contour is not None:
                overlay = disp.copy()
                cv2.fillPoly(overlay, [contour], (0, 255, 255))
                disp = cv2.addWeighted(overlay, 0.35, disp, 0.65, 0)

        for d in dets:
            tid = d["id"]
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            st = track_states.get(tid, {})
            stationary = st.get("stationary", False)
            ratio = compute_overlap_ratio(d["xyxy"], mask, footprint=0.5, denom="mask")
            violating = (
                stationary and ratio >= self.overlap
                and light_state in ("green", "flashing")
            )
            if violating:
                color, lw = (0, 0, 255), 3
            elif stationary:
                color, lw = (0, 255, 255), 2
            else:
                color, lw = (0, 200, 0), 1
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, lw)
            label = f"#{tid} {d['cls']}"
            if stationary:
                label += " STOP"
            cv2.putText(disp, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        light_color = {
            "red": (0, 0, 255), "green": (0, 200, 0),
            "flashing": (0, 165, 255), "yellow": (0, 215, 255),
            "unknown": (128, 128, 128),
        }.get(light_state, (128, 128, 128))
        cv2.rectangle(disp, (8, 8), (240, 40), (0, 0, 0), -1)
        cv2.putText(disp, f"LIGHT: {str(light_state).upper()}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, light_color, 2, cv2.LINE_AA)

        for p in plates:
            x1, y1, x2, y2 = [int(v) for v in p["xyxy"]]
            color = {"blue": (255, 150, 0), "green": (0, 200, 100),
                     "yellow": (0, 200, 255)}.get(p.get("color"), (200, 200, 200))
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            txt = p.get("text") or f"PLATE({p.get('color','?')})"
            cv2.putText(disp, txt, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        n_veh = len(dets)
        n_stop = sum(1 for d in dets if track_states.get(d["id"], {}).get("stationary"))
        n_viol = sum(1 for d in dets
                     if track_states.get(d["id"], {}).get("stationary")
                     and compute_overlap_ratio(d["xyxy"], mask, footprint=0.5, denom="mask") >= self.overlap
                     and light_state in ("green", "flashing"))
        hud = f"veh={n_veh} stop={n_stop} viol/rev={n_viol}"
        cv2.putText(disp, hud, (8, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return disp
