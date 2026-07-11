"""L5 可视化: 在画面上绘制斑马线/车辆/红绿灯/车牌与违规高亮。

违规高亮语义 (§5.3.1): 仅在 斑马线行人绿灯/闪烁 且 车辆静止压线 时标红。
"""
import cv2
import numpy as np

from ..infrastructure.geometry import mask_to_contour, compute_overlap_ratio


class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.overlap = getattr(cfg.crosswalk, "overlap_ratio", 0.30)

    def draw(self, frame, dets, track_states, mask, light_state, plates=None):
        if isinstance(light_state, dict):
            light_state = light_state.get("state", "unknown")
        disp = frame.copy()
        if plates is None:
            plates = []

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
            ratio = compute_overlap_ratio(d["xyxy"], mask)
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
                     and compute_overlap_ratio(d["xyxy"], mask) >= self.overlap
                     and light_state in ("green", "flashing"))
        hud = f"veh={n_veh} stop={n_stop} viol/rev={n_viol}"
        cv2.putText(disp, hud, (8, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return disp
