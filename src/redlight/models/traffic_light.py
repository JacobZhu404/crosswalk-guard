"""L3 红绿灯状态检测 v2 (strengthened): 斑马线行人信号灯识别。

强化点(相对 v1 颜色兜底, 落实 Q3 "不给算法减负"):
  - 候选灯连通域校验: 最小面积比 / 长宽比(拒绝细长尾灯) / 位置先验(画面上中部)
  - 时序持续性(persistence): 随车移动、时隐时现的红块(尾灯)判 unknown, 避免假红灯放过真违规
  - 偏向安全侧: red 与 green 并存且 red 非压倒性时, 偏向 green/flashing
  - 闪烁检测: 窗口内可见性跳变>=2 -> flashing
  - 富输出: {state, confidence, stable, is_flashing, reason}
信号类型假设: 斑马线行人灯 (见设计文档 §5.3.1 / Q5)
"""
import cv2
import numpy as np
from collections import Counter

from ..models.base_model import BaseModel, ModelInfo


class TrafficLightDetector(BaseModel):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        tl = getattr(cfg, "traffic_light", None)
        self.window = int(getattr(tl, "smoothing_window", 8))
        self.min_area_ratio = float(getattr(tl, "min_area_ratio", 0.0006))
        self.max_aspect = float(getattr(tl, "max_aspect_ratio", 3.0))
        self.band = tuple(getattr(tl, "search_band", (0.0, 0.65)))
        self.flicker_toggle = int(getattr(tl, "flicker_toggle_count", 2))
        self.min_pixels = int(getattr(tl, "color_min_pixels", 60))
        self.persist_min = float(getattr(tl, "persistence_min_ratio", 0.5))
        self.history = []   # 每帧: {"state","conf","box","present"}
        self._loaded = True
        self._vb = verbose

    def load(self, weights_path=None):
        self._loaded = True

    def get_info(self):
        return ModelInfo(name="TrafficLightDetector", version="color-v2-strengthened",
                         classes=["red", "green", "flashing", "unknown"], input_size=(0, 0))

    def infer(self, frame):
        return self.detect(frame)

    # ---------- 主入口 ----------
    def detect(self, frame):
        cand = self._best_candidate(frame)
        entry = self._to_entry(cand)
        self.history.append(entry)
        if len(self.history) > self.window:
            self.history.pop(0)
        return self._aggregate()

    # ---------- 候选灯检测(强化) ----------
    def _best_candidate(self, frame):
        if frame is None:
            return None
        h, w = frame.shape[:2]
        y0, y1 = int(h * self.band[0]), int(h * self.band[1])
        roi = frame[y0:y1, :]
        if roi.size == 0:
            return None
        rh, rw = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red = cv2.bitwise_or(r1, r2)
        green = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([90, 255, 255]))
        min_area = max(self.min_pixels, int(self.min_area_ratio * rh * rw))
        best = None
        for color, mask in (("red", red), ("green", green)):
            num, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, num):
                a = int(stats[i, cv2.CC_STAT_AREA])
                if a < min_area:
                    continue
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                if bw <= 0 or bh <= 0:
                    continue
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if aspect > self.max_aspect:   # 细长 -> 尾灯/反光, 拒绝
                    continue
                # 评分: 面积为主, 位置先验(上中部更高)为辅
                pos_y = (y + bh / 2) / rh       # 0=顶, 1=底
                pos_weight = 1.0 + (1.0 - pos_y) * 0.5
                score = a * pos_weight
                if best is None or score > best["score"]:
                    best = {"color": color, "box": (x, y + y0, x + bw, y + y0 + bh),
                            "centroid": (float(cents[i, 0]), float(cents[i, 1]) + y0),
                            "area": a, "score": score}
        return best

    def _to_entry(self, cand):
        if cand is None:
            return {"state": "unknown", "conf": 0.0, "box": None, "present": False}
        return {"state": cand["color"], "conf": min(1.0, cand["area"] / 2500.0),
                "box": cand["box"], "present": True}

    # ---------- 时序聚合(持续性/闪烁/偏向) ----------
    def _aggregate(self):
        if not self.history:
            return {"state": "unknown", "confidence": 0.0, "stable": False,
                    "is_flashing": False, "reason": "empty"}
        present = [e for e in self.history if e["present"]]
        persist = len(present) / len(self.history)
        # 闪烁: 可见性跳变次数
        seq = [1 if e["present"] else 0 for e in self.history]
        toggles = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        known = [e["state"] for e in present]
        if not known:
            return {"state": "unknown", "confidence": 0.0, "stable": persist >= self.persist_min,
                    "is_flashing": False, "reason": "no_signal"}
        cnt = Counter(known)
        dom = cnt.most_common(1)[0][0]
        dom_ratio = cnt[dom] / len(known)
        green_seen = cnt.get("green", 0) > 0
        red_dominant = (cnt.get("red", 0) > cnt.get("green", 0)) and (dom_ratio >= 0.6)
        if toggles >= self.flicker_toggle and green_seen:
            state, reason = "flashing", "flicker_green"
        elif green_seen and not red_dominant:
            # 安全侧: 有绿就偏绿(除非红明显压倒), 呼应 Q1 偏向多报
            state, reason = "green", "green_bias_safe"
        elif persist < self.persist_min:
            # 时隐时现的红块 -> 尾灯, 不误判红灯放过违规
            state, reason = "unknown", "intermittent_rejected"
        else:
            state, reason = dom, "stable_lamp"
        conf = float(np.mean([e["conf"] for e in present]))
        return {"state": state, "confidence": round(conf, 3),
                "stable": persist >= self.persist_min,
                "is_flashing": state == "flashing", "reason": reason}
