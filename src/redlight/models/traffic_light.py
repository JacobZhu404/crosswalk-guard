"""L3 红绿灯状态检测 (red / green / yellow / unknown) —— 经典颜色兜底。

时间平滑: 滑动窗口抑制单帧闪烁; 拍不到灯时整体返回 unknown。
"""
import cv2
import numpy as np

from ..models.base_model import BaseModel, ModelInfo


class TrafficLightDetector(BaseModel):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        self.method = getattr(cfg.traffic_light, "method", "auto")
        self.window = getattr(cfg.traffic_light, "smoothing_window", 5)
        self.min_pixels = getattr(cfg.traffic_light, "color_min_pixels", 150)
        self.history = []
        if self.method not in ("auto", "model", "color"):
            self.method = "color"
        self._loaded = True
        self._vb = verbose

    def load(self, weights_path=None):
        self._loaded = True

    def get_info(self) -> ModelInfo:
        return ModelInfo(name="TrafficLightDetector", version="color-v1",
                         classes=["red", "green", "yellow", "unknown"], input_size=(0, 0))

    def infer(self, frame: np.ndarray):
        return self.detect(frame)

    def detect(self, frame):
        state = self._color(frame)
        self.history.append(state)
        if len(self.history) > self.window:
            self.history.pop(0)
        return self._smooth()

    def _color(self, frame):
        h, w = frame.shape[:2]
        roi = frame[: int(h * 0.7), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red = cv2.bitwise_or(r1, r2)
        green = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([90, 255, 255]))
        red_area = self._max_blob_area(red)
        green_area = self._max_blob_area(green)
        if red_area < self.min_pixels and green_area < self.min_pixels:
            return "unknown"
        return "red" if red_area >= green_area else "green"

    @staticmethod
    def _max_blob_area(binary):
        if binary is None:
            return 0
        num, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if num <= 1:
            return 0
        return int(np.max(stats[1:, cv2.CC_STAT_AREA]))

    def _smooth(self):
        counts = {"red": 0, "green": 0, "yellow": 0, "unknown": 0}
        for s in self.history:
            counts[s] = counts.get(s, 0) + 1
        known = {k: v for k, v in counts.items() if k != "unknown"}
        if not known or max(known.values()) == 0:
            return "unknown"
        best = max(known, key=known.get)
        if counts["unknown"] >= max(known.values()):
            return "unknown"
        return best
