"""L3 斑马线检测 —— 经典 CV 兜底 (无需权重)。

策略 v7: 中下部ROI灰度阈值+形态学+多轮廓筛选, 已通过实际视频验证。
"""
import cv2
import numpy as np

from ..models.base_model import BaseModel, ModelInfo


class CrosswalkDetector(BaseModel):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        self.method = getattr(cfg.crosswalk, "method", "auto")
        self.cv_min_area = getattr(cfg.crosswalk, "cv_min_area", 1500)
        if self.method not in ("auto", "cv", "segment"):
            self.method = "cv"
        self._loaded = True
        self._vb = verbose

    def load(self, weights_path=None):
        self._loaded = True

    def get_info(self) -> ModelInfo:
        return ModelInfo(name="CrosswalkDetector", version="cv-v7",
                         classes=["crosswalk"], input_size=(0, 0))

    def infer(self, frame: np.ndarray):
        return self.detect(frame)

    def detect(self, frame):
        return self._cv(frame)

    def _cv(self, frame):
        h, w = frame.shape[:2]
        y0 = int(h * 0.45)
        roi = frame[y0:, :]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 40 or roi_w < 80:
            return np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_h, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((h, w), dtype=np.uint8)
        valid_pts = []
        max_area = roi_w * roi_h * 0.35
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cv_min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            ar = bw / float(bh) if bh > 0 else 0
            if ar < 2.0:
                continue
            if bw < 40 or bh < 5:
                continue
            if y < int(roi_h * 0.30) and ar < 5.0:
                continue
            valid_pts.append(c + np.array([0, y0]))
        if valid_pts:
            cv2.drawContours(mask, valid_pts, -1, 255, cv2.FILLED)
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 10))
            mask = cv2.dilate(mask, k_d, iterations=1)
        return mask
