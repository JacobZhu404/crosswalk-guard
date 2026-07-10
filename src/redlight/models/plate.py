"""L3 车牌识别 (HyperLPR3 HIGH 优先, 纯CV颜色定位兜底)。

返回列表元素: {"xyxy":[x1,y1,x2,y2], "text":str, "conf":float, "color":str}
"""
import cv2
import numpy as np

from ..models.base_model import BaseModel, ModelInfo

try:
    from hyperlpr3 import LicensePlateCatcher, DETECT_LEVEL_HIGH
    _HAS_HL = True
except Exception:
    _HAS_HL = False


def _plate_color(frame, box):
    """按车牌区域主色调粗略判断颜色 (blue/yellow/green/unknown)。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > 60) & (hsv[:, :, 1] > 40)
    if mask.sum() < 10:
        return "unknown"
    h = hsv[:, :, 0][mask].astype(int)
    hist = np.bincount(h, minlength=180)
    dom = int(np.argmax(hist))
    if 100 <= dom <= 130:
        return "blue"
    if 15 <= dom <= 40:
        return "yellow"
    if 35 <= dom <= 85:
        return "green"
    return "unknown"


class PlateRecognizer(BaseModel):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        self.conf_thres = getattr(cfg.inference, "plate_conf", 0.40)
        self.use_hl = False
        self.catcher = None
        self._backend = "cv"
        self._vb = verbose
        self.load()

    def load(self, weights_path=None):
        if _HAS_HL:
            try:
                self.catcher = LicensePlateCatcher(detect_level=DETECT_LEVEL_HIGH)
                self.use_hl = True
                self._backend = "hyperlpr3"
                self._loaded = True
                if self._vb:
                    print("[车牌] 使用 HyperLPR3 (HIGH) 检测+识别")
                return
            except Exception as e:
                if self._vb:
                    print(f"[车牌] HyperLPR3 初始化失败({e}), 降级纯CV定位")
        self._backend = "cv"
        self._loaded = True
        if self._vb:
            print("[车牌] HyperLPR3 不可用 -> 纯CV颜色定位(仅定位, 不识别)")

    def get_info(self) -> ModelInfo:
        return ModelInfo(name="PlateRecognizer", version=self._backend,
                         classes=["plate"], input_size=(0, 0))

    def infer(self, frame: np.ndarray):
        return self.detect(frame)

    def detect(self, frame, vehicle_boxes=None):
        if self.use_hl:
            return self._detect_hl(frame)
        return self._detect_cv(frame, vehicle_boxes)

    def _detect_hl(self, frame):
        res = self.catcher(frame)
        out = []
        for r in res:
            txt = str(r[0]); conf = float(r[1]); box = r[3]
            if conf < self.conf_thres:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            out.append({"xyxy": [x1, y1, x2, y2], "text": txt, "conf": conf,
                        "color": _plate_color(frame, box)})
        return out

    def _detect_cv(self, frame, vehicle_boxes=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, W = frame.shape[:2]
        blue = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))
        green = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
        yellow = cv2.inRange(hsv, np.array([20, 100, 120]), np.array([32, 255, 255]))
        mask = cv2.bitwise_or(blue, cv2.bitwise_or(green, yellow))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 30 or h < 8:
                continue
            ar = w / float(h)
            if ar < 1.5 or ar > 6.0:
                continue
            out.append({"xyxy": [float(x), float(y), float(x + w), float(y + h)],
                        "text": "", "conf": 0.0, "color": "unknown"})
        return out
