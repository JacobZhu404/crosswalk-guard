"""L3 车牌识别 (HyperLPR3 HIGH 优先, 纯CV颜色定位兜底)。

返回列表元素: {"xyxy":[x1,y1,x2,y2], "text":str, "conf":float, "color":str}

后处理: 中国车牌格式校验 (省份简称+字母+5~6位普通车牌, 或8位新能源车牌)
"""
import cv2
import numpy as np
import re

from ..models.base_model import BaseModel, ModelInfo

try:
    from hyperlpr3 import LicensePlateCatcher, DETECT_LEVEL_HIGH
    _HAS_HL = True
except Exception:
    _HAS_HL = False

_CHINA_PROVINCE = {
    "京", "津", "冀", "晋", "蒙", "辽", "吉", "黑",
    "沪", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
    "鄂", "湘", "粤", "桂", "琼", "渝", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新",
    "使", "领", "警", "学", "港", "澳",
}

_PROVINCE_ALIASES = {
    "京": {"京", "冀", "津", "晋"},
    "粤": {"粤", "桂", "琼"},
    "沪": {"沪", "苏", "浙"},
    "川": {"川", "渝", "贵", "云"},
}

_PLATE_REGEX = {
    "normal": re.compile(r"^([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领警学港澳][A-Z][A-Z0-9]{5})$"),
    "new_energy": re.compile(r"^([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领警学港澳][A-Z][A-Z0-9]{6})$"),
}


def _is_valid_plate(text):
    """校验中国车牌格式是否合法。"""
    if not text:
        return False
    if len(text) not in (7, 8):
        return False
    if text[0] not in _CHINA_PROVINCE:
        return False
    if not text[1].isupper():
        return False
    rest = text[2:]
    if not rest.isalnum():
        return False
    if len(text) == 7:
        return _PLATE_REGEX["normal"].match(text) is not None
    return _PLATE_REGEX["new_energy"].match(text) is not None


def _filter_by_format(plates, min_conf=0.6):
    """过滤不符合中国车牌格式的识别结果。"""
    filtered = []
    for p in plates:
        txt = p.get("text", "")
        conf = p.get("conf", 0.0)
        if _is_valid_plate(txt):
            filtered.append(p)
    return filtered


def _apply_province_prior(plates, prior_province="京", boost_confidence=0.1):
    """应用省份先验知识: 完全过滤非先验省份的车牌, 并提升先验省份车牌的置信度。
    
    Args:
        plates: 车牌识别结果列表
        prior_province: 优先省份 (如"京"), 为空则不启用
        boost_confidence: 对匹配省份的车牌增加的置信度值
    """
    if not prior_province:
        return plates
    
    prior_set = _PROVINCE_ALIASES.get(prior_province, {prior_province})
    
    filtered = []
    for p in plates:
        txt = p.get("text", "")
        if not txt:
            filtered.append(p)
            continue
        if txt[0] in prior_set:
            p = p.copy()
            p["conf"] = min(1.0, p.get("conf", 0.0) + boost_confidence)
            filtered.append(p)
    
    return filtered


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
        self.prior_province = getattr(cfg.inference, "plate_prior_province", "京")
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
            plates = self._detect_hl(frame)
        else:
            plates = self._detect_cv(frame, vehicle_boxes)
        plates = _filter_by_format(plates, min_conf=self.conf_thres)
        if self.prior_province:
            plates = _apply_province_prior(plates, self.prior_province)
        return plates

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
