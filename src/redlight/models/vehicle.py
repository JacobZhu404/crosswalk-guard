"""L3 车辆检测 (三后端: ultralytics YOLOv8n > cv2.dnn ONNX > 纯CV兜底)。

继承 BaseModel 统一接口; 内置轻量 IoU 跟踪器给每辆车稳定 id。
"""
import os
import cv2
import numpy as np

from ..models.base_model import BaseModel, ModelInfo, Detection
from ..infrastructure.config import project_root
from ..infrastructure.geometry import iou

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

try:
    from ultralytics import YOLO
    _HAS_ULT = True
except Exception:
    _HAS_ULT = False

try:
    from torch import cuda as _torch_cuda  # noqa
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def extract_light_boxes(raw_dets, conf_min=0.25):
    """从原始检测(每项含 name/xyxy/conf)过滤 COCO 'traffic light' 框。

    返回 [(x1,y1,x2,y2), ...] 像素整数框。用于给 M1 做候选(spec M1-D2)。
    """
    out = []
    for d in raw_dets:
        if d.get("name") == "traffic light" and float(d.get("conf", 0.0)) >= conf_min:
            out.append(tuple(int(v) for v in d["xyxy"]))
    return out


class SimpleTracker:
    """基于 IoU 的轻量多目标跟踪器。"""

    def __init__(self, max_disappeared=15, iou_thresh=0.3):
        self.next_id = 1
        self.tracks = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.iou_thresh = iou_thresh

    def update(self, dets):
        if not self.tracks:
            for d in dets:
                d["id"] = self.next_id
                self.tracks[self.next_id] = d["xyxy"]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return dets

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[t] for t in track_ids]
        matched_tr, matched_det = set(), set()
        order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
        for di in order:
            best_iou, best_t = -1.0, None
            for ti, tid in enumerate(track_ids):
                if tid in matched_tr:
                    continue
                iou_val = iou(dets[di]["xyxy"], track_boxes[ti])
                if iou_val > best_iou:
                    best_iou, best_t = iou_val, tid
            if best_t is not None and best_iou >= self.iou_thresh:
                dets[di]["id"] = best_t
                self.tracks[best_t] = dets[di]["xyxy"]
                self.disappeared[best_t] = 0
                matched_tr.add(best_t)
                matched_det.add(di)
        for di, d in enumerate(dets):
            if di not in matched_det:
                d["id"] = self.next_id
                self.tracks[self.next_id] = d["xyxy"]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        for tid in track_ids:
            if tid not in matched_tr:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
        return dets


class VehicleDetector(BaseModel):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        self.tracker = SimpleTracker()
        self.last_light_boxes = []   # M1: 同一次 YOLO 推理暴露的 COCO traffic-light 框
        self.classes = set(getattr(cfg, "vehicle_classes", ["car", "bus", "truck", "motorcycle"]))
        self.imgsz = getattr(cfg.inference, "imgsz", 640)
        self.conf = getattr(cfg.inference, "conf_thres", 0.35)
        self.iou = getattr(cfg.inference, "iou_thres", 0.45)
        self.use_ultra = False
        self.use_onnx = False
        self.model = None
        self.net = None
        self._backend = "cv"
        self._vb = verbose
        self.load()

    # ---- BaseModel ----
    def load(self, weights_path=None):
        # 1) ultralytics YOLOv8n (torch CPU)
        if _HAS_ULT:
            try:
                # 读 config 的 models.vehicle(如 "models/yolov8n.pt"); 相对路径按工程根解析。
                # 兼容旧字段 vehicle_pt; 都缺省时回退工程根 yolov8n.pt。
                cfg_path = getattr(self.cfg.models, "vehicle", None) or \
                    getattr(self.cfg.models, "vehicle_pt", None)
                if cfg_path and not os.path.isabs(cfg_path):
                    cfg_path = os.path.join(project_root(), cfg_path)
                yolo_path = weights_path or cfg_path or \
                    os.path.join(project_root(), "yolov8n.pt")
                if yolo_path.endswith(".pt") and os.path.isfile(yolo_path):
                    self.model = YOLO(yolo_path)
                    self.use_ultra = True
                    self._backend = "ultralytics"
                    self._loaded = True
                    if self._verbose:
                        print(f"[车辆检测] 使用 ultralytics YOLOv8n ({yolo_path})")
                    return
                elif self._verbose:
                    print(f"[车辆检测] 未找到权重 {yolo_path}, 尝试 ONNX")
            except Exception as e:
                if self._verbose:
                    print(f"[车辆检测] ultralytics 初始化失败({e}), 尝试 ONNX")
        # 2) cv2.dnn ONNX
        model_path = weights_path or getattr(self.cfg.models, "vehicle", "")
        if model_path and os.path.isfile(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.use_onnx = True
                self._backend = "onnx"
                self._loaded = True
                if self._verbose:
                    print(f"[车辆检测] 使用 YOLOv8 ONNX 模型: {model_path}")
                return
            except Exception as e:
                if self._verbose:
                    print(f"[车辆检测] ONNX 加载失败({e}), 降级纯CV兜底")
        # 3) 纯CV兜底
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)
        self._last_gray = None
        self._frame_count = 0
        self._backend = "cv"
        self._loaded = True
        if self._verbose:
            print("[车辆检测] 未找到权重 -> 纯CV兜底(背景减除)")

    @property
    def _verbose(self):
        return getattr(self, "_vb", True)

    def get_info(self) -> ModelInfo:
        return ModelInfo(name="VehicleDetector", version=self._backend,
                         classes=sorted(self.classes), input_size=(self.imgsz, self.imgsz))

    # ---- 推理 ----
    def infer(self, frame: np.ndarray):
        return self.detect(frame)

    def detect(self, frame):
        if self.use_ultra:
            dets = self._detect_ultra(frame)
        elif self.use_onnx:
            dets = self._detect_model(frame)
        else:
            dets = self._detect_cv(frame)
        return self.tracker.update(dets)

    def _detect_ultra(self, frame):
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)
        dets = []
        raw = []
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                name = self.model.names[cls]
                xyxy = [float(v) for v in b.xyxy[0].tolist()]
                raw.append({"name": name, "xyxy": xyxy, "conf": float(b.conf[0])})
                if name not in self.classes:
                    continue
                x1, y1, x2, y2 = xyxy
                dets.append({"id": -1, "xyxy": [x1, y1, x2, y2],
                             "conf": float(b.conf[0]), "cls": name})
        # M1: 复用同一次推理, 额外暴露 traffic-light 框(不进车辆跟踪)
        self.last_light_boxes = extract_light_boxes(raw, conf_min=0.25)
        return dets

    def _detect_model(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        pred = out[0].T
        cx = pred[:, 0]; cy = pred[:, 1]; bw = pred[:, 2]; bh = pred[:, 3]
        cls_scores = pred[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
        boxes, keep_confs, keep_cls = [], [], []
        for i in range(pred.shape[0]):
            if confs[i] < self.conf:
                continue
            name = COCO_CLASSES[cls_ids[i]]
            if name not in self.classes:
                continue
            x1 = (cx[i] - bw[i] / 2) / self.imgsz * w
            y1 = (cy[i] - bh[i] / 2) / self.imgsz * h
            x2 = (cx[i] + bw[i] / 2) / self.imgsz * w
            y2 = (cy[i] + bh[i] / 2) / self.imgsz * h
            boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            keep_confs.append(float(confs[i]))
            keep_cls.append(int(cls_ids[i]))
        dets = []
        if boxes:
            idxs = cv2.dnn.NMSBoxes(boxes, keep_confs, self.conf, self.iou)
            if len(idxs) > 0:
                idxs = [int(j[0]) if isinstance(j, (list, tuple, np.ndarray)) else int(j) for j in idxs]
                for i in idxs:
                    x, y, bw_i, bh_i = boxes[i]
                    dets.append({"id": -1, "xyxy": [x, y, x + bw_i, y + bh_i],
                                 "conf": keep_confs[i], "cls": COCO_CLASSES[keep_cls[i]]})
        return dets

    def _detect_cv(self, frame):
        self._frame_count += 1
        fg = self.bg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_gray is not None:
            diff = cv2.absdiff(gray, self._last_gray)
            _, diff_bin = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(fg, diff_bin)
        else:
            mask = fg
        self._last_gray = gray
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = frame.shape[:2]
        min_area = (W * H) * 0.004
        dets = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area:
                continue
            ar = w / float(h)
            if ar < 0.4 or ar > 4.0:
                continue
            dets.append({"id": -1, "xyxy": [float(x), float(y), float(x + w), float(y + h)],
                         "conf": 0.6, "cls": "car"})
        return dets
