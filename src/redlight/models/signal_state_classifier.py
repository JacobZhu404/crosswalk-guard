"""L3 行人信号灯状态分类器 (M1 spec M1-D4): ROI -> {walk,stand,off}。

运行时用 cv2.dnn 跑 ONNX(不依赖 torch, 跨 Mac/Windows)。无权重时 available=False,
编排层回退到 color 路径。训练/导出见 Phase 2。
"""
import os
import numpy as np

LABELS = ["walk", "stand", "off"]
_INPUT = 48


class SignalStateClassifier:
    def __init__(self, model_path=None, verbose=True):
        self.model_path = model_path
        self.net = None
        self.available = False
        if model_path and os.path.isfile(model_path):
            try:
                import cv2
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.available = True
                if verbose:
                    print(f"[信号分类器] ONNX 已加载: {model_path}")
            except Exception as e:
                if verbose:
                    print(f"[信号分类器] ONNX 加载失败({e}) -> 回退 color 路径")

    def _preprocess(self, roi_bgr):
        import cv2
        img = cv2.resize(roi_bgr, (_INPUT, _INPUT)).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[None, ...]   # NCHW

    def _infer(self, blob):
        self.net.setInput(blob)
        out = self.net.forward().reshape(-1)
        return out

    def classify(self, roi_bgr):
        """返回 (label, conf)。不可用或 ROI 空 -> ('off', 0.0)。"""
        if not self.available or roi_bgr is None or roi_bgr.size == 0:
            return ("off", 0.0)
        probs = self._infer(self._preprocess(roi_bgr))
        i = int(np.argmax(probs))
        return (LABELS[i], float(probs[i]))
