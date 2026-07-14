"""L3 行人信号灯状态分类器 (M1 spec M1-D4): ROI -> {walk,stand,off}。

权重格式按扩展名自动识别:
  - .pt / .pth : 直接 torch 加载(本项目运行时已绑 torch, 零额外依赖, 主路径)
  - .onnx      : cv2.dnn 加载(跨 Mac/Windows, 不依赖 torch, 可选导出产物)
无权重或加载失败时 available=False, 编排层回退到 color 路径。
训练/导出见 scripts/train_ped_signal.py (Phase 2)。
"""
import os
import numpy as np

LABELS = ["walk", "stand", "off"]
_INPUT = 48


def _build_net():
    """tiny CNN (3x48x48 -> 3)。与 SignalStateClassifier 的输入约定一致(/255, NCHW, 48x48)。

    单一真相源: 训练脚本与分类器都从这里取网络结构, 避免 state_dict 与架构对不上。
    """
    import torch.nn as nn
    # 固定尺寸(无 adaptive pool, 便于导出/对齐): 48->24->12, 16*12*12=2304
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 12 * 12, 3),
    )


class SignalStateClassifier:
    def __init__(self, model_path=None, verbose=True):
        self.model_path = model_path
        self.net = None
        self.available = False
        self._infer = None  # 后端推理函数, 由加载方式决定

        if not model_path or not os.path.isfile(model_path):
            return

        ext = os.path.splitext(model_path)[1].lower()
        try:
            if ext == ".onnx":
                self._load_onnx(model_path, verbose)
            elif ext in (".pt", ".pth"):
                self._load_torch(model_path, verbose)
            else:
                if verbose:
                    print(f"[信号分类器] 不支持的权重格式: {model_path} (仅支持 .pt/.pth/.onnx)")
        except Exception as e:
            if verbose:
                print(f"[信号分类器] 加载失败({e}) -> 回退 color 路径")

    def _load_onnx(self, model_path, verbose):
        import cv2
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._infer = self._infer_onnx
        self.available = True
        if verbose:
            print(f"[信号分类器] ONNX 已加载: {model_path}")

    def _load_torch(self, model_path, verbose):
        import torch
        net = _build_net()
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        self.net = net
        self._infer = self._infer_torch
        self.available = True
        if verbose:
            print(f"[信号分类器] PyTorch 权重已加载: {model_path}")

    def _preprocess(self, roi_bgr):
        import cv2
        img = cv2.resize(roi_bgr, (_INPUT, _INPUT)).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[None, ...]   # NCHW

    def _infer_onnx(self, blob):
        self.net.setInput(blob)
        return self.net.forward().reshape(-1)

    def _infer_torch(self, blob):
        import torch
        with torch.no_grad():
            t = torch.from_numpy(blob)
            return self.net(t).reshape(-1).numpy()

    def classify(self, roi_bgr):
        """返回 (label, conf)。不可用或 ROI 空 -> ('off', 0.0)。"""
        if not self.available or roi_bgr is None or roi_bgr.size == 0:
            return ("off", 0.0)
        probs = self._infer(self._preprocess(roi_bgr))
        i = int(np.argmax(probs))
        return (LABELS[i], float(probs[i]))
