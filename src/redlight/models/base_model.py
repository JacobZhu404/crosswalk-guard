"""L3 模型层: 统一模型接口 (BaseModel ABC)。

为什么重要?
- 当前 YOLO 返回 dets dict, HyperLPR3 返回 [text,conf,type,box], 格式不一
- 统一接口让评测 / A-B 测试 / 模型替换变得简单
- 所有检测器 / 识别器都继承此接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ModelInfo:
    name: str = "unknown"
    version: str = "unknown"
    classes: List[str] = field(default_factory=list)
    input_size: tuple = (640, 640)


@dataclass
class Detection:
    xyxy: List[float]
    conf: float = 0.0
    cls: str = ""
    id: int = -1


class BaseModel(ABC):
    def __init__(self):
        self._loaded = False

    @abstractmethod
    def load(self, weights_path: Optional[str] = None) -> None:
        """加载权重 / 初始化后端。"""

    @abstractmethod
    def infer(self, image: np.ndarray):
        """对单帧推理, 返回模型特定结果 (由子类定义结构)。"""

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """返回模型元信息 (名称/版本/类别/输入尺寸)。"""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def benchmark(self, image: np.ndarray, n_runs: int = 100) -> dict:
        """简单计时基准 (CPU 推理吞吐参考)。"""
        import time
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.infer(image)
        avg_ms = (time.perf_counter() - t0) / max(n_runs, 1) * 1000.0
        return {"n_runs": n_runs, "avg_ms": round(avg_ms, 2)}
