"""视频帧采样器：统一封装 cv2.VideoCapture 的采样逻辑。

支持两种模式:
- grab 模式 (默认): 跳帧时仅 grab 不解码，效率更高
- read 模式: 逐帧 read，与旧脚本行为完全一致

用法:
    with VideoSampler(video_path, sample_fps=8) as sampler:
        for frame_idx, timestamp, frame in sampler:
            results = detector.detect(frame)
"""

import os
from typing import Iterator, Tuple

import numpy as np


class VideoSampler:
    def __init__(self, video_path: str, sample_fps: float = 8.0, use_grab: bool = True):
        """初始化视频采样器。

        Args:
            video_path: 视频文件路径
            sample_fps: 目标采样帧率
            use_grab: 是否使用 grab+retrieve 优化。True 时跳帧只 grab 不解码；
                      False 时每帧都 read（与旧 eval_light_all.py 行为一致）
        """
        self._cap = None
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.use_grab = use_grab

        self.fps = 25.0
        self.total_frames = 0
        self.interval = 1
        self._frame_idx = 0
        self._released = True

        self._open()

    def _open(self):
        import cv2
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.interval = max(1, int(round(self.fps / self.sample_fps)))
        self._frame_idx = 0
        self._released = False

    # ---------- 迭代器协议 ----------
    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        return self

    def __next__(self) -> Tuple[int, float, np.ndarray]:
        while True:
            if self.use_grab:
                ret = self._cap.grab()
                if not ret:
                    raise StopIteration
                if self._frame_idx % self.interval == 0:
                    ret, frame = self._cap.retrieve()
                    if not ret:
                        self._frame_idx += 1
                        continue
                    idx = self._frame_idx
                    ts = idx / self.fps
                    self._frame_idx += 1
                    return idx, ts, frame
                self._frame_idx += 1
            else:
                ret, frame = self._cap.read()
                if not ret:
                    raise StopIteration
                if self._frame_idx % self.interval == 0:
                    idx = self._frame_idx
                    ts = idx / self.fps
                    self._frame_idx += 1
                    return idx, ts, frame
                self._frame_idx += 1

    # ---------- 上下文管理器 ----------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def release(self):
        """释放视频资源。"""
        if self._cap is not None and not self._released:
            self._cap.release()
            self._released = True

    def __del__(self):
        self.release()
