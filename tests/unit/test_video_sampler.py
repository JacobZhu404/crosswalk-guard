"""VideoSampler 单元测试。

覆盖: interval 计算、grab/read 模式采样、时间戳、上下文管理器、资源释放。
使用 mock 隔离 cv2.VideoCapture，不依赖真实视频文件。
"""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 为无 cv2 的环境注入 mock 模块，使 patch("cv2.VideoCapture") 能工作
if "cv2" not in sys.modules:
    _fake_cv2 = MagicMock()
    _fake_cv2.CAP_PROP_FPS = 5
    _fake_cv2.CAP_PROP_FRAME_COUNT = 7
    sys.modules["cv2"] = _fake_cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.video_sampler import VideoSampler


class TestIntervalCalculation:
    def test_exact_division(self):
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda k: 24.0 if k == 5 else 240  # fps=24, frames=240
            MockCap.return_value = cap
            vs = VideoSampler("dummy.mp4", sample_fps=8)
            assert vs.interval == 3  # 24/8=3
            vs.release()

    def test_rounding(self):
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda k: 25.0 if k == 5 else 250
            MockCap.return_value = cap
            vs = VideoSampler("dummy.mp4", sample_fps=8)
            assert vs.interval == 3  # round(25/8)=round(3.125)=3
            vs.release()

    def test_fps_lower_than_sample(self):
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda k: 5.0 if k == 5 else 50
            MockCap.return_value = cap
            vs = VideoSampler("dummy.mp4", sample_fps=8)
            assert vs.interval == 1  # max(1, round(5/8))=1
            vs.release()

    def test_default_fps_fallback(self):
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda k: 0.0 if k == 5 else 100
            MockCap.return_value = cap
            vs = VideoSampler("dummy.mp4", sample_fps=8)
            assert vs.fps == 25.0  # fallback
            assert vs.interval == 3  # round(25/8)=3
            vs.release()


class TestGrabModeSampling:
    def test_returns_only_sample_frames(self):
        cv2 = pytest.importorskip("cv2")
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda k: 10.0 if k == 5 else 100
            # grab: True x 100, retrieve returns frame on every 3rd call (0,3,6,...)
            cap.grab.side_effect = [True] * 100
            frames = [np.zeros((10, 10, 3), dtype=np.uint8) if i % 3 == 0 else None for i in range(100)]
            def retrieve_side_effect():
                f = frames[cap.grab.call_count - 1]
                return (f is not None), f
            cap.retrieve.side_effect = retrieve_side_effect
            MockCap.return_value = cap

            vs = VideoSampler("dummy.mp4", sample_fps=10/3, use_grab=True)
            # interval = round(10 / (10/3)) = round(3) = 3
            assert vs.interval == 3
            results = list(vs)
            # 应该返回 frame_idx = 0, 3, 6, 9, ...
            assert len(results) == 34  # floor(100/3) + 1 = 34 (0-based: 0,3,6,...,99)
            for i, (idx, ts, frame) in enumerate(results):
                assert idx == i * 3
                assert abs(ts - idx / 10.0) < 1e-9
                assert frame is not None
            vs.release()

    def test_skips_failed_retrieve(self):
        pytest.importorskip("cv2")
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda k: 10.0 if k == 5 else 10
            cap.grab.side_effect = [True] * 10
            # frame 3 retrieve fails
            retrieve_results = [
                (True, np.zeros((10, 10, 3))),   # 0
                (True, np.zeros((10, 10, 3))),   # 3
                (False, None),                    # 6 - fails
                (True, np.zeros((10, 10, 3))),   # 9
            ]
            cap.retrieve.side_effect = retrieve_results
            MockCap.return_value = cap

            vs = VideoSampler("dummy.mp4", sample_fps=10/3, use_grab=True)
            results = list(vs)
            # 0, 3, 9 应该返回；6 retrieve 失败被跳过
            assert len(results) == 3
            assert results[0][0] == 0
            assert results[1][0] == 3
            assert results[2][0] == 9
            vs.release()


class TestReadModeSampling:
    def test_returns_only_sample_frames(self):
        pytest.importorskip("cv2")
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda k: 10.0 if k == 5 else 10
            read_results = [
                (True, np.zeros((10, 10, 3))) if i % 3 == 0 else (True, np.zeros((10, 10, 3)))
                for i in range(10)
            ]
            cap.read.side_effect = read_results
            MockCap.return_value = cap

            vs = VideoSampler("dummy.mp4", sample_fps=10/3, use_grab=False)
            assert vs.interval == 3
            results = list(vs)
            assert len(results) == 4  # 0, 3, 6, 9
            for i, (idx, ts, frame) in enumerate(results):
                assert idx == i * 3
            vs.release()


class TestContextManager:
    def test_auto_release(self):
        pytest.importorskip("cv2")
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda k: 10.0 if k == 5 else 0
            cap.grab.return_value = False
            MockCap.return_value = cap

            with VideoSampler("dummy.mp4") as vs:
                pass
            assert cap.release.called

    def test_release_idempotent(self):
        pytest.importorskip("cv2")
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda k: 10.0 if k == 5 else 0
            cap.grab.return_value = False
            MockCap.return_value = cap

            vs = VideoSampler("dummy.mp4")
            vs.release()
            vs.release()  # 不应抛异常
            assert cap.release.called


class TestOpenFailure:
    def test_raises_on_open_failure(self):
        pytest.importorskip("cv2")
        with patch("cv2.VideoCapture") as MockCap:
            cap = MagicMock()
            cap.isOpened.return_value = False
            MockCap.return_value = cap
            with pytest.raises(IOError):
                VideoSampler("nonexistent.mp4")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
