"""image_utils 单元测试。

覆盖: save_jpg, conf_color_bgr, robust_imread。
不依赖模型，秒级跑完。
"""
import sys
import os
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.image_utils import save_jpg, conf_color_bgr, robust_imread


class TestSaveJpg:
    def test_roundtrip(self):
        cv2 = pytest.importorskip("cv2")
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :] = (128, 64, 32)  # BGR
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        try:
            assert save_jpg(img, path) is True
            assert os.path.getsize(path) > 0
            # 读取回来验证大致正确（JPEG 有损，只验证 shape）
            loaded = cv2.imread(path)
            assert loaded is not None
            assert loaded.shape == (10, 10, 3)
        finally:
            os.unlink(path)

    def test_invalid_image(self):
        pytest.importorskip("cv2")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        try:
            # 传入非 ndarray 或空数组应返回 False
            assert save_jpg(np.array([]), path) is False
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestConfColorBgr:
    def test_high_conf(self):
        assert conf_color_bgr(0.85) == (0, 180, 0)
        assert conf_color_bgr(1.0) == (0, 180, 0)

    def test_mid_conf(self):
        assert conf_color_bgr(0.6) == (0, 215, 230)
        assert conf_color_bgr(0.7) == (0, 215, 230)
        assert conf_color_bgr(0.849) == (0, 215, 230)

    def test_low_conf(self):
        assert conf_color_bgr(0.0) == (0, 0, 220)
        assert conf_color_bgr(0.599) == (0, 0, 220)


class TestRobustImread:
    def test_roundtrip(self):
        cv2 = pytest.importorskip("cv2")
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :] = (64, 128, 255)  # BGR
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            cv2.imwrite(path, img)
            loaded = robust_imread(path)
            assert loaded is not None
            assert loaded.shape == (10, 10, 3)
            # PNG 无损，像素应一致
            np.testing.assert_array_equal(loaded, img)
        finally:
            os.unlink(path)

    def test_missing_file(self):
        pytest.importorskip("cv2")
        assert robust_imread("/nonexistent/image.jpg") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
