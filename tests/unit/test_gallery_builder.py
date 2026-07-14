"""BaseGalleryBuilder / LightGalleryBuilder / PlateGalleryBuilder 单元测试。

因环境可能无 cv2, 在模块导入前 mock cv2/numpy 以隔离图像 I/O。
"""
import sys
import os
import tempfile
import csv
from unittest.mock import MagicMock, patch

# 在导入被测模块前 mock cv2, 避免 ImportError
# 条件注入: 避免覆盖真实 cv2 或之前测试留下的 mock, 防止全局状态泄漏
if "cv2" not in sys.modules:
    _cv2_mock = MagicMock()
    _cv2_mock.imencode.return_value = (True, MagicMock(tobytes=lambda: b"fakejpg"))
    _cv2_mock.IMREAD_COLOR = 1
    _cv2_mock.FONT_HERSHEY_SIMPLEX = 0
    _cv2_mock.MARKER_CROSS = 1
    _cv2_mock.CAP_PROP_FPS = 5
    _cv2_mock.CAP_PROP_FRAME_COUNT = 7
    _cv2_mock.error = Exception
    _cv2_mock.VideoCapture = MagicMock()  # callable mock, 使 patch("cv2.VideoCapture") 可正常保存/恢复
    sys.modules["cv2"] = _cv2_mock

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.gallery_builder import BaseGalleryBuilder
from redlight.evaluation.light_gallery import LightGalleryBuilder, _compress_records, _timeline_html
from redlight.evaluation.plate_gallery import PlateGalleryBuilder


# ---------- Mock 子类 ----------

class _MockBuilder(BaseGalleryBuilder):
    @property
    def title(self):
        return "Mock Gallery"

    @property
    def toolbar_label(self):
        return "Mock"

    def intro_html(self):
        return "intro"

    def verdict_options(self):
        return [("v1", "V1")]

    def reason_options(self):
        return [("r1", "R1")]

    def is_mismatch(self, item, gt):
        return item.get("pred") != gt

    def annotate_crop(self, frame, item, gt):
        return frame

    def annotate_full(self, frame, item, gt):
        return frame

    def item_meta_html(self, item, gt):
        return f"pred={item.get('pred')} gt={gt}"

    def feedback_key(self, video, item):
        return (video, str(item.get("idx", "")))


# ---------- 测试 ABC ----------

class TestABC:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            BaseGalleryBuilder("/tmp/eval")

    def test_mock_builder_instantiable(self):
        b = _MockBuilder("/tmp/eval")
        assert b.eval_dir == "/tmp/eval"


# ---------- 测试反馈加载 ----------

class TestFeedbackLoad:
    def test_load_empty_when_no_file(self):
        b = _MockBuilder("/tmp/eval", feedback_path="/nonexistent.csv")
        assert b._feedback == {}

    def test_load_feedback_csv(self):
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "fb.csv")
            with open(fp, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["video", "t_sec", "verdict", "reason", "note"])
                w.writeheader()
                w.writerow({"video": "v1", "t_sec": "1.5", "verdict": "algo_wrong", "reason": "color", "note": "n1"})
            b = _MockBuilder("/tmp/eval", feedback_path=fp)
            assert b._feedback == {
                ("v1", "1.5"): {"video": "v1", "t_sec": "1.5", "verdict": "algo_wrong", "reason": "color", "note": "n1"}
            }

    def test_lookup_fallback(self):
        b = _MockBuilder("/tmp/eval")
        assert b._lookup_feedback(("v", "1")) == {"verdict": "", "reason": "", "note": ""}


# ---------- 测试采样 ----------

class TestSampling:
    def test_sample_empty(self):
        b = _MockBuilder("/tmp/eval")
        assert b._sample_representatives([]) == []

    def test_sample_below_max(self):
        b = _MockBuilder("/tmp/eval", max_crops=10)
        mm = [{"i": i} for i in range(5)]
        assert len(b._sample_representatives(mm)) == 5

    def test_sample_above_max(self):
        b = _MockBuilder("/tmp/eval", max_crops=3)
        mm = [{"i": i} for i in range(10)]
        sampled = b._sample_representatives(mm)
        assert len(sampled) == 3
        # 均匀采样: step=max(1,10//3)=3, 取 0,3,6
        assert sampled[0]["i"] == 0
        assert sampled[1]["i"] == 3
        assert sampled[2]["i"] == 6


# ---------- 测试 HTML 生成 ----------

class TestHtmlGeneration:
    def test_build_option_html(self):
        b = _MockBuilder("/tmp/eval")
        html = b._build_option_html([("a", "A"), ("b", "B")], "b", "--选--")
        assert '<option value="" selected>--选--</option>' not in html  # placeholder 无 selected
        assert '<option value="b" selected>B</option>' in html

    def test_render_page_contains_title(self):
        b = _MockBuilder("/tmp/eval")
        page = b._render_page("")
        assert "Mock Gallery" in page
        assert "Mock" in page
        assert "intro" in page

    def test_build_creates_html_file(self):
        with tempfile.TemporaryDirectory() as td:
            eval_dir = os.path.join(td, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            # mock _resolve_frame 返回假图
            b = _MockBuilder(eval_dir, max_crops=2)
            b._resolve_frame = lambda video, idx: np.zeros((10, 10, 3), dtype=np.uint8)
            b.build({"v1": [{"idx": 0, "frame_idx": 0, "t_sec": 0.0, "pred": "red"}]}, {"v1": "green"})
            out = os.path.join(eval_dir, "gallery.html")
            assert os.path.exists(out)
            with open(out, encoding="utf-8") as f:
                content = f.read()
            assert "v1" in content
            assert "pred=red gt=green" in content


# ---------- 测试 LightGalleryBuilder 专有逻辑 ----------

class TestLightGalleryHelpers:
    def test_compress_records(self):
        recs = [
            {"pred": "red", "t_sec": "0.0"},
            {"pred": "red", "t_sec": "1.0"},
            {"pred": "green", "t_sec": "2.0"},
        ]
        segs = _compress_records(recs, "pred")
        assert segs == [["red", 0.0, 1.0], ["green", 2.0, 2.0]]

    def test_timeline_html(self):
        segs = [["red", 0.0, 5.0], ["green", 5.0, 10.0]]
        html = _timeline_html(segs, 10.0, "Test")
        assert "Test" in html
        assert "#ef4444" in html  # red color
        assert "#22c55e" in html  # green color

    def test_light_mismatch(self):
        b = LightGalleryBuilder("/tmp/eval")
        assert b.is_mismatch({"pred": "red", "gt": "green"}, None) is True
        assert b.is_mismatch({"pred": "red", "gt": "red"}, None) is False

    def test_light_feedback_key(self):
        b = LightGalleryBuilder("/tmp/eval")
        assert b.feedback_key("v1", {"t_sec": 1.5}) == ("v1", "1.5")

    def test_light_extra_attrs(self):
        b = LightGalleryBuilder("/tmp/eval")
        attrs = b.extra_data_attrs("v1", {"pred": "red", "gt": "green"}, None)
        assert attrs == {"pred": "red", "gt": "green"}


# ---------- 测试 PlateGalleryBuilder 专有逻辑 ----------

class TestPlateGalleryHelpers:
    def test_plate_mismatch(self):
        b = PlateGalleryBuilder("/tmp/eval")
        assert b.is_mismatch({"detected": "京A12345"}, ["京B99999"]) is True
        assert b.is_mismatch({"detected": "京A12345"}, ["京A12345"]) is False
        assert b.is_mismatch({"detected": "京A12345"}, []) is True
        assert b.is_mismatch({"detected": "京A12345"}, None) is True

    def test_plate_feedback_key(self):
        b = PlateGalleryBuilder("/tmp/eval")
        assert b.feedback_key("v1", {"frame_idx": 42}) == ("v1", "42")

    def test_plate_video_header(self):
        b = PlateGalleryBuilder("/tmp/eval")
        items = [{"t_sec": 5.0}, {"t_sec": 10.0}]
        html = b.video_header_html("v1", items, ["京A12345"])
        assert "京A12345" in html
        assert "10s" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
