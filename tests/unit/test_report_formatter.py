"""ReportFormatter 单元测试。

覆盖: 车牌评测控制台格式化、CSV 写入。
不依赖模型或 cv2，秒级跑完。
"""
import sys
import os
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.report_formatter import (
    format_plate_per_video,
    format_plate_summary,
    write_plate_csv,
)


def _make_result(video="违章02", matched=None, missed=None, false_positives=None):
    return {
        "video": video,
        "gt_plates": ["京LNE560"],
        "matched": matched or [],
        "missed": missed or [],
        "false_positives": false_positives or [],
        "stats": {
            "total_frames": 1000,
            "processed_frames": 125,
            "detected_plate_frames": 80,
            "unique_plates_detected": 3,
            "gt_plates_count": 1,
            "exact_matches": 1,
            "near_matches": 0,
            "missed": 0,
            "false_positives": 1,
            "accuracy": 1.0,
        },
    }


class TestFormatPlatePerVideo:
    def test_basic(self):
        r = _make_result(matched=[{
            "gt": "京LNE560", "detected": "京LNE560",
            "count": 42, "avg_conf": 0.92, "edit_dist": 0,
        }])
        lines = format_plate_per_video(r)
        assert "  处理帧数: 125/1000" in lines
        assert "  识别到车牌帧数: 80" in lines
        assert "  识别到独立车牌: 3" in lines
        assert "  ✅ 匹配结果:" in lines
        assert "    京LNE560 -> 京LNE560 [精确] 次数=42 平均conf=0.920" in lines
        assert "  ⚠️ 误检车牌: ['...']" not in lines  # false_positives 为空列表时不显示
        assert "  准确率: 100.0%" in lines

    def test_near_match(self):
        r = _make_result(matched=[{
            "gt": "京LNE560", "detected": "京LNE561",
            "count": 5, "avg_conf": 0.72, "edit_dist": 1,
        }])
        lines = format_plate_per_video(r)
        assert "[近似(ED=1)]" in lines[4]

    def test_missed_and_fp(self):
        r = _make_result(missed=["京ADH9206"], false_positives=["京A12345"])
        lines = format_plate_per_video(r)
        assert "  ❌ 未识别到: ['京ADH9206']" in lines
        assert "  ⚠️ 误检车牌: ['京A12345']" in lines

    def test_empty_matched(self):
        r = _make_result()
        r["stats"]["accuracy"] = 0.0
        lines = format_plate_per_video(r)
        assert "  ✅ 匹配结果:" not in lines
        assert "  准确率: 0.0%" in lines


class TestFormatPlateSummary:
    def test_basic(self):
        stats = {
            "videos_tested": 3,
            "gt_plates": 5,
            "exact_matches": 4,
            "near_matches": 1,
            "missed": 0,
            "false_positives": 2,
        }
        lines = format_plate_summary(stats)
        assert "测试视频数: 3" in lines
        assert "GT车牌总数: 5" in lines
        assert "精确匹配: 4" in lines
        assert "近似匹配(ED=1): 1" in lines
        assert "整体准确率: 100.0%" in lines

    def test_zero_gt(self):
        stats = {
            "videos_tested": 0,
            "gt_plates": 0,
            "exact_matches": 0,
            "near_matches": 0,
            "missed": 0,
            "false_positives": 0,
        }
        lines = format_plate_summary(stats)
        assert "整体准确率: N/A" in lines


class TestWritePlateCsv:
    def test_roundtrip(self):
        results = [
            _make_result("违章02", missed=["京ADH9206"]),
            _make_result("违章03", false_positives=["京A11111"]),
        ]
        results[1]["stats"]["accuracy"] = 0.5
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        os.unlink(path)  # delete so writer can create it
        try:
            write_plate_csv(results, path)
            with open(path, "r", encoding="utf-8-sig") as f:
                rows = list(f)
            assert rows[0].strip() == "video,gt_plates,detected_unique,exact_matches,near_matches,missed,false_positives,accuracy"
            assert rows[1].startswith("违章02,京LNE560,3,1,0,京ADH9206,,")
            assert rows[2].startswith("违章03,京LNE560,3,1,0,,京A11111,")
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
