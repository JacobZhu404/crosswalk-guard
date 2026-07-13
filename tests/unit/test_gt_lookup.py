"""GTLookup 单元测试。

覆盖: 车牌 GT 解析、灯态段级 GT 解析、人工标注 GT 加载、时间戳查询、evidence 语义展开。
全部用手工构造临时 CSV, 不依赖任何模型, 秒级跑完。
"""
import sys
import os
import tempfile
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.gt_lookup import (
    load_plate_gt,
    load_light_segments,
    load_light_state_csv,
    load_labeled_gt,
    state_at,
    expand_light_evidence,
)


class TestLoadPlateGT:
    def test_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("video,start_s,end_s,light_state,light_evidence,is_violation,violating_plates,other_plates,note\n")
            f.write("违章02,0,10,red,visible,1,京LNE560;无牌,京ADH9206,note\n")
            f.write("违章02,10,20,green,visible,0,,京LNE560,note2\n")
            f.write("违章03,0,30,red,visible,1,京ABV3428,,note3\n")
            path = f.name
        try:
            gt = load_plate_gt(path)
            assert "违章02" in gt
            assert "违章03" in gt
            # 去重 + 过滤 ?/无牌
            assert gt["违章02"] == ["京LNE560", "京ADH9206"]
            assert gt["违章03"] == ["京ABV3428"]
        finally:
            os.unlink(path)

    def test_filter_special_tokens(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("video,start_s,end_s,light_state,light_evidence,is_violation,violating_plates,other_plates,note\n")
            f.write("违章05,0,10,red,visible,1,京A12345;?;无牌,,note\n")
            path = f.name
        try:
            gt = load_plate_gt(path)
            assert gt["违章05"] == ["京A12345"]
        finally:
            os.unlink(path)

    def test_missing_file(self):
        gt = load_plate_gt("/nonexistent/events.csv")
        assert gt == {}

    def test_empty_plate_fields(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("video,start_s,end_s,light_state,light_evidence,is_violation,violating_plates,other_plates,note\n")
            f.write("违章01,0,10,red,visible,0,,,note\n")
            path = f.name
        try:
            gt = load_plate_gt(path)
            assert "违章01" not in gt
        finally:
            os.unlink(path)


class TestLoadLightSegments:
    def test_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("video,start_s,end_s,light_state,light_evidence,is_violation,violating_plates,other_plates,note\n")
            f.write("违章02,0,21,red,visible,0,京LNE560,,note\n")
            f.write("违章02,21,68,green,visible,1,京LNE560,,note2\n")
            f.write("违章03,0,30,red,inferred,1,,,note3\n")
            path = f.name
        try:
            gt = load_light_segments(path)
            assert "违章02" in gt
            assert "违章03" in gt
            segs = gt["违章02"]
            assert len(segs) == 2
            assert segs[0] == (0.0, 21.0, "red", "visible")
            assert segs[1] == (21.0, 68.0, "green", "visible")
            assert gt["违章03"][0] == (0.0, 30.0, "red", "inferred")
        finally:
            os.unlink(path)

    def test_unknown_evidence_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("video,start_s,end_s,light_state,light_evidence,is_violation,violating_plates,other_plates,note\n")
            f.write("违章02,0,10,unknown,,0,,,note\n")
            path = f.name
        try:
            gt = load_light_segments(path)
            assert gt["违章02"][0] == (0.0, 10.0, "unknown", "")
        finally:
            os.unlink(path)

    def test_sorting(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("video,start_s,end_s,light_state,light_evidence,is_violation,violating_plates,other_plates,note\n")
            f.write("违章02,21,68,green,visible,1,,,note2\n")
            f.write("违章02,0,21,red,visible,0,,,note\n")
            path = f.name
        try:
            gt = load_light_segments(path)
            segs = gt["违章02"]
            assert segs[0][0] == 0.0
            assert segs[1][0] == 21.0
        finally:
            os.unlink(path)


class TestLoadLightStateCSV:
    def test_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("video,start_s,end_s,state,confidence,note\n")
            f.write("违章02,0,20.9,red,confirmed,note\n")
            f.write("违章02,20.9,85,green,confirmed,note2\n")
            path = f.name
        try:
            gt = load_light_state_csv(path)
            segs = gt["违章02"]
            assert len(segs) == 2
            assert segs[0] == (0.0, 20.9, "red", "confirmed")
            assert segs[1] == (20.9, 85.0, "green", "confirmed")
        finally:
            os.unlink(path)


class TestLoadLabeledGT:
    def test_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("segment_id,start_ts,end_ts,predicted_state,predicted_reason,frame_count,mid_ts,thumb,gt_state,note\n")
            f.write("1,0.0,0.13,red,stable_lamp,2,0.07,thumb.jpg,red,\n")
            f.write("2,0.27,11.05,unknown,no_signal,81,5.66,thumb.jpg,,\n")
            f.write("3,11.18,16.3,red,stable_lamp,39,13.74,thumb.jpg,green,\n")
            path = f.name
        try:
            preds, gts = load_labeled_gt(path)
            assert preds == ["red", "red"]
            assert gts == ["red", "green"]
        finally:
            os.unlink(path)

    def test_missing_file(self):
        preds, gts = load_labeled_gt("/nonexistent/gt.csv")
        assert preds == []
        assert gts == []


class TestStateAt:
    def test_hit(self):
        segs = [(0.0, 21.0, "red", "visible"), (21.0, 68.0, "green", "visible")]
        assert state_at(segs, 10.0) == ("red", "visible")
        assert state_at(segs, 21.0) == ("red", "visible")  # 闭合区间: 先命中先返回
        assert state_at(segs, 30.0) == ("green", "visible")

    def test_miss_fallback(self):
        segs = [(0.0, 21.0, "red", "visible")]
        assert state_at(segs, 100.0) == ("red", "visible")  # 兜底返回最后一段

    def test_empty(self):
        assert state_at([], 10.0) == ("unknown", "confirmed")


class TestExpandLightEvidence:
    def test_visible(self):
        segs = [(0.0, 21.0, "red", "visible")]
        fn = expand_light_evidence(segs)
        assert fn(10.0) == ("red", "visible")

    def test_inferred(self):
        segs = [(0.0, 30.0, "green", "inferred")]
        fn = expand_light_evidence(segs)
        assert fn(10.0) == ("unknown", "inferred")

    def test_occluded(self):
        segs = [(0.0, 30.0, "red", "occluded")]
        fn = expand_light_evidence(segs)
        assert fn(10.0) == ("unknown", "occluded")

    def test_unknown_no_evidence(self):
        segs = [(0.0, 10.0, "unknown", "")]
        fn = expand_light_evidence(segs)
        assert fn(5.0) == ("unknown", "visible")

    def test_out_of_range(self):
        segs = [(0.0, 10.0, "red", "visible")]
        fn = expand_light_evidence(segs)
        assert fn(100.0) == ("unknown", "n/a")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
