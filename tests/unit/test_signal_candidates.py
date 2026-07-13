import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.models.signal_candidates import build_candidates


def test_union_keeps_disjoint_boxes():
    yolo = [(10, 10, 30, 30)]           # (x1,y1,x2,y2) 像素
    hsv = [(100, 100, 120, 120)]
    out = build_candidates(yolo, hsv, frame_w=200, frame_h=200, iou_thr=0.5)
    assert len(out) == 2
    assert {c["source"] for c in out} == {"yolo", "hsv"}


def test_overlapping_boxes_deduped_prefer_larger():
    yolo = [(10, 10, 50, 50)]           # area 1600 (大)
    hsv = [(12, 12, 30, 30)]            # area 324, 与上高度重叠
    out = build_candidates(yolo, hsv, frame_w=200, frame_h=200, iou_thr=0.3)
    assert len(out) == 1
    assert out[0]["source"] == "yolo"   # 保留面积大者
    assert out[0]["box"] == (10, 10, 50, 50)


def test_normalized_center_reported():
    out = build_candidates([(0, 0, 100, 100)], [], frame_w=200, frame_h=200)
    assert out[0]["cx"] == 0.25 and out[0]["cy"] == 0.25
