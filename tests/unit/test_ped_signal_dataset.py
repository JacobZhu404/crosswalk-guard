import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from redlight.data_pipeline.ped_signal_dataset import (
    light_state_to_label, assign_crop_labels, crop_box,
)


# ---- light_state_to_label: 灯态 -> 训练标签 ----
def test_state_to_label_mapping():
    assert light_state_to_label("green") == "walk"
    assert light_state_to_label("red") == "stand"
    assert light_state_to_label("flashing") == "walk"   # 清空相位, 行人仍在过街
    assert light_state_to_label("unknown") is None       # 未知不作训练标签
    assert light_state_to_label("") is None


# ---- assign_crop_labels: 用先验挑出信号候选, 其余=off ----
def _cand(cx, cy):
    return {"box": (int(cx * 100), int(cy * 100), int(cx * 100) + 10, int(cy * 100) + 10),
            "cx": cx, "cy": cy, "source": "hsv"}


def test_prior_labels_nearest_as_signal_rest_off():
    cands = [_cand(0.80, 0.15), _cand(0.30, 0.60)]  # 第1个靠近先验
    out = assign_crop_labels(cands, prior=(0.78, 0.15), seg_label="walk", radius=0.13)
    labels = {c["label"] for c in out}
    assert out[0]["label"] == "walk"   # 最近先验 -> 段标签
    assert out[1]["label"] == "off"    # 其余 -> off
    assert all(c["verified"] == 0 for c in out)   # 弱标签, 待人工校验


def test_no_prior_returns_unlabeled():
    cands = [_cand(0.80, 0.15), _cand(0.30, 0.60)]
    out = assign_crop_labels(cands, prior=None, seg_label="walk")
    assert all(c["label"] is None for c in out)   # 无先验 -> 无法定信号, 交人工


def test_nearest_beyond_radius_is_ambiguous():
    cands = [_cand(0.30, 0.60)]   # 离先验很远
    out = assign_crop_labels(cands, prior=(0.78, 0.15), seg_label="walk", radius=0.13)
    assert out[0]["label"] is None   # 信号本帧未出现在先验附近 -> 模糊, 不硬标


# ---- crop_box: 带 padding 裁剪 + 边界裁剪 ----
def test_crop_box_basic():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[20:40, 30:50] = 255
    sub = crop_box(frame, (30, 20, 50, 40), pad_ratio=0.0)
    assert sub.shape == (20, 20, 3)
    assert int(sub.mean()) == 255


def test_crop_box_clips_to_bounds():
    frame = np.zeros((100, 100, 3), np.uint8)
    sub = crop_box(frame, (95, 95, 130, 130), pad_ratio=0.0)  # 越界
    assert sub.shape[0] > 0 and sub.shape[1] > 0
    assert sub.shape[0] <= 5 and sub.shape[1] <= 5
