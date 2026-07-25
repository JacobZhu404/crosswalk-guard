"""L3 ped-vs-vehicle 判别头单测 (TDD)。用合成数据, 不依赖真实视频/标注文件。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from PIL import Image
import torch
from redlight.models.l3_ped_vehicle import (
    build_labeled_crops, group_by_video, compute_class_weights,
    train_model, crop_candidate, score_crop, L3PedVehicleNet, CLASSES, SIZE, _IDX,
)


def _fake_manifest():
    # manual 3 簇(其中 1 未标), auto_ped 2, auto_other 4
    return {
        "n_manual": 3, "n_auto_ped": 2, "n_auto_other": 4,
        "manual": [
            {"id": 0, "video": "V1", "label": "", "crop_path": "x0", "frame_path": "f0", "box_norm": [0, 0, 0.1, 0.2]},
            {"id": 1, "video": "V1", "label": "", "crop_path": "x1", "frame_path": "f1", "box_norm": [0, 0, 0.1, 0.2]},
            {"id": 2, "video": "V2", "label": "", "crop_path": "x2", "frame_path": "f2", "box_norm": [0, 0, 0.1, 0.2]},
        ],
        "auto_ped": [
            {"video": "V1", "crop_path": "ap0", "frame_path": "f3", "box_norm": [0, 0, 0.1, 0.2]},
            {"video": "V2", "crop_path": "ap1", "frame_path": "f4", "box_norm": [0, 0, 0.1, 0.2]},
        ],
        "auto_other": [
            {"video": "V1", "crop_path": "ao0", "frame_path": "f5", "box_norm": [0, 0, 0.1, 0.2]},
            {"video": "V1", "crop_path": "ao1", "frame_path": "f6", "box_norm": [0, 0, 0.1, 0.2]},
            {"video": "V2", "crop_path": "ao2", "frame_path": "f7", "box_norm": [0, 0, 0.1, 0.2]},
            {"video": "V2", "crop_path": "ao3", "frame_path": "f8", "box_norm": [0, 0, 0.1, 0.2]},
        ],
    }


def test_build_labeled_crops_respects_labels_and_auto():
    m = _fake_manifest()
    labels = {0: "ped", 1: "vehicle"}  # id2 未标应跳过
    crops = build_labeled_crops(m, labels)
    # manual 已标 2 + auto_ped 2 + auto_other 4 = 8
    assert len(crops) == 8
    by_lab = {}
    for c in crops:
        by_lab.setdefault(c["label"], 0)
        by_lab[c["label"]] += 1
    # manual 已标 2(id0 ped, id1 vehicle) + auto_ped 2 + auto_other 4 = 8
    # -> ped=1 manual + 2 auto = 3, vehicle=1, other=4
    assert by_lab == {"ped": 3, "vehicle": 1, "other": 4}


def test_group_by_video():
    m = _fake_manifest()
    crops = build_labeled_crops(m, {0: "ped", 1: "vehicle"})
    g = group_by_video(crops)
    assert set(g.keys()) == {"V1", "V2"}
    # V1: manual ped0, manual veh1, auto_ped ap0, auto_other ao0, ao1 = 5
    assert len(g["V1"]) == 5
    # V2: auto_ped ap1, auto_other ao2, ao3 = 3 (id2 未标跳过)
    assert len(g["V2"]) == 3


def test_compute_class_weights_shape_and_finite():
    w = compute_class_weights([0, 0, 1, 2, 2, 2])
    assert w.shape == (3,)
    assert torch.isfinite(w).all()
    # 归一均值 ~1
    assert abs(w.mean().item() - 1.0) < 1e-5


def test_train_and_score_runs_on_synthetic():
    # 合成裁图: ped=纯红, vehicle=纯蓝, other=纯绿, 易分
    def solid(color):
        return Image.new("RGB", (SIZE, SIZE), color)
    items = []
    for _ in range(8):
        items.append((solid((220, 30, 30)), _IDX["ped"]))
        items.append((solid((30, 30, 220)), _IDX["vehicle"]))
        items.append((solid((30, 220, 30)), _IDX["other"]))
    train_items = items[:18]
    val_items = items[18:]
    model = train_model(train_items, val_items, epochs=15, verbose=False)
    assert isinstance(model, L3PedVehicleNet)
    p = score_crop(model, _TRANSFORM_dummy(solid((220, 30, 30))))
    assert set(p.keys()) == set(CLASSES)
    assert abs(sum(p.values()) - 1.0) < 1e-4


def _TRANSFORM_dummy(img):
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                               transforms.Normalize((0.5,) * 3, (0.5,) * 3)])(img)


def test_crop_candidate_degenerate_box_safe():
    frame = Image.new("RGB", (100, 100), (10, 10, 10))
    t = crop_candidate(frame, (0.5, 0.5, 0.5, 0.5))  # 零面积框
    # 退化框 -> 返回合法尺寸张量(黑图经 normalize 后非全零, 但不崩、形状对)
    assert t.shape == (3, SIZE, SIZE)
    assert torch.isfinite(t).all()


if __name__ == "__main__":
    test_build_labeled_crops_respects_labels_and_auto()
    test_group_by_video()
    test_compute_class_weights_shape_and_finite()
    test_train_and_score_runs_on_synthetic()
    test_crop_candidate_degenerate_box_safe()
    print("all l3_ped_vehicle unit tests passed")
