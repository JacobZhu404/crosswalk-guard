#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3 ped-vs-vehicle 判别头 (Jacob 标样本训的 M2b 学习式判别层)。

输入: 候选灯的**紧框裁图**(tight crop of candidate box), 输出 ped/vehicle/other 三类概率。
用途: 在 M2b `select_gtfree` 的 L1(几何)+L2(时序) 之上, 用 P(ped) 作为可选评分翻盘
      "多固定设施几何平局"(ped 杆 vs 车信杆) 这类 L1/L2 结构性不可分的情况。

数据来源: scripts/build_l3_labeling.py 产出的 manifest.json(每簇含 video/fi/box_norm) +
          Jacob 导出的 labels.json(203 簇人工标)。auto_ped(auto, GT 强重合)/auto_other(auto, HSV 瞬时)
          作为训练负例, 不占 Jacob 人工。裁图在训练/打分时**实时从原帧 crop**(不依赖预存 crop 文件,
          auto_ped/auto_other 本就无预存裁图)。

护栏:
- 本模块不读任何逐帧 GT(与 select_gtfree 一致, 护栏1)。
- 评测须 leave-one-video-out: 视频 V 的 L3 模型只能在"其余视频标注裁图"上训,
  绝不在 V 自身上训(否则 gate 数字造假)。见 scripts/eval_selector_l3.py。

用法(评测流水线内部调用, 不在生产默认路径):
  from redlight.models.l3_ped_vehicle import (
      build_labeled_crops, group_by_video, L3PedVehicleNet,
      train_model, crop_candidate, score_crop)
"""
import os, sys, json, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]  # .../crosswalk-guard
CLASSES = ["ped", "vehicle", "other"]
_IDX = {c: i for i, c in enumerate(CLASSES)}
SIZE = 48

_TRANSFORM = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# 训练增强: 轻微抖动 + 随机水平翻转(灯杆左右基本对称)
_AUG = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

_FRAME_CACHE: Dict[Tuple[str, int], Image.Image] = {}


def _load_frame(video: str, fi: int) -> Image.Image:
    key = (video, fi)
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    p = ROOT / "datasets" / "frames" / video / f"frame_{fi:06d}.jpg"
    img = Image.open(str(p)).convert("RGB")
    _FRAME_CACHE[key] = img
    return img


def _crop_pil(frame: Image.Image, box_norm: Tuple[float, float, float, float],
              size: int = SIZE) -> Image.Image:
    """按归一化 box 从原帧裁出候选灯区域(PIL, 未 transform)。"""
    W, H = frame.size
    x1, y1, x2, y2 = box_norm
    px = (max(0, int(x1 * W)), max(0, int(y1 * H)),
          min(W, int(x2 * W)), min(H, int(y2 * H)))
    if px[2] <= px[0] or px[3] <= px[1]:
        return Image.new("RGB", (size, size), (0, 0, 0))  # 退化框 -> 黑图
    return frame.crop(px)


class L3PedVehicleNet(nn.Module):
    """tiny CNN: 3x48x48 -> 3 类。参数量 ~170k, CPU 可秒级推理。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_labeled_crops(manifest: dict, labels: Optional[dict] = None) -> List[dict]:
    """从 manifest 汇总标注裁图(每簇一条, 实时从原帧 crop)。
    labels: {cluster_id(int): 'ped'|'vehicle'|'other'} —— Jacob 人工标(仅 manual 簇需要)。
    auto_ped -> 'ped'(训练合法, GT 强重合); auto_other -> 'other'(HSV 瞬时)。
    返回 [ {video, fi, box_norm, label, source} , ... ]。fi = 该簇代表帧(rep_fi/fi)。"""
    out = []
    for c in manifest.get("manual", []):
        cid = c["id"]
        lab = labels.get(cid) if labels else None
        if lab is None:
            continue
        out.append({"video": c["video"], "fi": c.get("rep_fi", 0),
                    "box_norm": c.get("box_norm"), "label": lab, "source": "manual"})
    for c in manifest.get("auto_ped", []):
        out.append({"video": c["video"], "fi": c.get("rep_fi", 0),
                    "box_norm": c.get("box_norm"), "label": "ped", "source": "auto_ped"})
    for c in manifest.get("auto_other", []):
        out.append({"video": c["video"], "fi": c.get("fi", 0),
                    "box_norm": c.get("box_norm"), "label": "other", "source": "auto_other"})
    return out


def group_by_video(crops: List[dict]) -> Dict[str, List[dict]]:
    g: Dict[str, List[dict]] = {}
    for c in crops:
        g.setdefault(c["video"], []).append(c)
    return g


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """sqrt-dampened 逆频权重(软化 ped=81 / vehicle=15 / other=701 的极端不平衡)。"""
    cnt = {i: 0 for i in range(len(CLASSES))}
    for l in labels:
        cnt[l] += 1
    n = len(labels)
    k = len(CLASSES)
    w = []
    for i in range(k):
        c = cnt.get(i, 0) or 1
        w.append(1.0 / (c / n) ** 0.5)
    w = torch.tensor(w, dtype=torch.float32)
    return w / w.sum() * k  # 归一使均值=1, 不改变相对比例


class _CropDataset(Dataset):
    def __init__(self, items, aug=False):
        # items: list of (ref, label_idx)
        #   ref = PIL.Image | str(path) | tuple(video, fi, box_norm)
        self.items = items
        self.transform = _AUG if aug else _TRANSFORM

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ref, lab = self.items[i]
        if isinstance(ref, Image.Image):
            img = ref
        elif isinstance(ref, str):
            img = Image.open(ref).convert("RGB")
        else:  # tuple(video, fi, box_norm)
            video, fi, box = ref
            frame = _load_frame(video, fi)
            img = _crop_pil(frame, tuple(box))
        return self.transform(img), lab


def train_model(train_items: List[Tuple], val_items: List[Tuple],
                epochs: int = 40, lr: float = 1e-3, weight_decay: float = 1e-4,
                batch_size: int = 32, seed: int = 0, device: str = "cpu",
                verbose: bool = False) -> L3PedVehicleNet:
    """在 train_items=[(ref, label_idx)] 上训, val_items 监控早停(按 val acc)。
    ref: PIL / 路径 / (video,fi,box_norm)。返回 eval() 模式的模型。确定性(seed 锁)。"""
    random.seed(seed); torch.manual_seed(seed)
    train_labels = [lab for _, lab in train_items]
    weights = compute_class_weights(train_labels).to(device)
    crit = nn.CrossEntropyLoss(weight=weights)

    tr = _CropDataset(train_items, aug=True)
    va = _CropDataset(val_items, aug=False) if val_items else None
    tr_dl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    va_dl = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0) if va else None

    model = L3PedVehicleNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        if va_dl is not None:
            acc = _eval_acc(model, va_dl, device)
            if acc >= best_acc:
                best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  ep{ep:02d} val_acc={acc:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _eval_acc(model, dl, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total if total else 0.0


def score_crop(model: L3PedVehicleNet, img_tensor: torch.Tensor) -> Dict[str, float]:
    """对单张裁图张量(已 transform)输出三类概率 dict。"""
    model.eval()
    with torch.no_grad():
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        probs = torch.softmax(model(img_tensor), 1)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


def crop_candidate(frame_pil: Image.Image, box_norm: Tuple[float, float, float, float],
                   size: int = SIZE) -> torch.Tensor:
    """从原帧按归一化 box 裁出候选灯区域并 resize 成模型输入张量。"""
    return _TRANSFORM(_crop_pil(frame_pil, tuple(box_norm), size))


if __name__ == "__main__":
    m = L3PedVehicleNet()
    dummy = torch.rand(3, SIZE, SIZE)
    p = score_crop(m, dummy)
    assert set(p.keys()) == set(CLASSES)
    assert all(0.0 <= v <= 1.0 for v in p.values())
    assert abs(sum(p.values()) - 1.0) < 1e-5
    print("l3_ped_vehicle self-test OK:", p)
