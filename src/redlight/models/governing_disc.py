#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""governing_disc.py — §3.2 governing 判别器(有效行人灯 vs 干扰)。

R1 目标: 学「这是不是一盏真·处于信号态的行人灯」(拒反射/信号灯背面/车灯/绿树叶),
         **不是**「governing vs 非 governing」(governing 性是方向属性, 64×64 crop 看不见)。
         governing 选择(多盏有效灯里挑管这条道的)仍交 L1 几何 + L2 时序, 不压给判别器。
负样本: 主力=无灯帧候选(负B, 保证干扰); 负A(同帧非gov候选)仅消融(带A vs 不带A 各跑LOVO)。
R3: τ 用训练折内/inner-CV 定单一全局值, 不碰测试折; 报告 τ 敏感性曲线。
R4: τ 选取含 漏绿 ≤ 80 硬约束(由调用方在选灯评测里施加, 见 eval_selection_quality.py)。

本模块只定义模型 + 数据 + 单折训练, LOVO 编排在 scripts/train_governing_discriminator.py。不接线生产(红线)。
"""
import math, random, copy
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
from PIL import Image

from redlight.models.ped_light_selector import iou
from redlight.models.signal_candidates import build_candidates
from redlight.models.traffic_light import TrafficLightDetector

ROOT = Path(__file__).resolve().parents[3]  # 仓库根(crosswalk-guard), 含 input_video/ 与 models/
CROP = 64  # 判别器输入分辨率(RGB)


def _cfg_tl():
    import types
    tl = types.SimpleNamespace(method="color", smoothing_window=8, sat_min=130, value_floor=60,
                               min_area_px=30, max_area_ratio=0.008, max_aspect_ratio=3.5, color_s_min=22)
    return types.SimpleNamespace(traffic_light=tl)


class GoverningDiscNet(nn.Module):
    """tiny-CNN, 64×64 RGB -> 1。约 36KB 量级, 不堆容量(PhaseB 教训)。

    A3: 输出经温度缩放 sigmoid(logit/T), T 在 val 校准集(来自训练视频, 不碰测试折 R3)
    grid 选最小 BCE 拟合, 使 τ 跨折可比。训练时 T=1.0(裸 logit 训练稳定), 训后 fit_temperature 定 T。
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(48 * (CROP // 8) * (CROP // 8), 64)
        self.fc2 = nn.Linear(64, 1)
        self.temperature = 1.0

    def forward_logits(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    def forward(self, x):
        return torch.sigmoid(self.forward_logits(x) / self.temperature)


_TRANSFORM = None


def _get_transform():
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = lambda im: torch.from_numpy(
            np.asarray(im.resize((CROP, CROP)), dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return _TRANSFORM


def crop_candidate(frame: np.ndarray, box_norm: Tuple[float, float, float, float]) -> Optional[Image.Image]:
    """从帧按归一化框裁 crop, 返回 PIL RGB(与 L3 同口径, 保证训练/评测一致)。"""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box_norm
    px = (max(0, int(x1 * W)), max(0, int(y1 * H)), min(W, int(x2 * W)), min(H, int(y2 * H)))
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return Image.fromarray(cv2.cvtColor(frame[px[1]:px[3], px[0]:px[2]], cv2.COLOR_BGR2RGB))


_YOLO = None


def _lazy_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    return _YOLO


def _read_frames_at(video: str, fis: List[int]) -> Dict[int, np.ndarray]:
    """顺序读视频, 在指定 fi 取帧(与 canonical 测量同口径, 去循环)。返回 {fi: frame}。"""
    cap = cv2.VideoCapture(str(ROOT / "input_video" / f"{video}.mp4"))
    want = set(fis)
    out = {}
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in want:
            out[fi] = frame.copy()
        fi += 1
    cap.release()
    return out


def build_crop_dataset(gt: Dict, use_negative_a: bool = False) -> Tuple[List, List, List]:
    """从 canonical GT 产训练 crops。
    正: 717 governing 框 crop(有效行人灯)。
    负B(主力): 无灯帧的所有候选 crop(保证干扰)。
    负A(仅消融): governing 帧中 IoU<0.3 的非gov候选 crop(可能含平行斑马线真灯→噪声)。
    返回 (pos_items, neg_b_items, neg_a_items), 每项 (img_tensor, label)。"""
    tf = _get_transform()
    yolo = _lazy_yolo()
    det = TrafficLightDetector(_cfg_tl(), verbose=False)
    by_video: Dict[str, List[Dict]] = {}
    for fr in gt["frames"]:
        by_video.setdefault(fr["video"], []).append(fr)

    pos, neg_b, neg_a = [], [], []
    for video, frames in by_video.items():
        fis = [int(f["source_fi"]) for f in frames]
        fr_map = {int(f["source_fi"]): f for f in frames}
        frs = _read_frames_at(video, fis)
        for fi, frame in frs.items():
            g = fr_map[fi]
            H, W = frame.shape[:2]
            for b in g.get("boxes", []):
                if b.get("governing"):
                    crop = crop_candidate(frame, tuple(b["box_norm"]))
                    if crop is not None:
                        pos.append((tf(crop), 1.0))
            no_light = g.get("no_light", False)
            gov_boxes = [tuple(b["box_norm"]) for b in g.get("boxes", []) if b.get("governing")]
            if no_light or (use_negative_a and gov_boxes):
                res = yolo(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
                yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
                hsv_px = [s["box"] for s in det._candidates(frame)]
                cands = build_candidates(yolo_px, hsv_px, W, H)
                for c in cands:
                    crop = crop_candidate(frame, tuple(c["box"]))
                    if crop is None:
                        continue
                    item = (tf(crop), 0.0)
                    if no_light:
                        neg_b.append(item)
                    elif use_negative_a and gov_boxes:
                        bi = tuple(c["box"])
                        if all(iou(bi, gb) < 0.3 for gb in gov_boxes):
                            neg_a.append(item)
    return pos, neg_b, neg_a


class _CropDS(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        img, label = self.items[i]
        return img, torch.tensor(label, dtype=torch.float32)


def fit_temperature(model: GoverningDiscNet, val_items: List,
                    grid: Tuple[float, ...] = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)) -> None:
    """A3 温度缩放: T 在 val 校准集(来自训练视频, 不碰测试折 R3)上 grid 选最小 BCE。
    让 τ 跨折可比。val_items 为空则 T=1.0。"""
    if not val_items:
        model.temperature = 1.0
        return
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model.forward_logits(it[0].unsqueeze(0)) for it in val_items])
        y = torch.cat([torch.tensor(it[1], dtype=torch.float32).unsqueeze(0) for it in val_items])
    best_T, best_loss = 1.0, float("inf")
    for T in grid:
        p = torch.sigmoid(logits / T).squeeze(-1)
        loss = float(F.binary_cross_entropy(p, y).item())
        if loss < best_loss:
            best_loss, best_T = loss, T
    model.temperature = best_T


def train_model(pos_items, neg_items, seed: int = 0, epochs: int = 60,
                wd: float = 1e-4, batch: int = 32, patience: int = 10) -> GoverningDiscNet:
    """在给定正负样本上训练单折模型。LOVO 折外由 CLI 脚本切分。

    A3: 切 15% val 做早停(保最优 ckpt, 防过拟合); 训后 fit_temperature 校准(让 τ 跨折可比)。
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = pos_items + neg_items
    random.shuffle(data)
    n_val = max(1, int(0.15 * len(data)))
    val, tr = data[:n_val], data[n_val:]
    tr_ds, val_ds = _CropDS(tr), _CropDS(val)
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False)
    model = GoverningDiscNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    crit = nn.BCELoss()
    best_loss, best_state, waits = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = crit(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                p = model(xb).squeeze(1)
                vl += crit(p, yb).item() * len(yb)
                n += len(yb)
        vl /= max(1, n)
        if vl < best_loss:
            best_loss, best_state, waits = vl, copy.deepcopy(model.state_dict()), 0
        else:
            waits += 1
            if waits >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    fit_temperature(model, val)  # A3 校准(val 来自训练视频, 不碰测试折 R3)
    return model


def score_crop(model: GoverningDiscNet, img_tensor: torch.Tensor) -> float:
    """返回 P(有效行人灯); 经 fit_temperature 校准的 T 缩放(使 τ 跨折可比)。"""
    model.eval()
    with torch.no_grad():
        return float(model(img_tensor.unsqueeze(0)).item())
