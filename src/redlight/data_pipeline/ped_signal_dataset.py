"""L2 数据管线: 行人信号灯 crop 训练集构建 (M1 Phase2 spec §5)。

从帧(预抽帧或视频)在**可见灯态段**抽候选 ROI -> 弱标签(walk/stand/off) -> 落 crop 数据集,
供 M1 状态分类器训练。弱标签用 GT 段灯态 + 已知先验位置自举, 未确定者交灯态画廊人工校验。

复用: build_candidates(候选并集) / GTLookup.state_at(段灯态) / image_utils.save_jpg。
纯逻辑函数(light_state_to_label / assign_crop_labels / crop_box)不依赖 cv2, 便于单测。
"""
import os
import csv


# ---- 灯态 -> 训练标签 (纯函数) ----
def light_state_to_label(state):
    """行人灯态 -> 分类标签。green=walk, red=stand, flashing=walk(清空相位仍在过街);
    unknown/空 -> None(不作训练标签)。"""
    if state == "green" or state == "flashing":
        return "walk"
    if state == "red":
        return "stand"
    return None


# ---- 候选打标 (纯函数): 用先验挑信号, 其余 off ----
def assign_crop_labels(candidates, prior, seg_label, radius=0.13):
    """给一帧的候选 ROI 弱标签。

    - prior 给定: 离先验最近且在 radius 内的候选 = 信号 -> seg_label; 其余 -> 'off'。
      最近候选超出 radius(信号本帧未出现在先验附近) -> 全部 None(模糊, 不硬标)。
    - prior=None: 无法确定哪个是信号 -> 全部 None(交人工在画廊标)。
    所有自动标签 verified=0(弱标签, 待人工校验); 人工确认后置 1。
    """
    out = []
    if prior is None:
        for c in candidates:
            out.append({**c, "label": None, "verified": 0})
        return out
    px, py = prior
    dists = [((c["cx"] - px) ** 2 + (c["cy"] - py) ** 2) ** 0.5 for c in candidates]
    nearest_i = min(range(len(candidates)), key=lambda i: dists[i]) if candidates else None
    ambiguous = nearest_i is None or dists[nearest_i] > radius
    for i, c in enumerate(candidates):
        if ambiguous:
            label = None
        elif i == nearest_i:
            label = seg_label
        else:
            label = "off"
        out.append({**c, "label": label, "verified": 0})
    return out


# ---- 裁剪 (纯函数, 仅 numpy 切片) ----
def crop_box(frame, box, pad_ratio=0.0):
    """按 box=(x1,y1,x2,y2) 裁剪, 可选 pad_ratio 外扩, 越界自动裁到帧内。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    if pad_ratio > 0:
        pw = int((x2 - x1) * pad_ratio)
        ph = int((y2 - y1) * pad_ratio)
        x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return frame[0:0, 0:0]
    return frame[y1:y2, x1:x2]


LABELS_HEADER = ["crop_path", "video", "frame_ts", "x1", "y1", "x2", "y2",
                 "source", "label", "verified"]


def extract_crops(frame_iter, video, state_at_fn, candidates_fn, out_dir,
                  prior=None, pad_ratio=0.15, min_crop_px=8):
    """遍历一个视频的帧, 抽信号 crop 并弱标签, 写盘 + 返回 labels 行。

    frame_iter: 可迭代 (frame_idx, ts, frame_bgr)。
    state_at_fn(ts) -> (state, evidence): 通常 = functools.partial(gt_lookup.state_at, segments)。
    candidates_fn(frame) -> [{box,cx,cy,source}]: 通常包装 build_candidates + HSV/YOLO。
    prior: (px,py) 归一化, 或 None。
    返回: list of dict(LABELS_HEADER); 同时把 crop 写到 out_dir/<video>/。
    """
    from ..infrastructure.image_utils import save_jpg
    vid_dir = os.path.join(out_dir, video)
    os.makedirs(vid_dir, exist_ok=True)
    rows = []
    for frame_idx, ts, frame in frame_iter:
        if frame is None:
            continue
        state, evidence = state_at_fn(ts)
        # 只在"可见"段自动标注(inferred/occluded 段灯不在画面, 期望 unknown, 不作训练正样本)
        if evidence not in ("visible", "", None) and evidence is not False:
            if evidence in ("inferred", "occluded"):
                continue
        seg_label = light_state_to_label(state)
        if seg_label is None:
            continue
        cands = candidates_fn(frame)
        labeled = assign_crop_labels(cands, prior, seg_label)
        for j, c in enumerate(labeled):
            if c["label"] is None:
                continue
            x1, y1, x2, y2 = c["box"]
            if (x2 - x1) < min_crop_px or (y2 - y1) < min_crop_px:
                continue
            sub = crop_box(frame, c["box"], pad_ratio)
            if sub.size == 0:
                continue
            fname = "%s_t%.1f_%d_%s.jpg" % (video, ts, j, c["label"])
            fpath = os.path.join(vid_dir, fname)
            if not save_jpg(sub, fpath):
                continue
            rows.append({
                "crop_path": fpath, "video": video, "frame_ts": round(ts, 2),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "source": c.get("source", ""), "label": c["label"], "verified": 0,
            })
    return rows


def write_labels_csv(rows, path):
    """写/追加 labels.csv (crop 数据集索引)。"""
    exists = os.path.isfile(path)
    with open(path, "a" if exists else "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABELS_HEADER)
        if not exists:
            w.writeheader()
        w.writerows(rows)
