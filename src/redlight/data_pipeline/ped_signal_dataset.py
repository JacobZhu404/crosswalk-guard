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
                  prior=None, pad_ratio=0.15, min_crop_px=8,
                  prior_roi_mode=False, prior_roi_px=160, off_margin=0.25):
    """遍历一个视频的帧, 抽信号 crop 并弱标签, 写盘 + 返回 labels 行。

    frame_iter: 可迭代 (frame_idx, ts, frame_bgr)。
    state_at_fn(ts) -> (state, evidence): 通常 = functools.partial(gt_lookup.state_at, segments)。
    candidates_fn(frame) -> [{box,cx,cy,source}]: 通常包装 build_candidates + HSV/YOLO。
    prior: (px,py) 归一化, 或 None。
    prior_roi_mode: True 时走"先验 ROI 直抽"路径(推荐有 prior 的视频):
      每帧抽 2 个等尺寸 crop —
        (a) prior ROI 中心一块(信号灯位置) -> 标 walk/green段|stand/red段
        (b) prior 外随机位置一块(背景)    -> 标 off
      正负 ~1:1, 不再用全图 HSV 候选(避免一帧10+噪声 off 稀释训练集)。
      prior=None 时此模式回退到旧路径(全图候选)。
    prior_roi_px: prior_roi_mode 下 ROI 边长(px, 与 light_priors.json 第3项一致)。
    off_margin: prior 外随机 crop 中心距 prior 的最小归一化距离(确保是"非信号区域")。
    返回: list of dict(LABELS_HEADER); 同时把 crop 写到 out_dir/<video>/。

    crop_path 存**相对 out_dir 的路径**(如 `违章02/xxx.jpg`), 跨机可移植;
    load_labeled_crops 读取时按 labels.csv 所在目录解析回绝对路径。
    """
    from ..infrastructure.image_utils import save_jpg
    import random as _random
    vid_dir = os.path.join(out_dir, video)
    os.makedirs(vid_dir, exist_ok=True)
    rows = []
    use_prior_roi = prior_roi_mode and prior is not None

    def _save_crop(sub, box, label, source, ts, idx):
        x1, y1, x2, y2 = box
        if (x2 - x1) < min_crop_px or (y2 - y1) < min_crop_px:
            return None
        sub = crop_box(frame, box, pad_ratio)
        if sub.size == 0:
            return None
        fname = "%s_t%.1f_%d_%s.jpg" % (video, ts, idx, label)
        fpath = os.path.join(vid_dir, fname)
        if not save_jpg(sub, fpath):
            return None
        return {
            "crop_path": os.path.join(video, fname), "video": video, "frame_ts": round(ts, 2),
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "source": source, "label": label, "verified": 0,
        }

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

        if use_prior_roi:
            # (a) prior ROI 中心 crop = 信号灯位置(标 walk/stand)
            h, w = frame.shape[:2]
            px, py = prior
            cx_i, cy_i = int(px * w), int(py * h)
            half = prior_roi_px // 2
            box = (cx_i - half, cy_i - half, cx_i + half, cy_i + half)
            r = _save_crop(None, box, seg_label, "prior_roi", ts, 0)
            if r:
                rows.append(r)
            # (b) prior 外随机 crop = 背景(标 off); 中心距 prior >= off_margin
            for _try in range(8):
                rx = _random.random()
                ry = _random.random()
                if ((rx - px) ** 2 + (ry - py) ** 2) ** 0.5 >= off_margin:
                    break
            bx_i, by_i = int(rx * w), int(ry * h)
            off_box = (bx_i - half, by_i - half, bx_i + half, by_i + half)
            r = _save_crop(None, off_box, "off", "prior_off", ts, 1)
            if r:
                rows.append(r)
            continue

        # 旧路径(无 prior 或未开 prior_roi_mode): 全图 HSV 候选, prior 选最近=信号其余=off
        cands = candidates_fn(frame)
        labeled = assign_crop_labels(cands, prior, seg_label)
        for j, c in enumerate(labeled):
            if c["label"] is None:
                continue
            r = _save_crop(None, c["box"], c["label"], c.get("source", ""), ts, j)
            if r:
                rows.append(r)
    return rows


def lovo_folds(rows, verified_only=False):
    """留一视频交叉验证分折 (M1 spec §6): 每个视频轮流做测试集, 其余做训练集。

    rows: list of dict, 至少含 'video','label'(可含 'verified')。
    verified_only=True: 只用 label 非空且 verified==1 的行(人工校验过的)。
    返回: list of (test_video, train_rows, test_rows)。
    """
    def _ok(r):
        if r.get("label") in (None, ""):
            return False
        if verified_only and int(r.get("verified", 0)) != 1:
            return False
        return True
    usable = [r for r in rows if _ok(r)]
    videos = sorted({r["video"] for r in usable})
    folds = []
    for v in videos:
        test = [r for r in usable if r["video"] == v]
        train = [r for r in usable if r["video"] != v]
        folds.append((v, train, test))
    return folds


def load_labeled_crops(csv_path, verified_only=False):
    """读 labels.csv 为 rows(dict list); verified_only 时过滤未校验/无标签。

    crop_path 若为相对路径(如 `违章02/xxx.jpg`), 按 labels.csv 所在目录解析回绝对路径,
    保证跨机(Mac/Windows)可移植; 若已是绝对路径则原样保留。
    """
    rows = []
    if not os.path.isfile(csv_path):
        return rows
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if verified_only and (r.get("label") in (None, "") or int(r.get("verified", 0) or 0) != 1):
                continue
            p = r.get("crop_path", "")
            if p and not os.path.isabs(p):
                r["crop_path"] = os.path.join(base_dir, p)
            rows.append(r)
    return rows


def write_labels_csv(rows, path):
    """写/追加 labels.csv (crop 数据集索引)。"""
    exists = os.path.isfile(path)
    with open(path, "a" if exists else "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABELS_HEADER)
        if not exists:
            w.writeheader()
        w.writerows(rows)
