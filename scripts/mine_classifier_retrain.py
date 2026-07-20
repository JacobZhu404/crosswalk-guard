"""挖掘 light-state 判别器重训数据集 (Phase A, cc 238b2e3 批准)。

全新构建 datasets/classifier_retrain/, 不覆盖旧 datasets/ped_signal/(归档 legacy_238b)。
复用: gt_lookup.load_light_state_csv / FrameDataset / TrafficLightDetector.observe / ped_signal_dataset.crop_box。
注意: light_states.csv 第 4 字段是 confidence(非 evidence), 自写 _visible_state_at 解析, 不用 expand_light_evidence。

标签(复用 walk/stand/off):
  walk  = 真绿信号 (GT confirmed green 段 + prior ROI 抠出)
  stand = 真红信号 (GT confirmed red 段 + prior ROI 抠出)
  off   = impostor(非信号绿: 背心/植物/反射/信号外强绿斑) + 限量背景正则

挖掘源(半自动, 降 Jacob 标注成本):
  真信号:  FrameDataset 帧, GT visible green/red 段 -> prior ROI 抠图 (06/07 暗绿 + 04 短绿自然覆盖)
  impostor(a)+(b): 同帧 observe() obs=="green" 且 非(GT green&visible) -> prior ROI 抠图
  impostor(c): observe candidates 中距 prior > radius 的绿斑 -> 抠 bbox
  背景正则: prior 外随机区域, 限量(<=20% off 目标), 防"暗=off"偏见(07 暗绿被屠根因之一)

按视频 split(防过拟合红线): val={01,07,11} train={02,03,04,05,06,08,09,10} (无泄漏, 全 11 覆盖)。
最终接进 observe() 的模型须 train+val 合并全 11 重训(计划 §3.1), 本脚本只挖数据 + manifest。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/mine_classifier_retrain.py
"""
import os
import sys
import csv
import json
import argparse
import random

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.evaluation import gt_lookup
from redlight.evaluation.frame_dataset import FrameDataset
from redlight.models.traffic_light import TrafficLightDetector
from redlight.data_pipeline.ped_signal_dataset import crop_box, light_state_to_label
from redlight.infrastructure.image_utils import save_jpg

VIDEOS = [f"违章{i:02d}" for i in range(1, 12)]
NEG_VIDEOS = {"违章01", "违章10"}
VAL_VIDEOS = {"违章01", "违章07", "违章11"}
TRAIN_VIDEOS = {v for v in VIDEOS if v not in VAL_VIDEOS}
SPLIT = {v: ("val" if v in VAL_VIDEOS else "train") for v in VIDEOS}

IMPRIOR_RADIUS = 0.13      # 距 prior 最小归一化距离 -> 视为"信号外"绿斑
BG_OFF_RATIO = 0.20        # 背景正则 off 上限(占 off 目标)
WALK_MAX_MULT = 8          # walk 过采样上限(相对 base, 平衡红多绿少)
JITTER_PX = 12             # walk 过采样微抖动像素
BG_OFF_PROB = 0.05         # 每帧采背景正则概率

HEADER = ["crop_path", "video", "frame_ts", "x1", "y1", "x2", "y2", "source", "label", "verified"]


def _load_priors(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {v: (a[0], a[1], (a[2] if len(a) > 2 else 160)) for v, a in raw.items()
            if isinstance(a, (list, tuple)) and len(a) >= 2}


def _visible_state_at(segments, ts):
    """light_states.csv 解析: 段为 (start, end, state, confidence)。

    返回 (state, visible_bool):
      - visible = confidence == "confirmed" (确认可见真信号才抠图)
      - occluded / tentative 视为不可见(不抠真信号)
      - 未命中返回 ("unknown", False)
    不能用 gt_lookup.expand_light_evidence(它把 confidence 当 evidence 解释,
    导致 visible 永不命中, walk/stand 全为 0)。
    """
    for a, b, stt, conf in segments:
        if a <= ts <= b:
            return stt, (conf == "confirmed")
    return "unknown", False


def _trim_source(ann, src, keep_n):
    """保留 ann 中 source==src 的前 keep_n 个, 丢弃其余(确定性, 可复现)。"""
    out, kept = [], 0
    for a in ann:
        if a["source"] == src:
            if kept < keep_n:
                out.append(a)
                kept += 1
            # else 丢弃
        else:
            out.append(a)
    return out


def _prior_box(frame, px, py, roi_px, dx=0, dy=0):
    h, w = frame.shape[:2]
    cx, cy = int(px * w) + dx, int(py * h) + dy
    half = roi_px // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    return (x1, y1, x2, y2)


def main():
    ap = argparse.ArgumentParser(description="挖掘 light-state 判别器重训数据集")
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--light-states", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--priors", default=os.path.join(ROOT, "configs", "light_priors.json"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "classifier_retrain"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    cfg = load_config(args.config)
    det = TrafficLightDetector(cfg, verbose=False)
    segs = gt_lookup.load_light_state_csv(args.light_states)
    priors = _load_priors(args.priors)
    fd = FrameDataset(args.frames_dir)
    os.makedirs(args.out, exist_ok=True)

    counts = {v: {"walk": 0, "stand": 0, "off": 0, "impostor": 0,
                  "impostor_outside": 0, "prior_off": 0, "verified": 0} for v in VIDEOS}
    src_counts = {"neg_videos": sorted(NEG_VIDEOS), "false_green_scan": 0,
                  "outside_prior_green": 0, "prior_off": 0}
    rows = []

    for video in VIDEOS:
        segs_v = segs.get(video, [])
        prior = priors.get(video)
        if prior is None:
            print(f"[跳过] {video}: 无 prior")
            continue
        px, py, roi_px = prior
        det.set_video_prior(video)

        # ---- pass 1: 收集标注(不写盘) ----
        # light_states.csv 第 4 字段是 confidence(非 evidence), 不能用 expand_light_evidence
        # (它会把 confidence 当 evidence, 导致 visible 永不命中)。
        ann = []            # 真信号 + impostor(false-green) + 背景正则
        outside_ann = []    # impostor 第三源: 信号外绿斑(后截断, 防长视频爆炸)
        green_refs = []     # (fi, ts) 用于 walk 过采样
        for fi, ts, frame in fd.iter_video(video):
            if frame is None:
                continue
            state, visible = _visible_state_at(segs_v, ts)
            # 真信号 (GT confirmed green/red 且可见)
            if state in ("green", "red") and visible:
                label = light_state_to_label(state)   # green->walk, red->stand
                if label is None:
                    continue
                ann.append({"fi": fi, "ts": ts, "box": _prior_box(frame, px, py, roi_px),
                            "source": "prior_roi", "label": label})
                if label == "walk":
                    green_refs.append((fi, ts))
            # impostor(a)+(b): 引擎读绿 且 非(GT 真绿可见)
            res = det.observe(frame)
            if res.get("obs") == "green" and not (state == "green" and visible):
                ann.append({"fi": fi, "ts": ts, "box": _prior_box(frame, px, py, roi_px),
                            "source": "impostor", "label": "off"})
            # impostor(c): 信号外强绿斑 -> 暂存 outside_ann(后截断)
            # 守卫: 真绿可见帧不标 outside(否则把信号自身外的绿斑误当 impostor, 见 02 t=31.35)
            for s in res.get("candidates", []):
                if s.get("color") == "green" and not (state == "green" and visible):
                    dx, dy = s["cx"] - px, s["cy"] - py
                    if (dx * dx + dy * dy) ** 0.5 > IMPRIOR_RADIUS:
                        b = s["box"]
                        outside_ann.append({"fi": fi, "ts": ts,
                                            "box": (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                                            "source": "impostor_outside", "label": "off"})
            # 背景正则(限量, 后截断)
            if random.random() < BG_OFF_PROB:
                h, w = frame.shape[:2]
                rx, ry = random.random(), random.random()
                if ((rx - px) ** 2 + (ry - py) ** 2) ** 0.5 >= IMPRIOR_RADIUS:
                    bx, by = int(rx * w), int(ry * h)
                    half = roi_px // 2
                    ann.append({"fi": fi, "ts": ts,
                                "box": (max(0, bx - half), max(0, by - half),
                                        min(w, bx + half), min(h, by + half)),
                                "source": "prior_off", "label": "off"})

        # ---- impostor_outside 截断(防止 05/06 等长视频绿斑爆炸) ----
        imp_n = sum(1 for a in ann if a["source"] == "impostor")
        outside_cap = max(200, int(imp_n * 1.5))
        if len(outside_ann) > outside_cap:
            outside_ann = random.sample(outside_ann, outside_cap)
        ann.extend(outside_ann)

        # ---- 计数 ----
        c = {"walk": 0, "stand": 0, "off": 0, "impostor": 0, "impostor_outside": 0, "prior_off": 0}
        for a in ann:
            c[a["label"]] += 1
            if a["source"] == "impostor":
                c["impostor"] += 1
            elif a["source"] == "impostor_outside":
                c["impostor_outside"] += 1
            elif a["source"] == "prior_off":
                c["prior_off"] += 1

        # ---- off 预算截断: off <= 1.5*(walk+stand), 优先砍 outside 再 bg(保 impostor 价值) ----
        off_budget = max(10, int(1.5 * (c["walk"] + c["stand"])))
        excess = (c["impostor"] + c["impostor_outside"] + c["prior_off"]) - off_budget
        if excess > 0:
            cut = min(c["impostor_outside"], excess)
            if cut > 0:
                ann = _trim_source(ann, "impostor_outside", c["impostor_outside"] - cut)
                c["impostor_outside"] -= cut
                excess -= cut
            if excess > 0:
                cut = min(c["prior_off"], excess)
                if cut > 0:
                    ann = _trim_source(ann, "prior_off", c["prior_off"] - cut)
                    c["prior_off"] -= cut
                    excess -= cut
            if excess > 0:  # 极端才砍最珍贵的 false-green impostor
                cut = min(c["impostor"], excess)
                if cut > 0:
                    ann = _trim_source(ann, "impostor", c["impostor"] - cut)
                    c["impostor"] -= cut
                    excess -= cut

        # ---- walk 过采样: 补齐到 ~max(stand, off), 上限 WALK_MAX_MULT(红多/误绿多时补绿) ----
        walk_base = c["walk"]
        ref = max(c["stand"], c["impostor"] + c["impostor_outside"] + c["prior_off"])
        walk_target = min(ref, walk_base * WALK_MAX_MULT) if walk_base > 0 else 0
        extra = max(0, walk_target - walk_base)
        if extra > 0 and green_refs:
            for i in range(extra):
                fi, ts = green_refs[i % len(green_refs)]
                frame = fd.get_frame(video, fi)
                if frame is None:
                    continue
                dx = random.randint(-JITTER_PX, JITTER_PX)
                dy = random.randint(-JITTER_PX, JITTER_PX)
                ann.append({"fi": fi, "ts": ts, "box": _prior_box(frame, px, py, roi_px, dx, dy),
                            "source": "prior_roi", "label": "walk"})
                c["walk"] += 1

        # ---- pass 2: 写盘 ----
        vid_dir = os.path.join(args.out, video)
        os.makedirs(vid_dir, exist_ok=True)
        n_w = 0
        for a in ann:
            frame = fd.get_frame(video, a["fi"])
            if frame is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in a["box"]]
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            sub = crop_box(frame, (x1, y1, x2, y2))
            if sub is None or sub.size == 0:
                continue
            fname = f"{video}_t{a['ts']:.1f}_{n_w}_{a['label']}.jpg"
            fpath = os.path.join(vid_dir, fname)
            if not save_jpg(sub, fpath):
                continue
            rows.append({"crop_path": os.path.join(video, fname), "video": video,
                         "frame_ts": round(a["ts"], 2), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "source": a["source"], "label": a["label"], "verified": 0})
            counts[video][a["label"]] += 1
            if a["source"] == "impostor":
                counts[video]["impostor"] += 1
                src_counts["false_green_scan"] += 1
            elif a["source"] == "impostor_outside":
                counts[video]["impostor_outside"] += 1
                src_counts["outside_prior_green"] += 1
            elif a["source"] == "prior_off":
                counts[video]["prior_off"] += 1
                src_counts["prior_off"] += 1
            n_w += 1

        print(f"[{video}][{SPLIT[video]}] crops={n_w} walk={counts[video]['walk']} "
              f"stand={counts[video]['stand']} off={counts[video]['off']} "
              f"(imp={counts[video]['impostor']} out={counts[video]['impostor_outside']} bg={counts[video]['prior_off']})")

    # ---- 写 labels.csv + manifest.json ----
    labels_path = os.path.join(args.out, "labels.csv")
    with open(labels_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)

    total = {"walk": 0, "stand": 0, "off": 0, "impostor": 0, "impostor_outside": 0, "prior_off": 0}
    for v in VIDEOS:
        for k in total:
            total[k] += counts[v].get(k, 0)
    manifest = {
        "schema": "walk/stand/off",
        "split": {"train": sorted(TRAIN_VIDEOS), "val": sorted(VAL_VIDEOS)},
        "prior_roi_px": {v: int(priors[v][2]) for v in VIDEOS if v in priors},
        "counts": {v: counts[v] for v in VIDEOS},
        "total": total,
        "impostor_sources": src_counts,
        "note": "off=impostor(非信号绿)+限量背景正则; 最终接线模型须 train+val 合并全11重训(计划§3.1)",
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n完成: {len(rows)} crops -> {args.out}")
    print(f"  总计数: walk={total['walk']} stand={total['stand']} off={total['off']} "
          f"(impostor={total['impostor']} outside={total['impostor_outside']} bg={total['prior_off']})")
    print(f"  manifest -> {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
