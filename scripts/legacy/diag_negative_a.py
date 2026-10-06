#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_negative_a.py — ① A 消融的**前置诊断**(不跑全量 LOVO, 不接线)。

背景: governing_disc.build_crop_dataset 的 docstring 明确警告 —
      负A(同帧 IoU<0.3 的非gov候选)「可能含平行斑马线真灯 → 噪声」。
      若 neg_a 混入真·亮着的行人灯, 等于教模型拒真绿 → 漏绿恶化, 该消融就不该硬跑。

本脚本三件事(对应派活 5a/5b/5c):
  a) 逐 LOVO 折统计 len(neg_a), 与 pos/neg_b 量级对比(neg_b 曾 1→964, 看 neg_a 是否同样非空且合理);
  b) 抽 ~20 个 neg_a crop 存 PNG(放大版 + 带红框上下文版)供人眼判「干扰 vs 真灯」, 估污染率;
  c) 单折 10pos/10neg(含 neg_a)过拟合 sanity: 逐 epoch BCE + pos/neg 平均 logit, 确认可分
     (复刻 1ef9c45 的排查手法)。

用法(必须 caffeinate 防 macOS 挂起):
  cd /Users/jacob/personal/crosswalk-guard-wb && caffeinate -i -s env PYTHONPATH=src \
    /Users/jacob/personal/crosswalk-guard/.venv/bin/python scripts/diag_negative_a.py --stage build
  ... --stage sample    # 抽样存 PNG(读 build 产的缓存)
  ... --stage sanity    # 10pos/10neg 过拟合自检
"""
import argparse, json, os, random, re, sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from redlight.models import governing_disc as gd
from redlight.models.ped_light_selector import iou
from redlight.models.signal_candidates import build_candidates
from redlight.models.traffic_light import TrafficLightDetector

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
CACHE = ROOT / "models" / "governing_disc" / "diag_negA_cache.pt"   # *.pt 已 gitignore
SAMPLE_DIR = ROOT / "diag_negA_samples"
MANIFEST = SAMPLE_DIR / "manifest.json"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _vcode(video: str) -> str:
    """违章07 -> 07(文件名走 ASCII, 避免 cv2.imwrite 非 ASCII 路径坑)。"""
    m = re.search(r"(\d+)", video)
    return m.group(1) if m else video


# ---------------------------------------------------------------- stage: build
def stage_build():
    """逐视频重放 build_crop_dataset 的 neg_a 抽取路径(同口径), 额外留住原始像素框+帧号,
    以便 b) 抽样存图。计数结果按视频存, LOVO 折计数 = 总数 - 该折留出视频的计数。"""
    gt = json.load(open(GT, encoding="utf-8"))
    tf = gd._get_transform()
    yolo = gd._lazy_yolo()
    det = TrafficLightDetector(gd._cfg_tl(), verbose=False)

    by_video = {}
    for fr in gt["frames"]:
        by_video.setdefault(fr["video"], []).append(fr)

    store = {}  # video -> {"pos": [...], "neg_b": [...], "neg_a": [(tensor, meta)]}
    for video in sorted(by_video):
        frames = by_video[video]
        fis = [int(f["source_fi"]) for f in frames]
        fr_map = {int(f["source_fi"]): f for f in frames}
        frs = gd._read_frames_at(video, fis)
        pos, neg_b, neg_a = [], [], []
        for fi in sorted(frs):
            frame = frs[fi]
            g = fr_map[fi]
            H, W = frame.shape[:2]
            for b in g.get("boxes", []):
                if b.get("governing"):
                    if gd.crop_candidate(frame, tuple(b["box_norm"])) is not None:
                        pos.append(fi)
            no_light = g.get("no_light", False)
            gov_boxes = [tuple(b["box_norm"]) for b in g.get("boxes", []) if b.get("governing")]
            if not (no_light or gov_boxes):
                continue
            res = yolo(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
            yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
            hsv_px = [s["box"] for s in det._candidates(frame)]
            cands = build_candidates(yolo_px, hsv_px, W, H)
            for c in cands:
                bx0, by0, bx1, by1 = c["box"]
                box_norm = (bx0 / W, by0 / H, bx1 / W, by1 / H)
                crop = gd.crop_candidate(frame, box_norm)
                if crop is None:
                    continue
                if no_light:
                    neg_b.append(fi)
                elif gov_boxes and all(iou(box_norm, gb) < 0.3 for gb in gov_boxes):
                    max_i = max(iou(box_norm, gb) for gb in gov_boxes)
                    neg_a.append((tf(crop), {"video": video, "fi": int(fi),
                                             "box_norm": [float(v) for v in box_norm],
                                             "source": c.get("source"), "max_iou_gov": float(max_i),
                                             "wh_px": [int(bx1 - bx0), int(by1 - by0)]}))
        store[video] = {"n_pos": len(pos), "n_neg_b": len(neg_b), "neg_a": neg_a}
        print(f"[{video}] pos={len(pos)} neg_b={len(neg_b)} neg_a={len(neg_a)}", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, CACHE)

    videos = sorted(store)
    tot_p = sum(store[v]["n_pos"] for v in videos)
    tot_b = sum(store[v]["n_neg_b"] for v in videos)
    tot_a = sum(len(store[v]["neg_a"]) for v in videos)
    print("\n=== 5a) 逐 LOVO 折(留出该视频)训练集样本量 ===")
    print(f"{'留出折':10} {'pos':>5} {'neg_b':>6} {'neg_a':>6} {'neg合计':>7} {'neg_a占neg':>10}")
    for V in videos:
        p = tot_p - store[V]["n_pos"]
        b = tot_b - store[V]["n_neg_b"]
        a = tot_a - len(store[V]["neg_a"])
        print(f"{V:10} {p:>5} {b:>6} {a:>6} {b+a:>7} {a/(b+a)*100:>9.1f}%")
    print(f"{'全量(参考)':10} {tot_p:>5} {tot_b:>6} {tot_a:>6} {tot_b+tot_a:>7} "
          f"{tot_a/(tot_b+tot_a)*100:>9.1f}%")


# --------------------------------------------------------------- stage: sample
def stage_sample(n=20, seed=0):
    """分层抽样 ~n 个 neg_a crop 存 PNG: 放大版(看清亮不亮/什么灯) + 带红框上下文版(判是否平行斑马线真灯)。"""
    store = torch.load(CACHE, weights_only=False)
    videos = [v for v in sorted(store) if store[v]["neg_a"]]
    rng = random.Random(seed)
    # 按视频 neg_a 占比分层, 每视频至少 1 个
    tot = sum(len(store[v]["neg_a"]) for v in videos)
    quota = {v: max(1, round(n * len(store[v]["neg_a"]) / tot)) for v in videos}
    picks = []
    for v in videos:
        idxs = rng.sample(range(len(store[v]["neg_a"])), min(quota[v], len(store[v]["neg_a"])))
        picks += [(v, i) for i in idxs]
    rng.shuffle(picks)
    picks = picks[:n]
    picks.sort()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    # 按视频批量读帧(避免重复解码)
    need = {}
    for v, i in picks:
        need.setdefault(v, []).append(store[v]["neg_a"][i][1]["fi"])
    frames_by_v = {v: gd._read_frames_at(v, fis) for v, fis in need.items()}

    manifest = []
    for k, (v, i) in enumerate(picks, 1):
        _, meta = store[v]["neg_a"][i]
        frame = frames_by_v[v][meta["fi"]]
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = meta["box_norm"]
        px = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
        base = f"negA_{k:02d}_v{_vcode(v)}_fi{meta['fi']}"
        # ① crop 放大(最近邻, 保留原始像素质感, 长边放到 ~192)
        crop = frame[px[1]:px[3], px[0]:px[2]]
        s = max(1, int(192 / max(1, max(crop.shape[:2]))))
        big = cv2.resize(crop, (crop.shape[1] * s, crop.shape[0] * s), interpolation=cv2.INTER_NEAREST)
        p_crop = SAMPLE_DIR / f"{base}_crop.png"
        Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)).save(p_crop)
        # ② 上下文(框外扩 6 倍, 画红框) — 判「平行斑马线的真行人灯」必须看环境
        cw, ch = px[2] - px[0], px[3] - px[1]
        mx, my = max(60, cw * 3), max(60, ch * 3)
        cx0, cy0 = max(0, px[0] - mx), max(0, px[1] - my)
        cx1, cy1 = min(W, px[2] + mx), min(H, px[3] + my)
        ctx = frame[cy0:cy1, cx0:cx1].copy()
        cv2.rectangle(ctx, (px[0] - cx0, px[1] - cy0), (px[2] - cx0, px[3] - cy0), (0, 0, 255), 2)
        sc = max(1, int(480 / max(1, max(ctx.shape[:2]))))
        if sc > 1:
            ctx = cv2.resize(ctx, (ctx.shape[1] * sc, ctx.shape[0] * sc), interpolation=cv2.INTER_NEAREST)
        p_ctx = SAMPLE_DIR / f"{base}_ctx.png"
        Image.fromarray(cv2.cvtColor(ctx, cv2.COLOR_BGR2RGB)).save(p_ctx)
        rec = dict(meta, idx=k, crop_png=str(p_crop), ctx_png=str(p_ctx))
        manifest.append(rec)
        print(f"[{k:02d}] {v} fi={meta['fi']} wh={meta['wh_px']} src={meta['source']} "
              f"maxIoU_gov={meta['max_iou_gov']:.2f}\n     {p_crop}\n     {p_ctx}", flush=True)
    _atomic_write(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[out] {MANIFEST}  (n={len(manifest)})")


# --------------------------------------------------------------- stage: sanity
def stage_sanity(holdout="违章01", seed=0, epochs=60):
    """5c) 单折 10pos/10neg(含 neg_a)过拟合自检: 能过拟合 = 输入可分、标签没塌; 逐 epoch 打 BCE + 均值 logit。"""
    store = torch.load(CACHE, weights_only=False)
    gt = json.load(open(GT, encoding="utf-8"))
    sub = {"frames": [f for f in gt["frames"] if f["video"] != holdout]}
    # pos: 直接从 GT governing 框裁(无需 YOLO, 便宜)
    tf = gd._get_transform()
    by_video = {}
    for fr in sub["frames"]:
        by_video.setdefault(fr["video"], []).append(fr)
    rng = random.Random(seed)
    pos = []
    for v in sorted(by_video):
        cand = [f for f in by_video[v] if any(b.get("governing") for b in f.get("boxes", []))]
        if not cand:
            continue
        f = rng.choice(cand)
        frs = gd._read_frames_at(v, [int(f["source_fi"])])
        frame = frs.get(int(f["source_fi"]))
        if frame is None:
            continue
        for b in f.get("boxes", []):
            if b.get("governing"):
                c = gd.crop_candidate(frame, tuple(b["box_norm"]))
                if c is not None:
                    pos.append((tf(c), 1.0))
                    break
        if len(pos) >= 10:
            break
    pool_a = [(t, m) for v in sorted(store) if v != holdout for (t, m) in store[v]["neg_a"]]
    neg = [(t, 0.0) for t, _ in rng.sample(pool_a, min(10, len(pool_a)))]
    print(f"[sanity] holdout={holdout} pos={len(pos)} neg_a={len(neg)} (10/10 过拟合自检)")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = gd.GoverningDiscNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    X = torch.stack([t for t, _ in pos + neg])
    y = torch.tensor([lab for _, lab in pos + neg], dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model.forward_logits(X).squeeze(1)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                lg = model.forward_logits(X).squeeze(1)
            pm, nm = float(lg[y == 1].mean()), float(lg[y == 0].mean())
            acc = float(((lg > 0).float() == y).float().mean())
            print(f"  ep{ep:>3} BCE={float(loss):.4f} pos_logit={pm:+.3f} neg_logit={nm:+.3f} "
                  f"margin={pm-nm:+.3f} acc={acc*100:.0f}%", flush=True)
    model.eval()
    with torch.no_grad():
        lg = model.forward_logits(X).squeeze(1)
    pm, nm = float(lg[y == 1].mean()), float(lg[y == 0].mean())
    ok = pm > 0 > nm
    print(f"[sanity] 结论: pos均值logit={pm:+.3f} neg均值logit={nm:+.3f} → "
          f"{'PASS ✅ 可分(标签未塌, neg_a 非恒正)' if ok else 'FAIL ❌ 不可分'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "sample", "sanity"], required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="违章01")
    a = ap.parse_args()
    {"build": lambda: stage_build(),
     "sample": lambda: stage_sample(a.n, a.seed),
     "sanity": lambda: stage_sanity(a.holdout, a.seed)}[a.stage]()


if __name__ == "__main__":
    main()
