#!/usr/bin/env python3
"""Part B1 — 斑马线掩膜评测: 预测带 vs GT poly 的 mask-IoU + recall/precision。

主指标: CrosswalkDetector.detect(frame) **不带 vehicle_boxes**(本征掩膜质量, 与 cc 裁定一致)。
GT = datasets/gt/crosswalk/{video}.json 里的逐关键帧 polygon(Jacob 标; 像素坐标)。

v2 --temporal 模式(B1 方案): v2 走完整视频时序聚合(running-max), 镜像生产采样节奏
(interval=round(fps/cfg.inference.fps), 默认 ~4 帧取 1), 在 GT anchor 帧记录当前聚合 mask。

用法:
  python scripts/eval_crosswalk_mask.py                           # v11 全部已标视频
  python scripts/eval_crosswalk_mask.py --detector v2             # v2 单帧独立(对照)
  python scripts/eval_crosswalk_mask.py --detector v2 --temporal  # v2 时序聚合(生产口径)
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.crosswalk_v2 import CrosswalkDetectorV2
from redlight.evaluation.module_metrics import poly_to_mask, mask_iou, mask_band


def _make_detector(detector_name, cfg):
    return CrosswalkDetectorV2(cfg) if detector_name == "v2" else CrosswalkDetector(cfg)


def _mask_recall_precision(pred_mask, gt_mask):
    """recall = |pred∩GT|/|GT|, precision = |pred∩GT|/|pred|。"""
    p = pred_mask > 0
    g = gt_mask > 0
    inter = int(np.logical_and(p, g).sum())
    gt_area = int(g.sum())
    pred_area = int(p.sum())
    recall = inter / gt_area if gt_area > 0 else 0.0
    precision = inter / pred_area if pred_area > 0 else 0.0
    return recall, precision


def eval_video(video, gt_frames, cfg, preset, detector_name="v11", temporal=False):
    """评测单视频。

    temporal=False(默认): 每 GT 帧新建独立实例(量单帧掩膜质量)。
    temporal=True(仅 v2): 每视频一个实例, 顺序跑完整视频(按 cfg.inference.fps 采样),
                          在 GT anchor 帧记录当前聚合 mask(镜像生产节奏)。
    """
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if temporal and detector_name == "v2":
        # ---- 时序聚合模式(镜像生产节奏) ----
        det = _make_detector(detector_name, cfg)
        inf_fps = cfg.inference.fps
        interval = max(1, int(round(fps / inf_fps)))

        # 预计算 GT anchor 对应的采样帧索引
        gt_targets = {}  # frame_idx -> (gt_ts, poly)
        for fr in gt_frames:
            ts = fr["ts"]
            poly = fr.get("poly")
            if not poly or len(poly) < 3:
                print(f"    [跳过 {video}@{ts}] poly 未标注")
                continue
            target_fi = int(round(ts * fps / interval) * interval)
            gt_targets[target_fi] = (ts, poly)

        per_frame = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                mask = det.detect(frame)  # 更新 _accum
                if frame_idx in gt_targets:
                    gt_ts, poly = gt_targets[frame_idx]
                    H, W = mask.shape[:2]
                    gt_mask = poly_to_mask(poly, H, W)
                    iou = mask_iou(mask, gt_mask)
                    rec, prec = _mask_recall_precision(mask, gt_mask)
                    per_frame.append({
                        "ts": gt_ts, "pred_band": mask_band(mask), "gt_poly": poly,
                        "iou": round(iou, 3), "recall": round(rec, 3), "precision": round(prec, 3),
                    })
            frame_idx += 1
    else:
        # ---- 单帧独立模式(原口径, v11 默认) ----
        per_frame = []
        for fr in gt_frames:
            ts = fr["ts"]
            poly = fr.get("poly")
            if not poly or len(poly) < 3:
                print(f"    [跳过 {video}@{ts}] poly 未标注")
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, int(ts * 1000))
            ret, frame = cap.read()
            if not ret:
                print(f"    [跳过 {video}@{ts}] 取帧失败")
                continue
            det = _make_detector(detector_name, cfg)
            mask = det.detect(frame)
            H, W = mask.shape[:2]
            gt_mask = poly_to_mask(poly, H, W)
            iou = mask_iou(mask, gt_mask)
            rec, prec = _mask_recall_precision(mask, gt_mask)
            per_frame.append({
                "ts": ts, "pred_band": mask_band(mask), "gt_poly": poly,
                "iou": round(iou, 3), "recall": round(rec, 3), "precision": round(prec, 3),
            })
    cap.release()
    if not per_frame:
        return None
    mean_iou = sum(f["iou"] for f in per_frame) / len(per_frame)
    return {"video": video, "mean_iou": round(mean_iou, 3), "frames": per_frame}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--detector", default="v11", choices=["v11", "v2"],
                    help="v11=全宽水平带(基线) | v2=透视梯形(Plan v6 Phase 1 探针)")
    ap.add_argument("--temporal", action="store_true",
                    help="v2 时序聚合模式(每视频单实例, 镜像生产节奏; 仅 v2 有效)")
    ap.add_argument("--gt-crosswalk", default=os.path.join(ROOT, "datasets", "gt", "crosswalk"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    files = []
    if args.videos:
        files = [(v, os.path.join(args.gt_crosswalk, f"{v}.json")) for v in args.videos]
    else:
        if not os.path.isdir(args.gt_crosswalk):
            print("无 datasets/gt/crosswalk/ 目录, 先跑 gen_gt_skeleton.py 并由 Jacob 标")
            return
        for fn in sorted(os.listdir(args.gt_crosswalk)):
            if fn.endswith(".json"):
                files.append((fn[:-5], os.path.join(args.gt_crosswalk, fn)))

    mode = "时序聚合" if args.temporal else "单帧独立"
    print(f"=== B1 斑马线掩膜评测(preset={args.preset}, detector={args.detector}, 模式={mode}) ===\n")
    results = []
    for video, path in files:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            gt = json.load(f)
        r = eval_video(video, gt.get("frames", []), cfg, args.preset, args.detector, args.temporal)
        if r is None:
            continue
        results.append(r)
        mean_rec = sum(f.get("recall", 0) for f in r["frames"]) / len(r["frames"])
        mean_prec = sum(f.get("precision", 0) for f in r["frames"]) / len(r["frames"])
        print(f"[{video}] mask-IoU={r['mean_iou']:.3f} recall={mean_rec:.3f} precision={mean_prec:.3f} (n={len(r['frames'])})")
        for fr in r["frames"]:
            print(f"    @{fr['ts']:.1f}s  band={fr['pred_band']} IoU={fr['iou']:.3f} "
                  f"rec={fr.get('recall',0):.3f} prec={fr.get('precision',0):.3f}")
    if results:
        overall = sum(r["mean_iou"] for r in results) / len(results)
        all_recs = [f.get("recall", 0) for r in results for f in r["frames"]]
        all_precs = [f.get("precision", 0) for r in results for f in r["frames"]]
        print(f"\n=== 总体 mask-IoU={overall:.3f} recall={np.mean(all_recs):.3f} "
              f"precision={np.mean(all_precs):.3f} (视频数={len(results)}) ===")
        low = [r["video"] for r in results if r["mean_iou"] < 0.5]
        if low:
            print(f"⚠️ 掩膜失准视频(均值<0.5): {low}")


if __name__ == "__main__":
    main()
