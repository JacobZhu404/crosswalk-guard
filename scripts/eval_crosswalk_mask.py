#!/usr/bin/env python3
"""Part B1 — 斑马线掩膜评测: 预测带 vs GT y-band 的 band-IoU。

主指标: CrosswalkDetector.detect(frame) **不带 vehicle_boxes**(本征掩膜质量, 与 cc 裁定一致)。
预测带 = 掩膜非零行区间 [y0_pred, y1_pred](v11 掩膜全宽, 竖直范围即带)。
GT = datasets/gt/crosswalk/{video}.json 里的逐关键帧 y0/y1(Jacob 标; 每帧独立, 兼容相机漂移)。

用法:
  python scripts/eval_crosswalk_mask.py                       # 跑全部已标视频
  python scripts/eval_crosswalk_mask.py --videos 违章11        # 单视频
"""
import os
import sys
import json
import argparse

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.crosswalk_v2 import CrosswalkDetectorV2
from redlight.evaluation.module_metrics import poly_to_mask, mask_iou, mask_band


def _make_detector(detector_name, cfg):
    return CrosswalkDetectorV2(cfg) if detector_name == "v2" else CrosswalkDetector(cfg)


def eval_video(video, gt_frames, cfg, preset, detector_name="v11"):
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
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
        # v2 有状态(时间聚合), 但 B1 关键帧彼此远离 -> 每帧独立实例, 量单帧掩膜质量
        det = _make_detector(detector_name, cfg)
        mask = det.detect(frame)  # 不带 vehicle_boxes(本征掩膜质量)
        H, W = mask.shape[:2]
        gt_mask = poly_to_mask(poly, H, W)
        iou = mask_iou(mask, gt_mask)
        per_frame.append({"ts": ts, "pred_band": mask_band(mask), "gt_poly": poly, "iou": round(iou, 3)})
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

    print(f"=== B1 斑马线掩膜评测(preset={args.preset}, 主指标=无车框带, detector={args.detector}) ===\n")
    results = []
    for video, path in files:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            gt = json.load(f)
        r = eval_video(video, gt.get("frames", []), cfg, args.preset, args.detector)
        if r is None:
            continue
        results.append(r)
        print(f"[{video}] 平均 mask-IoU={r['mean_iou']:.3f} (n={len(r['frames'])})")
        for fr in r["frames"]:
            print(f"    @{fr['ts']:.1f}s  pred_band={fr['pred_band']} gt_poly={fr['gt_poly']} maskIoU={fr['iou']:.3f}")
    if results:
        overall = sum(r["mean_iou"] for r in results) / len(results)
        print(f"\n=== 总体平均 mask-IoU={overall:.3f} (视频数={len(results)}) ===")
        # 暴露失准: 任何视频均值 < 0.5 即为掩膜失准重点
        low = [r["video"] for r in results if r["mean_iou"] < 0.5]
        if low:
            print(f"⚠️ 掩膜失准视频(均值<0.5): {low}")


if __name__ == "__main__":
    main()
