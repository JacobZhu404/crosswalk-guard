#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_candidates.py — 为 28 标注帧收集逐帧候选(YOLO cls=9 + HSV 亮斑并集), 归一化存储。

供 PedLightSelector + leave-some-out 评测离线使用(评测时不需再跑 YOLO, 快且确定)。
纯数据准备, 不改 pipeline。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/collect_candidates.py [--imgsz 1280]
  PYTHONPATH=src ./.venv/bin/python scripts/collect_candidates.py --window 12 --step 4
    (沿每个 boxed 标注帧采 ±window 帧窗口, 供 L2 跨帧复现评分, 输出 candidates_temporal.json)
输出: data/output/candidates_28.json (单帧) 或 candidates_temporal.json (窗口)
  records: [{video, fi, anchor, candidates:[{box_norm,source,cx,cy,area}], gt_box_norm, gt_wh}]
"""
import json, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import types
from redlight.models.traffic_light import TrafficLightDetector
from redlight.models.signal_candidates import build_candidates


def _cfg():
    tl = types.SimpleNamespace(method="color", smoothing_window=8, sat_min=130,
                               value_floor=60, min_area_px=30, max_area_ratio=0.008,
                               max_aspect_ratio=3.5, color_s_min=22)
    return types.SimpleNamespace(traffic_light=tl)


def load_boxed_annotations(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for a in d.get("annotations", []):
        tb = a.get("true_box_norm")
        if tb is None:
            continue
        out.append((a["video"], a["fi"], tb, (tb[2] - tb[0], tb[3] - tb[1])))
    return out


def _collect_frame(img, det, model, args, W, H):
    res = model(img, conf=args.conf, classes=[9], imgsz=args.imgsz, verbose=False)[0]
    yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
    hsv_px = [s["box"] for s in det._candidates(img)]
    cands = build_candidates(yolo_px, hsv_px, W, H)
    rec = []
    for c in cands:
        bx = c["box"]
        rec.append({
            "box_norm": (bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H),
            "source": c["source"], "cx": c["cx"], "cy": c["cy"], "area": c["area"],
        })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(ROOT / "datasets" / "light_location_gt.json"))
    ap.add_argument("--frames-dir", default=str(ROOT / "datasets" / "frames"))
    ap.add_argument("--yolo-weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--window", type=int, default=0,
                    help="沿 boxed 标注帧采 ±window 帧窗口(供 L2); 0=单帧(默认)")
    ap.add_argument("--step", type=int, default=4, help="窗口采样步长")
    ap.add_argument("--out", default=str(ROOT / "data" / "output" / "candidates_28.json"))
    args = ap.parse_args()

    # 窗口模式默认输出候选时序文件, 避免覆盖单帧基线
    if args.window > 0 and args.out == str(ROOT / "data" / "output" / "candidates_28.json"):
        args.out = str(ROOT / "data" / "output" / "candidates_temporal.json")

    annots = load_boxed_annotations(args.gt)
    det = TrafficLightDetector(_cfg(), verbose=False)
    from ultralytics import YOLO
    model = YOLO(args.yolo_weights)
    frames_dir = Path(args.frames_dir)
    records = []
    for video, fi, gt_box, gt_wh in annots:
        if args.window > 0:
            fis = list(range(fi - args.window, fi + args.window + 1, args.step))
        else:
            fis = [fi]
        n_collected = 0
        for f in fis:
            fpath = frames_dir / video / f"frame_{f:06d}.jpg"
            if not fpath.is_file():
                continue
            img = cv2.imread(str(fpath))
            H, W = img.shape[:2]
            rec_cands = _collect_frame(img, det, model, args, W, H)
            records.append({"video": video, "fi": f, "anchor": fi,
                            "candidates": rec_cands,
                            "gt_box_norm": tuple(gt_box), "gt_wh": (gt_wh[0], gt_wh[1])})
            n_collected += 1
        print(f"  {video} anchor={fi}: {n_collected} 窗口帧, gt_wh={gt_wh[0]:.3f}x{gt_wh[1]:.3f}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n": len(records), "imgsz": args.imgsz,
                   "window": args.window, "step": args.step, "records": records},
                  f, ensure_ascii=False, indent=2)
    print(f"[out] {args.out}  ({len(records)} frames)")


if __name__ == "__main__":
    main()
