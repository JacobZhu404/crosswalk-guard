#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_small_light_res.py — 小灯救援"代理"测试(cheap, 不下载 m/x 权重)

背景: cc 要求先用 yolov8m/x 大模型测 05/08 小灯(<2% 帧宽)能否被框住, 以定小灯路线。
但沙箱下载 ultralytics 资产 CDN 被 502 拦截, 无法取得 m/x 权重。

本脚本用 **yolov8n + 高推理分辨率(imgsz=1280)** 重跑 Diag3 v2 公平测作为代理:
小灯漏检的主因是"有效分辨率不足"(灯在 640 输入下仅 ~10-13px 宽), 提高推理分辨率
是 yolov8m/x 对小灯增益的主要来源之一, 故 imgsz 提升可代理"更大模型/更高分辨率"路线
的可行性。若 imgsz=1280 救回 05/08, 则小灯路线=提高有效分辨率(imgsz 提升 或 m/x);
若仍失败, 则需模板匹配/从邻近命中帧跟踪。

指标与 Diag3 v2 完全一致(中心距 + IoU vs 逐帧 GT, 去循环), 仅改推理分辨率。
纯只读: 不加载 pipeline, 不重挖, 不写 configs。复用 diag_yolo_vs_gt 的比对常量/函数。
"""
import os, sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
from diag_yolo_vs_gt import (load_boxed_annotations, box_center, iou,
                             CENTER_HIT, IOU30, IOU50, WEAK)


def run_yolo_boxes(model, frame_path, conf, imgsz):
    img = cv2.imread(str(frame_path))
    if img is None:
        return [], 0, 0
    H, W = img.shape[:2]
    res = model(img, conf=conf, classes=[9], imgsz=imgsz, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        xyxy = b.xyxy[0].tolist()
        boxes.append({"box": [xyxy[0] / W, xyxy[1] / H, xyxy[2] / W, xyxy[3] / H],
                      "conf": float(b.conf[0])})
    return boxes, W, H


def render_overlay(frame_path, gt_box, yolo_boxes, out_png, video, fi, imgsz):
    img = cv2.imread(str(frame_path))
    if img is None:
        return False
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v * (W if i % 2 == 0 else H)) for i, v in enumerate(gt_box)]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(img, "GT ped", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for yb in yolo_boxes:
        bx = yb["box"]
        xx1, yy1, xx2, yy2 = [int(v * (W if i % 2 == 0 else H)) for i, v in enumerate(bx)]
        cv2.rectangle(img, (xx1, yy1), (xx2, yy2), (0, 0, 255), 2)
        cv2.putText(img, f"{yb['conf']:.2f}", (xx1, max(yy1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f"{video} fi={fi} imgsz{imgsz}", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_png), img)
    return True


def main():
    ap = argparse.ArgumentParser(description="小灯救援代理测试: yolov8n @ imgsz=1280 (vs Diag3 v2 n@640)")
    ap.add_argument("--gt", default=str(ROOT / "datasets" / "light_location_gt.json"))
    ap.add_argument("--frames-dir", default=str(ROOT / "datasets" / "frames"))
    ap.add_argument("--yolo-weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out-json", default=str(ROOT / "data" / "output" / "diag_small_light_imgsz1280.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "output" / "diag_small_light_imgsz1280"))
    args = ap.parse_args()

    annots, schema = load_boxed_annotations(args.gt)
    print(f"[GT] {len(annots)} boxed frames; imgsz={args.imgsz} conf={args.conf}")
    from ultralytics import YOLO
    model = YOLO(args.yolo_weights)
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for video, fi, gt_box, color, no_light in annots:
        fpath = frames_dir / video / f"frame_{fi:06d}.jpg"
        if not fpath.is_file():
            print(f"  skip missing {fpath.name}")
            continue
        yolo_boxes, W, H = run_yolo_boxes(model, fpath, args.conf, args.imgsz)
        gcx, gcy = box_center(gt_box)
        if yolo_boxes:
            dists = [((box_center(yb["box"])[0] - gcx) ** 2 +
                      (box_center(yb["box"])[1] - gcy) ** 2) ** 0.5 for yb in yolo_boxes]
            best_iou = max(iou(yb["box"], gt_box) for yb in yolo_boxes)
            nearest = min(dists)
            near_idx = dists.index(nearest)
        else:
            best_iou = 0.0
            nearest = 1.0
            near_idx = -1
        rec = {
            "video": video, "fi": fi, "color": color, "no_light": no_light,
            "n_yolo": len(yolo_boxes), "best_iou": best_iou,
            "nearest_center_dist": nearest, "center_hit": nearest < CENTER_HIT,
            "iou30": best_iou >= IOU30, "iou50": best_iou >= IOU50,
            "nearest_box_conf": yolo_boxes[near_idx]["conf"] if near_idx >= 0 else None,
            "gt_wh": [gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]],
        }
        results.append(rec)
        render_overlay(fpath, gt_box, yolo_boxes, out_dir / f"{video}_fi{fi}.png", video, fi, args.imgsz)

    if not results:
        print("no results")
        return
    n = len(results)
    c_hit = sum(1 for r in results if r["center_hit"])
    i30 = sum(1 for r in results if r["iou30"])
    i50 = sum(1 for r in results if r["iou50"])
    mean_iou = sum(r["best_iou"] for r in results) / n
    mean_dist = sum(r["nearest_center_dist"] for r in results) / n
    print(f"\n=== imgsz={args.imgsz} 汇总 (conf={args.conf}, {n} 帧) ===")
    print(f"  中心命中: {c_hit}/{n} = {c_hit/n*100:.1f}%   (v2 n@640 基线=64.3%)")
    print(f"  IoU>=0.3: {i30}/{n} = {i30/n*100:.1f}%   (v2 基线=57.1%)")
    print(f"  IoU>=0.5: {i50}/{n} = {i50/n*100:.1f}%   (v2 基线=35.7%)")
    print(f"  mean IoU={mean_iou:.3f}  mean ctrD={mean_dist:.3f}")

    print(f"\n=== 弱视频单列 (03/05/08) imgsz={args.imgsz} ===")
    for v in sorted(WEAK):
        rs = [r for r in results if r["video"] == v]
        if not rs:
            continue
        ch = sum(1 for r in rs if r["center_hit"])
        i3 = sum(1 for r in rs if r["iou30"])
        print(f"  {v}: {len(rs)}帧 中心命中 {ch}/{len(rs)}  IoU>=0.3 {i3}/{len(rs)}")
        for r in rs:
            print(f"    fi={r['fi']} nYOLO={r['n_yolo']} IoU={r['best_iou']:.2f} "
                  f"ctrD={r['nearest_center_dist']:.3f} cHit={'Y' if r['center_hit'] else '.'} "
                  f"i50={'Y' if r['iou50'] else '.'} gtWH={r['gt_wh'][0]:.3f}x{r['gt_wh'][1]:.3f} {r['color']}")

    out = {
        "schema": schema, "conf": args.conf, "imgsz": args.imgsz, "n_frames": n,
        "baseline_v2_n640": {"center_hit": 0.643, "iou30": 0.571, "iou50": 0.357},
        "summary": {"center_hit": c_hit / n, "iou30": i30 / n, "iou50": i50 / n,
                     "mean_best_iou": mean_iou, "mean_nearest_center_dist": mean_dist},
        "per_frame": results,
        "weak_videos": {v: [r for r in results if r["video"] == v] for v in WEAK},
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[out] {args.out_json}")


if __name__ == "__main__":
    main()
