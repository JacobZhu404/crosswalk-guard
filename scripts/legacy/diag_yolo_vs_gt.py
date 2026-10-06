#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_yolo_vs_gt.py — 只读诊断(修正版): YOLO 能否当"逐帧检测器"定位行人信号灯?

⚠️ v2 指标修正(v1 被 cc 指出受相机运动污染):
  v1 把每帧 YOLO 框去比**同一个静态 canonical 框**(全视频一个平均位置)。
  但相机手持会动、灯逐帧位移 → 多数帧静态参照本就错位, YOLO 即便框住(移动后的)
  灯也判 miss → 命中率被压到 0–4%(运动假象)。
  v2 改为: 对 GT 里每个**有 true_box_norm 的标注帧 fi**, 加载该帧 frame_{fi:06d}.jpg,
  把 YOLO 框与**该帧 true_box_norm** 比。这才是公平测试。

双指标:
  - 中心距 center-dist: YOLO 最近框中心到 GT 中心的距离; <0.06 记"中心命中"(更能判"找没找到",
    因 IoU 被 YOLO 框(整灯头)与 GT 框(紧框)尺寸差压低)。
  - IoU: 帧内 best-IoU; 报 ≥0.3 与 ≥0.5 两档。

人眼确认: 每帧渲染叠框 PNG(GT 蓝框 + 全部 YOLO 红框)到 data/output/diag3_overlays/,
供直接看"近中心 YOLO 框是行人灯还是恰好在旁的车灯"(最关键的未知)。

弱视频单列: 03/05/08(cc 点名)单独报, 区分"YOLO 真漏" vs "灯太小/被遮"。

⚠️ 纯只读: 不加载/不修改任何 pipeline 代码, 不重挖, 不写 configs。
复现:
  PYTHONPATH=src ./.venv/bin/python scripts/diag_yolo_vs_gt.py
"""
import os, sys, json, glob, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

CENTER_HIT = 0.06   # 中心命中阈值(与 cc 公平测试一致)
IOU30 = 0.3
IOU50 = 0.5
WEAK = {"违章03", "违章05", "违章08"}


def load_boxed_annotations(path):
    """返回 [(video, fi, true_box_norm, color, no_light)] 仅 true_box_norm 非 null。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for a in d.get("annotations", []):
        tb = a.get("true_box_norm")
        if tb is None:
            continue
        out.append((a["video"], a["fi"], tb, a.get("color"), a.get("no_light", False)))
    return out, d.get("schema", "?")


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def run_yolo_boxes(model, frame_path, conf):
    img = cv2.imread(str(frame_path))
    if img is None:
        return [], 0, 0
    H, W = img.shape[:2]
    res = model(img, conf=conf, classes=[9], verbose=False)[0]
    boxes = []
    for b in res.boxes:
        xyxy = b.xyxy[0].tolist()
        boxes.append({
            "box": [xyxy[0] / W, xyxy[1] / H, xyxy[2] / W, xyxy[3] / H],
            "conf": float(b.conf[0]),
        })
    return boxes, W, H


def render_overlay(frame_path, gt_box, yolo_boxes, out_png, video, fi):
    img = cv2.imread(str(frame_path))
    if img is None:
        return False
    H, W = img.shape[:2]
    # GT 蓝框
    x1, y1, x2, y2 = [int(v * (W if i % 2 == 0 else H)) for i, v in enumerate(gt_box)]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(img, "GT ped", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)
    # YOLO 红框
    for yb in yolo_boxes:
        bx = yb["box"]
        xx1, yy1, xx2, yy2 = [int(v * (W if i % 2 == 0 else H)) for i, v in enumerate(bx)]
        cv2.rectangle(img, (xx1, yy1), (xx2, yy2), (0, 0, 255), 2)
        cv2.putText(img, f"{yb['conf']:.2f}", (xx1, max(yy1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f"{video} fi={fi}", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_png), img)
    return True


def main():
    ap = argparse.ArgumentParser(description="YOLO vs 行人灯硬GT 命中率验证 (修正版, 只读)")
    ap.add_argument("--gt", default=str(ROOT / "datasets" / "light_location_gt.json"))
    ap.add_argument("--frames-dir", default=str(ROOT / "datasets" / "frames"))
    ap.add_argument("--yolo-weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--conf", type=float, default=0.05, help="YOLO 置信阈(低阈)")
    ap.add_argument("--out-json", default=str(ROOT / "data" / "output" / "diag_yolo_vs_gt.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "output" / "diag3_overlays"))
    ap.add_argument("--no-overlay", action="store_true", help="不渲染叠框 PNG")
    args = ap.parse_args()

    annots, schema = load_boxed_annotations(args.gt)
    print(f"[GT] schema={schema}, 有框标注帧数={len(annots)} (公平测试集)")

    try:
        from ultralytics import YOLO
        model = YOLO(args.yolo_weights)
        print(f"[YOLO] 已加载 {args.yolo_weights}")
    except Exception as e:
        print(f"[YOLO] 加载失败: {e}", file=sys.stderr)
        sys.exit(1)

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for video, fi, gt_box, color, no_light in annots:
        fpath = frames_dir / video / f"frame_{fi:06d}.jpg"
        if not fpath.is_file():
            print(f"  ⚠️ {video} fi={fi}: 帧文件缺失 {fpath.name}, 跳过")
            continue
        yolo_boxes, W, H = run_yolo_boxes(model, fpath, args.conf)
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
            "n_yolo": len(yolo_boxes),
            "best_iou": best_iou,
            "nearest_center_dist": nearest,
            "center_hit": nearest < CENTER_HIT,
            "iou30": best_iou >= IOU30,
            "iou50": best_iou >= IOU50,
            "nearest_box_conf": yolo_boxes[near_idx]["conf"] if near_idx >= 0 else None,
            "gt_wh": [gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]],
        }
        results.append(rec)
        if not args.no_overlay:
            render_overlay(fpath, gt_box, yolo_boxes, out_dir / f"{video}_fi{fi}.png", video, fi)

    if not results:
        print("无结果")
        return

    # 汇总
    n = len(results)
    c_hit = sum(1 for r in results if r["center_hit"])
    i30 = sum(1 for r in results if r["iou30"])
    i50 = sum(1 for r in results if r["iou50"])
    mean_iou = sum(r["best_iou"] for r in results) / n
    mean_dist = sum(r["nearest_center_dist"] for r in results) / n
    print(f"\n=== 公平测试汇总 (conf={args.conf}, {n} 标注帧) ===")
    print(f"  中心命中(<{CENTER_HIT}): {c_hit}/{n} = {c_hit/n*100:.1f}%")
    print(f"  IoU≥{IOU30}: {i30}/{n} = {i30/n*100:.1f}%")
    print(f"  IoU≥{IOU50}: {i50}/{n} = {i50/n*100:.1f}%")
    print(f"  mean best-IoU={mean_iou:.3f}  mean 最近中心距={mean_dist:.3f}")

    # 逐视频表
    print(f"\n=== 逐视频 (conf={args.conf}) ===")
    print(f"{'video':7} {'fi':>6} {'nYOLO':>5} {'IoU':>5} {'ctrD':>5} {'cHit':>4} {'i30':>3} {'i50':>3} {'gtWH':>9} {'col':>5}")
    for r in results:
        print(f"{r['video']:7} {r['fi']:>6} {r['n_yolo']:>5} {r['best_iou']:>5.2f} "
              f"{r['nearest_center_dist']:>5.3f} {'Y' if r['center_hit'] else '.':>4} "
              f"{'Y' if r['iou30'] else '.':>3} {'Y' if r['iou50'] else '.':>3} "
              f"{r['gt_wh'][0]:.3f}x{r['gt_wh'][1]:.3f} {r['color']:>5}")

    # 弱视频单列
    print(f"\n=== 弱视频单列 (cc 点名 03/05/08) ===")
    for v in sorted(WEAK):
        rs = [r for r in results if r["video"] == v]
        if not rs:
            continue
        ch = sum(1 for r in rs if r["center_hit"])
        i3 = sum(1 for r in rs if r["iou30"])
        print(f"  {v}: {len(rs)} 帧, 中心命中 {ch}/{len(rs)}, IoU≥0.3 {i3}/{len(rs)}; "
              f"gt_wh={[round(r['gt_wh'][0],3) for r in rs]}/{[round(r['gt_wh'][1],3) for r in rs]} "
              f"(灯宽/高, 越小越难)")
        for r in rs:
            print(f"    fi={r['fi']} nYOLO={r['n_yolo']} IoU={r['best_iou']:.2f} "
                  f"ctrD={r['nearest_center_dist']:.3f} cHit={'Y' if r['center_hit'] else '.'} "
                  f"gtWH={r['gt_wh'][0]:.3f}x{r['gt_wh'][1]:.3f} {r['color']}")

    # JSON
    out = {
        "schema": schema, "conf": args.conf, "n_frames": n,
        "center_hit_thresh": CENTER_HIT, "iou30": IOU30, "iou50": IOU50,
        "summary": {
            "center_hit": c_hit / n, "iou30": i30 / n, "iou50": i50 / n,
            "mean_best_iou": mean_iou, "mean_nearest_center_dist": mean_dist,
        },
        "per_frame": results,
        "weak_videos": {v: [r for r in results if r["video"] == v] for v in WEAK},
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[out] {args.out_json}")
    print(f"[out] overlays: {out_dir}/")


if __name__ == "__main__":
    main()
