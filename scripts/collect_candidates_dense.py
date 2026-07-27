#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_candidates_dense.py — 从源视频密集抽逐帧候选(YOLO cls=9 + HSV 亮斑并集), 归一化存储。

**纯 GT-free**: 不读 light_location_gt.json, 不写 gt_box_norm(那属 eval 用途, 测量严禁 GT 播种)。
供量"误绿"测量脚本离线使用(测量时不需再跑 YOLO, 快且确定)。

与 collect_candidates.py 的区别:
  - 输入从源视频 input_video/*.mp4 逐帧抽(而非 datasets/frames 稀疏图);
  - 记 source_fi(源帧号)用于 t=source_fi/fps 映射 events.csv;
  - 不写任何 GT 字段。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/collect_candidates_dense.py --step 4
  PYTHONPATH=src ./.venv/bin/python scripts/collect_candidates_dense.py --step 4 --limit 20   # 冒烟(每视频前20采样帧)
输出: data/output/candidates_dense.json
  {fps, step, n_videos, n_frames, videos: {video: [{fi, candidates:[{box_norm,source,cx,cy,area}]}, ...]}}
"""
import json, sys, argparse, time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "input_video"))
    ap.add_argument("--yolo-weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--step", type=int, default=4, help="源视频采样步长(帧)")
    ap.add_argument("--limit", type=int, default=0, help="每视频最多抽 N 个采样帧(0=全部)")
    ap.add_argument("--out", default=str(ROOT / "data" / "output" / "candidates_dense.json"))
    args = ap.parse_args()

    det = TrafficLightDetector(_cfg(), verbose=False)
    from ultralytics import YOLO
    model = YOLO(args.yolo_weights)

    vids = sorted(Path(args.videos_dir).glob("*.mp4"))
    out_videos = {}
    fps_per_video = {}
    total_frames = 0
    for vp in vids:
        video = vp.stem
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.70
        fps_per_video[video] = round(float(fps), 3)
        recs = []
        src_fi = 0
        sampled = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            if src_fi % args.step == 0:
                H, W = img.shape[:2]
                res = model(img, conf=args.conf, classes=[9], imgsz=args.imgsz, verbose=False)[0]
                yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
                hsv_px = [s["box"] for s in det._candidates(img)]
                cands = build_candidates(yolo_px, hsv_px, W, H)
                crec = []
                for c in cands:
                    bx = c["box"]
                    crec.append({
                        "box_norm": (bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H),
                        "source": c["source"], "cx": c["cx"], "cy": c["cy"], "area": c["area"],
                    })
                recs.append({"fi": src_fi, "candidates": crec})
                sampled += 1
                if args.limit and sampled >= args.limit:
                    break
            src_fi += 1
        cap.release()
        out_videos[video] = recs
        total_frames += len(recs)
        print(f"  {video}: 源帧={src_fi}, 采样帧={len(recs)}, fps={fps:.2f}")
    payload = {"fps": 29.70, "fps_per_video": fps_per_video, "step": args.step,
               "n_videos": len(out_videos), "n_frames": total_frames, "videos": out_videos}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[out] {args.out}  ({total_frames} 采样帧, {len(out_videos)} 视频)")


if __name__ == "__main__":
    main()
