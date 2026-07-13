"""逐帧车牌识别: 对整段视频每个采样帧都做 HyperLPR3 识别, 统计每帧结果与置信度。

用法:
    python scripts/scan_plates_per_frame.py <video> [--fps 8] [--truth 京JLE560]
输出:
    data/output/plate_scan/<video_name>/per_frame.csv
    控制台打印: 每帧车牌/置信度/位置/颜色, 以及 Top-N 汇总。
"""
import os
import sys
import csv
import argparse
from collections import Counter

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer
from redlight.evaluation.metrics import levenshtein


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=r"E:\BaiduNetdiskDownload\违章02.mp4")
    ap.add_argument("--fps", type=int, default=8, help="采样帧率")
    ap.add_argument("--truth", default="京JLE560", help="真值车牌用于比对")
    ap.add_argument("--conf", type=float, default=0.0, help="最低置信度过滤, 0=全部")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(ROOT, "data", "output", "plate_scan", video_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("无法打开视频:", args.video)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    interval = max(1, int(round(fps / args.fps)))

    plate = PlateRecognizer(cfg)

    rows = []
    frame_idx = 0
    proc = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            proc += 1
            ts = frame_idx / fps
            plates = plate.detect(frame)
            if not plates:
                if proc % 10 == 0:
                    print(f"[帧{frame_idx:5d} @{ts:6.2f}s] 无车牌")
            for p in plates:
                txt = p.get("text", "")
                conf = p.get("conf", 0.0)
                color = p.get("color", "unknown")
                x1, y1, x2, y2 = p.get("xyxy", [0, 0, 0, 0])
                w = int(x2 - x1)
                h = int(y2 - y1)
                ed = levenshtein(txt, args.truth) if txt else -1
                rows.append({
                    "frame": frame_idx, "time": round(ts, 2), "plate": txt,
                    "conf": round(conf, 4), "color": color,
                    "x1": int(x1), "y1": int(y1), "w": w, "h": h,
                    "edit_dist_to_truth": ed,
                })
                flag = " ✅" if ed == 0 else (" ~" if ed <= 2 else "")
                print(f"[帧{frame_idx:5d} @{ts:6.2f}s] 车牌={txt:12s} conf={conf:.3f} "
                      f"颜色={color:7s} 尺寸={w}x{h} 编辑距离={ed}{flag}")
        frame_idx += 1
    cap.release()

    # 写 CSV
    csv_path = os.path.join(out_dir, "per_frame.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time", "plate", "conf", "color",
                    "x1", "y1", "w", "h", "edit_dist_to_truth"])
        for r in rows:
            w.writerow([r["frame"], r["time"], r["plate"], r["conf"], r["color"],
                        r["x1"], r["y1"], r["w"], r["h"], r["edit_dist_to_truth"]])

    # 汇总
    print("\n" + "=" * 80)
    print(f"采样帧数: {proc}  识别到车牌的帧数: {len(rows)}")
    print(f"真值: {args.truth}")
    print(f"CSV 已保存: {csv_path}")

    if not rows:
        return

    # Top 车牌
    cnt = Counter(r["plate"] for r in rows if r["plate"])
    print("\n=== 车牌识别频次 Top-10 ===")
    for txt, n in cnt.most_common(10):
        confs = [r["conf"] for r in rows if r["plate"] == txt]
        avg = sum(confs) / len(confs)
        ed = levenshtein(txt, args.truth)
        flag = " ✅" if ed == 0 else (" ~" if ed <= 2 else "")
        print(f"  {txt:12s} 次数={n:3d} 平均conf={avg:.3f} 编辑距离={ed}{flag}")

    # 完全匹配真值的帧
    exact = [r for r in rows if r["plate"] == args.truth]
    print(f"\n=== 完全匹配真值 '{args.truth}' 的帧: {len(exact)} ===")
    for r in exact:
        print(f"  帧{r['frame']:5d} @{r['time']:.2f}s conf={r['conf']:.3f} "
              f"尺寸={r['w']}x{r['h']} 位置=({r['x1']},{r['y1']})")

    # 编辑距离<=2 的近似帧
    near = [r for r in rows if 0 < r["edit_dist_to_truth"] <= 2]
    print(f"\n=== 近似匹配 (编辑距离1~2) 的帧: {len(near)} ===")
    for r in near[:20]:
        print(f"  帧{r['frame']:5d} @{r['time']:.2f}s 识别={r['plate']:12s} conf={r['conf']:.3f} "
              f"尺寸={r['w']}x{r['h']} 编辑距离={r['edit_dist_to_truth']}")

    # 置信度分布
    confs = [r["conf"] for r in rows]
    confs_sorted = sorted(confs)
    n = len(confs)
    print(f"\n=== 置信度分布 ===")
    print(f"  min={confs_sorted[0]:.3f}  p25={confs_sorted[n//4]:.3f}  "
          f"median={confs_sorted[n//2]:.3f}  p75={confs_sorted[3*n//4]:.3f}  "
          f"max={confs_sorted[-1]:.3f}")

    # 车牌尺寸 vs 置信度 (尺寸越大通常越准)
    print(f"\n=== 尺寸 vs 置信度 (按宽度分组) ===")
    groups = {"<60": [], "60-100": [], "100-150": [], ">=150": []}
    for r in rows:
        wv = r["w"]
        if wv < 60:
            groups["<60"].append(r)
        elif wv < 100:
            groups["60-100"].append(r)
        elif wv < 150:
            groups["100-150"].append(r)
        else:
            groups[">=150"].append(r)
    for k, grp in groups.items():
        if not grp:
            continue
        avg = sum(r["conf"] for r in grp) / len(grp)
        ed_avg = sum(r["edit_dist_to_truth"] for r in grp) / len(grp)
        print(f"  宽度{k:8s} 帧数={len(grp):3d} 平均conf={avg:.3f} 平均编辑距离={ed_avg:.2f}")


if __name__ == "__main__":
    main()
