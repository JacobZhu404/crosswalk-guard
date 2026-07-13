"""车牌识别评测脚本: 对所有视频进行逐帧车牌识别, 与GT对比计算准确率。

GT来源: datasets/gt/events.csv 和 datasets/gt/violation_events/*.csv
输出: 每个视频的识别结果统计, 整体准确率报告
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


def parse_gt_plates():
    gt = {}
    events_csv = os.path.join(ROOT, "datasets", "gt", "events.csv")
    with open(events_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row["video"]
            plates = row["violating_plates"] or ""
            other = row["other_plates"] or ""
            all_plates = []
            for p in plates.split(";") + other.split(";"):
                p = p.strip()
                if p and p != "?" and p != "无牌" and not p.startswith("["):
                    all_plates.append(p)
            if all_plates:
                gt[video] = list(set(all_plates))
    return gt


def evaluate_video(video_path, video_name, gt_plates, cfg, sample_fps=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ 无法打开视频: {video_name}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    interval = max(1, int(round(fps / sample_fps)))

    plate = PlateRecognizer(cfg, verbose=False)
    results = []
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
            for p in plates:
                txt = p.get("text", "")
                conf = p.get("conf", 0.0)
                if not txt or conf < 0.1:
                    continue
                x1, y1, x2, y2 = p.get("xyxy", [0, 0, 0, 0])
                w = int(x2 - x1)
                h = int(y2 - y1)
                results.append({
                    "frame": frame_idx, "time": round(ts, 2), "plate": txt,
                    "conf": round(conf, 4), "w": w, "h": h,
                })
        frame_idx += 1
    cap.release()

    if not results:
        print(f"  ⚠️ 未识别到任何车牌: {video_name}")
        return {"video": video_name, "gt_plates": gt_plates, "detected": [], "stats": {}}

    cnt = Counter(r["plate"] for r in results)
    top_plates = [(p, n, sum(r["conf"] for r in results if r["plate"] == p) / n)
                  for p, n in cnt.most_common(20)]

    matched = []
    missed = []
    for gp in gt_plates:
        found = False
        for p, n, avg_conf in top_plates:
            ed = levenshtein(p, gp)
            if ed == 0:
                matched.append({"gt": gp, "detected": p, "count": n, "avg_conf": avg_conf, "edit_dist": 0})
                found = True
                break
            elif ed <= 1:
                matched.append({"gt": gp, "detected": p, "count": n, "avg_conf": avg_conf, "edit_dist": ed})
                found = True
                break
        if not found:
            missed.append(gp)

    detected_plates = [p for p, _, _ in top_plates]
    false_positives = [p for p in detected_plates if p not in [m["detected"] for m in matched]]

    stats = {
        "total_frames": total_frames,
        "processed_frames": proc,
        "detected_plate_frames": len(results),
        "unique_plates_detected": len(cnt),
        "gt_plates_count": len(gt_plates),
        "exact_matches": len([m for m in matched if m["edit_dist"] == 0]),
        "near_matches": len([m for m in matched if m["edit_dist"] == 1]),
        "missed": len(missed),
        "false_positives": len(false_positives),
        "accuracy": len(matched) / len(gt_plates) if gt_plates else 0.0,
    }

    return {
        "video": video_name,
        "gt_plates": gt_plates,
        "matched": matched,
        "missed": missed,
        "false_positives": false_positives[:5],
        "top_plates": top_plates[:10],
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", help="指定视频文件名(不含.mp4)")
    ap.add_argument("--fps", type=int, default=8, help="采样帧率")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    gt_plates = parse_gt_plates()

    video_dir = os.path.join(ROOT, "input_video")
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

    if args.videos:
        video_files = [f"{v}.mp4" for v in args.videos if f"{v}.mp4" in video_files]

    print(f"{'='*80}")
    print(f"车牌识别评测")
    print(f"GT来源: datasets/gt/events.csv")
    print(f"视频目录: {video_dir}")
    print(f"采样帧率: {args.fps} fps")
    print(f"待测试视频: {len(video_files)} 个")
    print(f"{'='*80}\n")

    all_results = []
    total_stats = {
        "gt_plates": 0,
        "exact_matches": 0,
        "near_matches": 0,
        "missed": 0,
        "false_positives": 0,
        "videos_tested": 0,
    }

    for vf in video_files:
        video_name = os.path.splitext(vf)[0]
        video_path = os.path.join(video_dir, vf)
        gt = gt_plates.get(video_name, [])

        print(f"\n--- {video_name} ---")
        print(f"  GT车牌: {gt}")

        result = evaluate_video(video_path, video_name, gt, cfg, args.fps)
        if not result:
            continue

        all_results.append(result)

        s = result["stats"]
        print(f"  处理帧数: {s['processed_frames']}/{s['total_frames']}")
        print(f"  识别到车牌帧数: {s['detected_plate_frames']}")
        print(f"  识别到独立车牌: {s['unique_plates_detected']}")

        if result["matched"]:
            print(f"  ✅ 匹配结果:")
            for m in result["matched"]:
                flag = "精确" if m["edit_dist"] == 0 else f"近似(ED={m['edit_dist']})"
                print(f"    {m['gt']} -> {m['detected']} [{flag}] 次数={m['count']} 平均conf={m['avg_conf']:.3f}")

        if result["missed"]:
            print(f"  ❌ 未识别到: {result['missed']}")

        if result["false_positives"]:
            print(f"  ⚠️ 误检车牌: {result['false_positives']}")

        print(f"  准确率: {s['accuracy']:.1%}")

        total_stats["gt_plates"] += s["gt_plates_count"]
        total_stats["exact_matches"] += s["exact_matches"]
        total_stats["near_matches"] += s["near_matches"]
        total_stats["missed"] += s["missed"]
        total_stats["false_positives"] += s["false_positives"]
        total_stats["videos_tested"] += 1

    print(f"\n{'='*80}")
    print(f"综合评测结果")
    print(f"{'='*80}")
    print(f"测试视频数: {total_stats['videos_tested']}")
    print(f"GT车牌总数: {total_stats['gt_plates']}")
    print(f"精确匹配: {total_stats['exact_matches']}")
    print(f"近似匹配(ED=1): {total_stats['near_matches']}")
    print(f"未识别: {total_stats['missed']}")
    print(f"误检车牌数: {total_stats['false_positives']}")
    print(f"\n整体准确率: {(total_stats['exact_matches'] + total_stats['near_matches']) / total_stats['gt_plates']:.1%}"
          if total_stats['gt_plates'] > 0 else "\n整体准确率: N/A")

    out_dir = os.path.join(ROOT, "data", "output", "plate_eval")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "plate_eval_results.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "gt_plates", "detected_unique", "exact_matches", "near_matches",
                    "missed", "false_positives", "accuracy"])
        for r in all_results:
            s = r["stats"]
            w.writerow([
                r["video"],
                ",".join(r["gt_plates"]),
                s["unique_plates_detected"],
                s["exact_matches"],
                s["near_matches"],
                ",".join(r["missed"]),
                ",".join(r["false_positives"]),
                f"{s['accuracy']:.2%}",
            ])
    print(f"\n结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
