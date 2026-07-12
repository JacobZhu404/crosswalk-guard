"""静态图车牌识别调优脚本: 基于抽帧结果做车牌识别调优，输出逐图识别结果与GT对比报告。

用法:
    python scripts/tune_plate_static.py --video 违章02
    python scripts/tune_plate_static.py --all  (处理所有抽帧)

输出:
    - plate_results.csv: 逐图识别结果
    - accuracy_report.txt: 准确率报告
    - 误检/漏检分析
"""
import os
import sys
import csv
import argparse
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer


def levenshtein(a, b):
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def main():
    ap = argparse.ArgumentParser(description="静态图车牌识别调优")
    ap.add_argument("--video", help="指定视频名称(不含.mp4)")
    ap.add_argument("--all", action="store_true", help="处理所有抽帧")
    ap.add_argument("--conf_thres", type=float, default=0.4, help="置信度阈值")
    ap.add_argument("--prior_province", default="京", help="省份先验(如'京')")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    cfg.inference.plate_conf = args.conf_thres
    cfg.inference.plate_prior_province = args.prior_province

    manifest_path = os.path.join(ROOT, "datasets", "frames", "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"索引文件不存在，请先运行 extract_frames.py")
        return

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_frames = list(reader)

    if args.video:
        all_frames = [f for f in all_frames if f["video"] == args.video]
        if not all_frames:
            print(f"未找到视频 {args.video} 的抽帧，请先运行 extract_frames.py")
            return

    plate = PlateRecognizer(cfg, verbose=False)

    results = []
    all_detected_plates = []
    gt_plates_set = set()

    for frame_info in all_frames:
        video_name = frame_info["video"]
        frame_idx = int(frame_info["frame_idx"])
        timestamp = float(frame_info["timestamp"])
        file_path = frame_info["file_path"]
        gt_plates_str = frame_info["gt_plates"]
        
        gt_plates = [p.strip() for p in gt_plates_str.split(",")] if gt_plates_str else []
        gt_plates_set.update(gt_plates)

        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            continue
        if frame is None:
            continue

        plates = plate.detect(frame)

        for p in plates:
            txt = p.get("text", "")
            conf = p.get("conf", 0.0)
            color = p.get("color", "unknown")
            x1, y1, x2, y2 = p.get("xyxy", [0, 0, 0, 0])
            
            best_gt = None
            best_ed = float('inf')
            for gp in gt_plates:
                ed = levenshtein(txt, gp)
                if ed < best_ed:
                    best_ed = ed
                    best_gt = gp

            results.append({
                "video": video_name,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "file_path": file_path,
                "detected_plate": txt,
                "confidence": round(conf, 4),
                "color": color,
                "x1": int(x1),
                "y1": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "gt_plate": best_gt or "",
                "edit_distance": best_ed if best_gt else -1,
                "is_match": best_ed == 0,
            })

            if txt:
                all_detected_plates.append(txt)

    if not results:
        print("未识别到任何车牌")
        return

    output_dir = os.path.join(ROOT, "datasets", "frames", "analysis")
    os.makedirs(output_dir, exist_ok=True)

    results_csv = os.path.join(output_dir, "plate_results.csv")
    with open(results_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame_idx", "timestamp", "file_path",
                                          "detected_plate", "confidence", "color",
                                          "x1", "y1", "w", "h",
                                          "gt_plate", "edit_distance", "is_match"])
        w.writeheader()
        w.writerows(results)

    matches = [r for r in results if r["is_match"]]
    exact_matches = len(matches)
    unique_matched = len(set(r["detected_plate"] for r in matches))
    total_gt = len(gt_plates_set)

    cnt = Counter(all_detected_plates)
    top_plates = cnt.most_common(20)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("静态图车牌识别调优报告")
    report_lines.append("=" * 80)
    report_lines.append(f"置信度阈值: {args.conf_thres}")
    report_lines.append(f"总帧数: {len(all_frames)}")
    report_lines.append(f"识别到车牌的帧数: {len(set(r['file_path'] for r in results))}")
    report_lines.append(f"车牌检测总数: {len(results)}")
    report_lines.append(f"GT车牌总数: {total_gt}")
    report_lines.append(f"精确匹配数: {exact_matches}")
    report_lines.append(f"匹配的独立车牌数: {unique_matched}")
    report_lines.append(f"准确率: {unique_matched/total_gt*100:.1%}" if total_gt > 0 else "准确率: N/A")
    report_lines.append("")

    report_lines.append("=== GT车牌匹配情况 ===")
    for gp in gt_plates_set:
        matched_frames = [r for r in results if r["gt_plate"] == gp and r["is_match"]]
        near_matches = [r for r in results if r["gt_plate"] == gp and 0 < r["edit_distance"] <= 2]
        report_lines.append(f"  {gp}:")
        report_lines.append(f"    精确匹配: {len(matched_frames)} 帧")
        if matched_frames:
            confs = [r["confidence"] for r in matched_frames]
            report_lines.append(f"    平均置信度: {sum(confs)/len(confs):.3f}")
        report_lines.append(f"    近似匹配(ED<=2): {len(near_matches)} 帧")

    report_lines.append("")
    report_lines.append("=== 检测频次 Top-10 ===")
    for txt, n in top_plates[:10]:
        confs = [r["confidence"] for r in results if r["detected_plate"] == txt]
        avg_conf = sum(confs) / len(confs) if confs else 0
        report_lines.append(f"  {txt:12s} 次数={n:4d} 平均conf={avg_conf:.3f}")

    report_lines.append("")
    report_lines.append("=== 误检分析 ===")
    false_positives = [r for r in results if not r["is_match"] and r["detected_plate"] and not r["gt_plate"]]
    fp_counter = Counter(r["detected_plate"] for r in false_positives)
    report_lines.append(f"误检总数: {len(false_positives)}")
    for txt, n in fp_counter.most_common(10):
        report_lines.append(f"  {txt:12s} 次数={n:3d}")

    report_path = os.path.join(output_dir, "accuracy_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\n详细结果已保存: {results_csv}")
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
