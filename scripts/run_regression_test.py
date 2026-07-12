"""回归测试脚本: 自动验证评测集准确率，避免回退。

用法:
    python scripts/run_regression_test.py
    python scripts/run_regression_test.py --verbose
"""
import os
import sys
import argparse
import csv

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer


def run_regression_test(eval_set_dir, verbose=False):
    meta_path = os.path.join(eval_set_dir, "meta.csv")
    if not os.path.exists(meta_path):
        print(f"评测集索引文件不存在: {meta_path}")
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    if not samples:
        print("评测集为空")
        return None

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    plate = PlateRecognizer(cfg, verbose=False)

    results = []
    for sample in samples:
        img_path = sample["image_path"]
        if not os.path.exists(img_path):
            print(f"图像不存在: {img_path}")
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"无法读取图像: {img_path}")
            continue

        plates = plate.detect(frame)
        detected_plates = [p["text"] for p in plates if p.get("text")]
        
        gt_plates = sample["gt_plates"].split(",") if sample["gt_plates"] else []
        original_detected = sample["detected"]
        original_status = sample["status"]

        correct = False
        for gt in gt_plates:
            if gt in detected_plates:
                correct = True
                break

        results.append({
            "video": sample["video"],
            "time": float(sample["time"]),
            "original_detected": original_detected,
            "original_status": original_status,
            "current_detected": ",".join(detected_plates),
            "gt_plates": sample["gt_plates"],
            "is_correct": correct,
            "image_path": img_path,
        })

        if verbose:
            status = "✅" if correct else "❌"
            print(f"{status} {sample['video']} @{float(sample['time']):.2f}s")
            print(f"    原识别: {original_detected} ({original_status})")
            print(f"    当前识别: {','.join(detected_plates)}")
            print(f"    GT: {gt_plates}")

    return results


def main():
    ap = argparse.ArgumentParser(description="运行车牌识别回归测试")
    ap.add_argument("--verbose", action="store_true", help="显示详细结果")
    args = ap.parse_args()

    eval_set_dir = os.path.join(ROOT, "datasets", "plate_eval_set")
    
    results = run_regression_test(eval_set_dir, args.verbose)
    
    if results is None:
        return

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"回归测试报告")
    print(f"{'='*60}")
    print(f"总样本数: {total}")
    print(f"正确识别: {correct}")
    print(f"准确率: {accuracy:.2f}%")

    original_correct = sum(1 for r in results if r["original_status"] == "correct")
    original_incorrect = sum(1 for r in results if r["original_status"] == "incorrect")
    original_missed = sum(1 for r in results if r["original_status"] == "missed")

    print(f"\n原始状态分布:")
    print(f"  正确识别: {original_correct}")
    print(f"  错误识别: {original_incorrect}")
    print(f"  未识别: {original_missed}")

    regressed = [r for r in results if r["original_status"] == "correct" and not r["is_correct"]]
    improved = [r for r in results if r["original_status"] != "correct" and r["is_correct"]]

    print(f"\n回归分析:")
    print(f"  ✅ 无回退: {len([r for r in results if r['original_status'] == 'correct' and r['is_correct']])}/{original_correct}")
    print(f"  ❌ 回退: {len(regressed)}")
    print(f"  📈 改善: {len(improved)}")

    if regressed:
        print(f"\n回退样本:")
        for r in regressed:
            print(f"  - {r['video']} @{r['time']:.2f}s")
            print(f"    原识别: {r['original_detected']}")
            print(f"    当前识别: {r['current_detected']}")
            print(f"    GT: {r['gt_plates']}")

    if improved:
        print(f"\n改善样本:")
        for r in improved:
            print(f"  - {r['video']} @{r['time']:.2f}s")
            print(f"    原状态: {r['original_status']}")
            print(f"    当前识别: {r['current_detected']}")
            print(f"    GT: {r['gt_plates']}")


if __name__ == "__main__":
    main()
