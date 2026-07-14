"""将画廊人工校验反馈 (ped_signal_feedback.json) 合并回 labels.csv。

反馈格式 (make_ped_signal_gallery.py 导出):
  [{"crop_path": "...", "label": "walk|stand|off", "video":..., "t":...}, ...]
  label="delete" -> 从 labels.csv 删该行(难判/废图)。

合并规则:
  - 校验过的 crop: verified=1, label 用人工标(覆盖弱标签)
  - label=delete: 删行
  - 未在反馈里的 crop: 保持原样(verified=0, 弱标签)

用法:
  python scripts/apply_ped_signal_feedback.py
  python scripts/apply_ped_signal_feedback.py --feedback path/to/ped_signal_feedback.json
"""
import os
import sys
import csv
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.data_pipeline.ped_signal_dataset import LABELS_HEADER


def main():
    ap = argparse.ArgumentParser(description="合并人工校验反馈到 labels.csv")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "ped_signal", "labels.csv"))
    ap.add_argument("--feedback", default=os.path.join(ROOT, "data", "output", "ped_signal_gallery", "ped_signal_feedback.json"))
    args = ap.parse_args()

    if not os.path.exists(args.feedback):
        print(f"反馈文件不存在: {args.feedback}")
        print("先在画廊里校验并导出 ped_signal_feedback.json")
        return

    with open(args.feedback, encoding="utf-8") as f:
        feedback = json.load(f)
    fb_map = {item["crop_path"]: item for item in feedback}

    with open(args.labels, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    kept = []
    verified = 0
    deleted = 0
    for r in rows:
        fb = fb_map.get(r["crop_path"])
        if fb is None:
            kept.append(r)
            continue
        if fb.get("label") == "delete":
            deleted += 1
            continue
        r["label"] = fb["label"]
        r["verified"] = "1"
        verified += 1
        kept.append(r)

    with open(args.labels, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABELS_HEADER)
        w.writeheader()
        w.writerows(kept)

    print(f"[OK] 已合并反馈 -> {args.labels}")
    print(f"  校验置 verified=1: {verified}")
    print(f"  删除(难判/废): {deleted}")
    print(f"  未校验(保持弱标签 verified=0): {len(kept) - verified}")
    print(f"  labels.csv 总行数: {len(kept)}")
    print(f"下一步: python scripts/train_ped_signal.py  (默认 --verified-only 只用 verified=1 训练)")


if __name__ == "__main__":
    main()
