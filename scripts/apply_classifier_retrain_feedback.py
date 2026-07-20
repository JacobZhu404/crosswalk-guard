"""合并 light-state 重训数据集抽检反馈 -> labels.csv (verified=1)。

读取 make_classifier_retrain_gallery.py 导出的 classifier_retrain_feedback.json,
对每条 feedback: 按 crop_path 匹配 labels.csv 行 -> 改 label + verified=1;
label=="delete" -> 删除该行(难判废图)。写回 labels.csv, 并重新统计 manifest 计数。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/apply_classifier_retrain_feedback.py
"""
import os
import sys
import csv
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

HEADER = ["crop_path", "video", "frame_ts", "x1", "y1", "x2", "y2", "source", "label", "verified"]
VALID = {"walk", "stand", "off"}


def main():
    ap = argparse.ArgumentParser(description="合并抽检反馈到 labels.csv")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "classifier_retrain", "labels.csv"))
    ap.add_argument("--feedback", default=os.path.join(ROOT, "data", "output", "classifier_retrain_gallery", "classifier_retrain_feedback.json"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "datasets", "classifier_retrain", "manifest.json"))
    args = ap.parse_args()

    if not os.path.isfile(args.feedback):
        print(f"[跳过] 无反馈文件: {args.feedback}")
        return

    with open(args.feedback, encoding="utf-8") as f:
        fb = json.load(f)
    fb_by_path = {item["crop_path"]: item for item in fb}

    rows = []
    with open(args.labels, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    n_changed = 0
    n_deleted = 0
    n_verified = 0
    out = []
    for r in rows:
        cp = r["crop_path"]
        if cp in fb_by_path:
            item = fb_by_path[cp]
            new_label = item.get("label")
            if new_label == "delete":
                n_deleted += 1
                continue
            if new_label in VALID:
                if r["label"] != new_label:
                    r["label"] = new_label
                    n_changed += 1
                r["verified"] = 1
                n_verified += 1
        out.append(r)

    with open(args.labels, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(out)

    # 重新统计 manifest 计数
    if os.path.isfile(args.manifest):
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        counts = {v: {"walk": 0, "stand": 0, "off": 0, "impostor": 0,
                      "impostor_outside": 0, "prior_off": 0, "verified": 0} for v in manifest.get("counts", {})}
        for r in out:
            v = r["video"]
            if v not in counts:
                continue
            counts[v][r["label"]] = counts[v].get(r["label"], 0) + 1
            src = r["source"]
            if src == "impostor":
                counts[v]["impostor"] += 1
            elif src == "impostor_outside":
                counts[v]["impostor_outside"] += 1
            elif src == "prior_off":
                counts[v]["prior_off"] += 1
            if int(r.get("verified", 0) or 0) == 1:
                counts[v]["verified"] += 1
        manifest["counts"] = counts
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] 合并反馈 -> {args.labels}")
    print(f"  反馈条数={len(fb)} 改标={n_changed} 删除={n_deleted} 标记verified={n_verified}")
    print(f"  剩余行数={len(out)} (原 {len(rows)})")


if __name__ == "__main__":
    main()
