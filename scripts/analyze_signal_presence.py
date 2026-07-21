"""Phase B 回炉证据: 统计 walk/stand crop 的「信号色像素占比」, 复现 cc 的 ~60% 无信号发现。

cc 裁定(2026-07-21): 全局约六成 walk/stand crop 里根本没有对应信号色像素 ->
prior-ROI 挖矿频繁没框住信号灯, 模型在记忆 ROI 背景而非信号。本脚本用生产同款 HSV
阈值(traffic_light.py:640-642)客观量化该比例, 作为「先治数据、后加容量」的硬证据。

信号色定义(与生产 HSV 直采一致):
  green: [35,60,40]-[95,255,255]
  red:   [0,60,40]-[12,255,255] ∪ [158,60,40]-[180,255,255]

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/analyze_signal_presence.py
  PYTHONPATH=src ./.venv/bin/python scripts/analyze_signal_presence.py --labels datasets/classifier_retrain/labels.csv
"""
import os
import sys
import csv
import json
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# 与生产 traffic_light.py:640-642 完全一致
G_LO = np.array([35, 60, 40]); G_HI = np.array([95, 255, 255])
R1_LO = np.array([0, 60, 40]); R1_HI = np.array([12, 255, 255])
R2_LO = np.array([158, 60, 40]); R2_HI = np.array([180, 255, 255])

SIGNAL_FLOOR = 0.01   # 信号色像素占比 < 1% 视为「无信号」(cc 判定阈值)


def signal_ratio(img, want):
    """返回 crop 中 want∈{green,red} 信号色像素占比(0..1)。img 为 BGR。"""
    if img is None or img.size == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if want == "green":
        mask = cv2.inRange(hsv, G_LO, G_HI)
    else:  # red
        mask = cv2.inRange(hsv, R1_LO, R1_HI) | cv2.inRange(hsv, R2_LO, R2_HI)
    size = mask.size
    if size == 0:
        return None
    # ⚠️ cv2.inRange 返回 0/255(8-bit), 必须按布尔像素计数, 不能 mask.sum()(=255×真实占比)
    return float((mask > 0).sum()) / size


def load_rows(labels_csv):
    with open(labels_csv, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f)
                if r.get("label") in ("walk", "stand") and r.get("label") != "delete"]


def main():
    ap = argparse.ArgumentParser(description="统计 walk/stand crop 信号色像素占比")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "classifier_retrain", "labels.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "output", "signal_presence_report.json"))
    ap.add_argument("--floor", type=float, default=SIGNAL_FLOOR)
    args = ap.parse_args()

    rows = load_rows(args.labels)
    crops_root = os.path.dirname(args.labels)  # datasets/classifier_retrain (crop_path 已含 视频子目录)
    per_video = {}
    buckets = {"walk": {"<0.01": 0, "0.01-0.05": 0, "0.05-0.2": 0, ">=0.2": 0},
               "stand": {"<0.01": 0, "0.01-0.05": 0, "0.05-0.2": 0, ">=0.2": 0}}
    missing = {"walk": 0, "stand": 0}
    n = {"walk": 0, "stand": 0}
    no_signal = {"walk": 0, "stand": 0}

    for r in rows:
        lab = r["label"]
        n[lab] += 1
        path = os.path.join(crops_root, r["crop_path"])
        img = cv2.imread(path)
        ratio = signal_ratio(img, "green" if lab == "walk" else "red")
        if ratio is None:
            missing[lab] += 1
            continue
        v = r.get("video", "?")
        per_video.setdefault(v, {"walk": {"n": 0, "no_signal": 0}, "stand": {"n": 0, "no_signal": 0}})
        per_video[v][lab]["n"] += 1
        if ratio < args.floor:
            no_signal[lab] += 1
            per_video[v][lab]["no_signal"] += 1
        if ratio < 0.01:
            buckets[lab]["<0.01"] += 1
        elif ratio < 0.05:
            buckets[lab]["0.01-0.05"] += 1
        elif ratio < 0.2:
            buckets[lab]["0.05-0.2"] += 1
        else:
            buckets[lab][">=0.2"] += 1

    for lab in ("walk", "stand"):
        if missing[lab]:
            print(f"[warn] {lab}: {missing[lab]} 张 crop 读取失败(跳过)")

    print(f"\n=== 信号色像素占比 < {args.floor:.2f} 比例(无信号 crop) ===")
    tot_n = n["walk"] + n["stand"]
    tot_no = no_signal["walk"] + no_signal["stand"]
    for lab in ("walk", "stand"):
        if n[lab]:
            pct = 100.0 * no_signal[lab] / n[lab]
            print(f"  {lab}: {no_signal[lab]}/{n[lab]} = {pct:.1f}% 无信号")
    if tot_n:
        print(f"  全局(walk+stand): {tot_no}/{tot_n} = {100.0*tot_no/tot_n:.1f}% 无信号")

    print("\n=== 分布桶(walk/stand) ===")
    for lab in ("walk", "stand"):
        b = buckets[lab]
        print(f"  {lab}: " + "  ".join(f"{k}={b[k]}" for k in b))

    print("\n=== 分视频(无信号率) ===")
    for v in sorted(per_video):
        for lab in ("walk", "stand"):
            d = per_video[v][lab]
            if d["n"]:
                pct = 100.0 * d["no_signal"] / d["n"]
                print(f"  {v}[{lab}] {d['no_signal']}/{d['n']} = {pct:.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report = {
        "signal_floor": args.floor,
        "hsv": {"green": [G_LO.tolist(), G_HI.tolist()],
                "red": [[R1_LO.tolist(), R1_HI.tolist()], [R2_LO.tolist(), R2_HI.tolist()]]},
        "totals": {"walk_n": n["walk"], "walk_no_signal": no_signal["walk"],
                   "stand_n": n["stand"], "stand_no_signal": no_signal["stand"],
                   "global_n": tot_n, "global_no_signal": tot_no,
                   "global_no_signal_ratio": round(tot_no / tot_n, 3) if tot_n else None},
        "buckets": buckets,
        "per_video": per_video,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON -> {args.out}")


if __name__ == "__main__":
    main()
