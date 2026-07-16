#!/usr/bin/env python3
"""生成 Part B 的 GT 标注骨架: 时间戳/窗预填, 像素坐标留空待 Jacob 标。

B1 斑马线掩膜: datasets/gt/crosswalk/{video}.json
    frames: 每个 is_violation==1 窗取 3 帧(窗 10/50/90% 时刻) + 1 帧窗外中性帧;
            y0/y1=None 待 Jacob 标(每帧独立标, 兼容相机漂移)。
B2 跟踪:     datasets/gt/tracking/{video}.json
    anchors: 每个 is_violation==1 窗取 2 锚帧(窗 20/70% 时刻);
            box=None 待 Jacob 标(违章车在该帧的框)。

wb 不代标 —— 只填 ts/window, 像素值一律 None。
"""
import os
import sys
import csv
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.violation_eval import load_violation_gt


def _neutral_ts(segs, dur_hint=2.0):
    """找一个不在任何违章窗内的中性时刻(用于查无假带)。无合适点返回 None。"""
    # 从 1.0s 起找第一个空隙
    cand = 1.0
    for _ in range(200):
        inside = any(s["start_s"] <= cand <= s["end_s"] for s in segs)
        if not inside:
            return round(cand, 1)
        cand += 1.0
    return None


def gen(video, segs, out_crosswalk, out_tracking):
    # ---- B1 ----
    b1 = {"video": video, "frames": []}
    for s in segs:
        dur = s["end_s"] - s["start_s"]
        for frac in (0.1, 0.5, 0.9):
            ts = round(s["start_s"] + dur * frac, 1)
            b1["frames"].append({
                "ts": ts, "y0": None, "y1": None,
                "note": f"违章窗[{s['start_s']:.0f}-{s['end_s']:.0f}]@{frac*100:.0f}%",
            })
    nts = _neutral_ts(segs)
    if nts is not None:
        b1["frames"].append({"ts": nts, "y0": None, "y1": None, "note": "中性帧(窗外, 查无假带)"})
    b1["frames"].sort(key=lambda f: f["ts"])
    path1 = os.path.join(out_crosswalk, f"{video}.json")
    with open(path1, "w", encoding="utf-8") as f:
        json.dump(b1, f, ensure_ascii=False, indent=2)

    # ---- B2 ----
    b2 = {"video": video, "anchors": []}
    for s in segs:
        dur = s["end_s"] - s["start_s"]
        frames = []
        for frac in (0.2, 0.7):
            ts = round(s["start_s"] + dur * frac, 1)
            frames.append({"ts": ts, "box": None})
        b2["anchors"].append({"window": [s["start_s"], s["end_s"]], "frames": frames})
    path2 = os.path.join(out_tracking, f"{video}.json")
    with open(path2, "w", encoding="utf-8") as f:
        json.dump(b2, f, ensure_ascii=False, indent=2)
    return path1, path2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--out-crosswalk", default=os.path.join(ROOT, "datasets", "gt", "crosswalk"))
    ap.add_argument("--out-tracking", default=os.path.join(ROOT, "datasets", "gt", "tracking"))
    args = ap.parse_args()

    os.makedirs(args.out_crosswalk, exist_ok=True)
    os.makedirs(args.out_tracking, exist_ok=True)
    gt = load_violation_gt(args.events)
    if not gt:
        print("无 is_violation==1 段, 未生成骨架")
        return
    for video, segs in gt.items():
        p1, p2 = gen(video, segs, args.out_crosswalk, args.out_tracking)
        print(f"[{video}] B1骨架={os.path.basename(p1)} ({len(segs)}窗) B2骨架={os.path.basename(p2)}")
    print(f"\n已生成骨架。像素坐标(y0/y1, box)为 null, 由 Jacob 标后 eval 才能跑。")


if __name__ == "__main__":
    main()
