"""逐时段红绿灯识别对照 (便于人工逐段核对 GT vs 检测器)。

对 datasets/gt/events.csv 的每一个时段:
  - 在该时段内采样视频帧(8fps)跑信号灯检测器
  - 统计检测器输出的灯态分布 + 多数投票
  - 与 GT 对照: GT灯态 / 证据(visible→期望字面灯态; inferred/occluded→期望unknown)

输出:
  data/output/segment_light_comparison.csv  (机器可读)
  data/output/segment_light_comparison.md   (人工可读, 逐段对照表)

用法:
  python scripts/seg_light_compare.py
  python scripts/seg_light_compare.py --out data/output/segment_light_comparison
"""
import sys
import os
import csv
import argparse
import glob
import collections

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
from redlight.infrastructure.config import load_config, project_root
from redlight.models.traffic_light import TrafficLightDetector

SAMPLE_FPS = 8


def fmt_dist(counter):
    order = ["green", "red", "flashing", "unknown", "none"]
    parts = []
    for k in order:
        if counter.get(k):
            parts.append(f"{k}={counter[k]}")
    # 其它(理论上无)
    for k, v in counter.items():
        if k not in order and v:
            parts.append(f"{k}={v}")
    return " ".join(parts) if parts else "-"


def expected_state(state, evidence):
    if evidence in ("inferred", "occluded"):
        return "unknown"
    return state  # visible / 空 -> 期望字面灯态(含 unknown)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "output", "segment_light_comparison"))
    args = ap.parse_args()

    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))

    # 载入段级 GT
    segments = []
    with open(os.path.join(ROOT, "datasets", "gt", "events.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            segments.append({
                "video": r["video"],
                "start": float(r["start_s"]),
                "end": float(r["end_s"]),
                "state": r["light_state"],
                "evidence": (r.get("light_evidence") or "").strip(),
                "viol": r["is_violation"],
                "note": r.get("note", "") or "",
            })

    videos = sorted(glob.glob(os.path.join(ROOT, "input_video", "违章*.mp4")))
    rows = []  # 输出行
    per_video_pred = {}  # name -> [(ts, state)]

    for V in videos:
        name = os.path.splitext(os.path.basename(V))[0]
        cap = cv2.VideoCapture(V)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(round(fps / SAMPLE_FPS)))
        det = TrafficLightDetector(cfg, verbose=False)
        preds = []
        fi = 0
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if fi % interval == 0:
                st = det.detect(fr).get("state")
                preds.append((fi / fps, st))
            fi += 1
        cap.release()
        per_video_pred[name] = preds
        print(f"[done] {name}: {len(preds)} sampled frames", flush=True)

    # 逐时段统计
    for seg in segments:
        preds = per_video_pred.get(seg["video"], [])
        sub = [s for (t, s) in preds if seg["start"] <= t <= seg["end"]]
        if not sub:
            counter = collections.Counter()
            majority = "none"
        else:
            counter = collections.Counter(sub)
            # 多数投票(忽略 none)
            vote = {k: v for k, v in counter.items() if k is not None}
            majority = max(vote, key=vote.get) if vote else "unknown"
        exp = expected_state(seg["state"], seg["evidence"])
        agree = (majority == exp)
        rows.append({
            "video": seg["video"],
            "start": seg["start"], "end": seg["end"],
            "gt_state": seg["state"], "evidence": seg["evidence"],
            "expected": exp, "majority": majority,
            "dist": fmt_dist(counter),
            "agree": "✅" if agree else "❌",
            "viol": seg["viol"], "note": seg["note"],
        })

    # 写 CSV
    csv_path = args.out + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "start", "end", "gt_state", "evidence",
                                          "expected", "majority", "dist", "agree", "viol", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 写 Markdown
    md_path = args.out + ".md"
    lines = []
    lines.append("# 逐时段红绿灯识别对照 (GT vs 检测器)\n")
    lines.append(f"- 采样帧率: {SAMPLE_FPS}fps | 检测器: TrafficLightDetector v6 (sat_min={cfg.traffic_light.sat_min})\n")
    lines.append("- **期望(检测器)**: visible 段期望字面灯态; inferred/occluded 段期望 `unknown`(灯不可见→进 review)\n")
    lines.append("- **识别多数**: 该时段内检测器输出的多数投票灯态; **识别分布**: 各类采样帧数\n")
    lines.append("")
    lines.append("| 视频 | 时段(s) | GT灯态 | 证据 | 期望 | 识别多数 | 识别分布 | 一致 | 违章 |")
    lines.append("|------|---------|--------|------|------|----------|----------|------|------|")
    for r in rows:
        seg = f"{r['start']:.0f}-{r['end']:.1f}"
        lines.append(
            f"| {r['video']} | {seg} | {r['gt_state']} | {r['evidence'] or '-'} | {r['expected']} "
            f"| **{r['majority']}** | {r['dist']} | {r['agree']} | {r['viol']} |")
    lines.append("")
    # 不一致汇总
    bad = [r for r in rows if r["agree"] == "❌"]
    lines.append(f"## 不一致时段: {len(bad)}/{len(rows)}\n")
    if bad:
        for r in bad:
            seg = f"{r['start']:.0f}-{r['end']:.1f}"
            lines.append(f"- **{r['video']} {seg}**: GT={r['gt_state']}({r['evidence'] or '-'}) "
                         f"期望 {r['expected']}, 但识别多数=**{r['majority']}** | 分布 {r['dist']} | {r['note'][:30]}")
    else:
        lines.append("- 全部一致 ✅\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n写出: {csv_path}\n      {md_path}")
    print(f"不一致时段: {len(bad)}/{len(rows)}")


if __name__ == "__main__":
    main()
