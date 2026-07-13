"""从人工标注构建灯态回归评测集(防回退用).

输入: data/output/annotated/light_feedback.csv (serve_gallery 落盘)
筛选: verdict == 'algo_wrong' 且 gt 有效 -> 固化用例 (video, t_sec, expected_state=gt)
      'other' 等模糊项不入回归集(留给融合层/人工).

输出: datasets/gt/light_regression.csv

用法:
  python scripts/build_light_regression.py
  python scripts/build_light_regression.py --src <其他标注.csv>
"""
import os
import sys
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "output", "annotated", "light_feedback.csv")
DST = os.path.join(ROOT, "datasets", "gt", "light_regression.csv")
VALID_STATES = {"red", "green", "flashing", "unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    args = ap.parse_args()

    rows = []
    if os.path.exists(args.src):
        with open(args.src, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("verdict") != "algo_wrong":
                    continue
                gt = r.get("gt", "").strip()
                if gt not in VALID_STATES:
                    continue
                try:
                    t = float(r["t_sec"])
                except (ValueError, KeyError):
                    continue
                rows.append({
                    "video": r["video"],
                    "frame_idx": r.get("frame_idx", ""),
                    "t_sec": f"{t:.1f}",
                    "expected_state": gt,
                    "pred_was": r.get("pred", ""),
                    "reason": r.get("reason", ""),
                })

    # 按 (video, frame_idx) 去重(优先 frame_idx, 回退 t_sec)
    seen = {}
    for r in rows:
        key = (r["video"], r.get("frame_idx") or r["t_sec"])
        seen[key] = r
    out = sorted(seen.values(), key=lambda r: (r["video"], float(r["t_sec"])))

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame_idx", "t_sec", "expected_state", "pred_was", "reason"])
        w.writeheader()
        w.writerows(out)

    from collections import Counter
    c = Counter(r["video"] for r in out)
    print(f"回归用例: {len(out)} 条 (来自 {args.src})")
    for v, n in sorted(c.items()):
        print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
