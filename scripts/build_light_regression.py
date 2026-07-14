"""从人工标注构建灯态回归评测集(防回退用).

输入: data/output/annotated/light_feedback.csv (serve_gallery 落盘)
筛选: verdict in {algo_wrong, both_wrong, other} 且非 label_wrong -> 固化用例.
      expected_state 规则:
        - 备注含 "遮挡/unknown/没有红绿灯/看不清" -> 'unknown'
          (人类判定: 遮挡或无灯时应输出 unknown, 与 E12 安全语义一致)
        - 否则 expected_state = gt (修定位后应得正确相位态)
      label_wrong(算法其实对, GT错) 不入回归集.

合并语义: 本脚本即"合并入口" — 每次重跑都用最新 light_feedback.csv
         全量重建 datasets/gt/light_regression.csv, 自动覆盖旧的小集.

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
# 备注中出现这些词 -> 人类认为应出 unknown(遮挡/无灯/看不清)
UNKNOWN_HINTS = ("unknown", "遮挡", "没有红绿灯", "看不清")


def should_be_unknown(note):
    n = (note or "").lower()
    return any(h in n for h in UNKNOWN_HINTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    args = ap.parse_args()

    rows = []
    if os.path.exists(args.src):
        with open(args.src, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                verdict = r.get("verdict", "").strip()
                if verdict == "label_wrong":
                    continue  # 算法其实对, 不入回归
                gt = r.get("gt", "").strip()
                if gt not in VALID_STATES:
                    continue
                try:
                    t = float(r["t_sec"])
                except (ValueError, KeyError):
                    continue
                # expected_state: 遮挡/无灯 -> unknown; 否则用 GT 相位态
                expected = "unknown" if should_be_unknown(r.get("note")) else gt
                rows.append({
                    "video": r["video"],
                    "frame_idx": r.get("frame_idx", ""),
                    "t_sec": f"{t:.1f}",
                    "expected_state": expected,
                    "pred_was": r.get("pred", ""),
                    "reason": r.get("reason", ""),
                })

    # 按 (video, t_sec) 去重 — 匹配 --regression 读取键(避免同 t 重复校验)
    seen = {}
    for r in rows:
        key = (r["video"], r["t_sec"])
        seen[key] = r
    out = sorted(seen.values(), key=lambda r: (r["video"], float(r["t_sec"])))

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame_idx", "t_sec", "expected_state", "pred_was", "reason"])
        w.writeheader()
        w.writerows(out)

    from collections import Counter
    c = Counter(r["video"] for r in out)
    eu = Counter(r["expected_state"] for r in out)
    print(f"回归用例: {len(out)} 条 (来自 {args.src})")
    for v, n in sorted(c.items()):
        print(f"  {v}: {n}")
    print(f"  expected 分布: {dict(eu)}")


if __name__ == "__main__":
    main()
