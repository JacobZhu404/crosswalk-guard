"""qw 接线护栏②聚合: 2×2 消融 4 组 violations.csv → 事件级 P/R/F1 表。

4 组(v11+mask / v2+mask / v11+box / v2+box)同 harness(cli.run + match_violation_events)
聚合, 拆解"v2 时序聚合"与"denom=box"各自对 F1 的贡献。

用法:
    python scripts/qw_wiring_ablation.py [--videos-csv datasets/gt/videos.csv] [--events datasets/gt/violation_events.json]
数据源(默认):
    v11+mask: data/output/qw/wiring_pre/run1/run_{v}/violations.csv   (护栏①接线前快照, 双跑 0-diff)
    v2+mask : data/output/qw/abl_v2mask/run_{v}/violations.csv        (新跑)
    v11+box : data/output/qw/abl_v11box/run_{v}/violations.csv        (新跑)
    v2+box  : ../crosswalk-guard/data/output/qw/sweep_{v}_bo0.20/violations.csv (sweep 复用)
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.violation_eval import load_violation_gt, match_violation_events, aggregate

VIDEOS = [f"违章{i:02d}" for i in range(1, 12)]


def read_csv(path):
    """读 violations.csv → events list(与 eval_violations._read_violations_csv 同构)。"""
    events = []
    if not os.path.exists(path):
        return events
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                events.append({
                    "status": (row.get("status") or "").strip(),
                    "start_ts": float(row["start_ts"]),
                    "end_ts": float(row["end_ts"]),
                    "plate": (row.get("plate") or "").strip(),
                    "track_id": row.get("track_id", ""),
                    "light_state": (row.get("light_state") or "").strip(),
                })
            except (KeyError, ValueError):
                continue
    return events


def confirmed(events):
    return [e for e in events if e.get("status") == "confirmed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-csv", default=os.path.join(ROOT, "datasets", "gt", "videos.csv"))
    ap.add_argument("--events", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--min-overlap-s", type=float, default=0.5)
    ap.add_argument("--wiring-pre", default=os.path.join(ROOT, "data", "output", "qw", "wiring_pre", "run1"))
    ap.add_argument("--abl-v2mask", default=os.path.join(ROOT, "data", "output", "qw", "abl_v2mask"))
    ap.add_argument("--abl-v11box", default=os.path.join(ROOT, "data", "output", "qw", "abl_v11box"))
    ap.add_argument("--sweep-root", default=os.path.join(ROOT, "..", "crosswalk-guard", "data", "output", "qw"))
    args = ap.parse_args()

    from redlight.evaluation.violation_eval import load_video_metadata
    gt = load_violation_gt(args.events)
    meta = load_video_metadata(args.videos_csv)

    groups = {
        "v11+mask(基线)": {v: os.path.join(args.wiring_pre, f"run_{v}", "violations.csv") for v in VIDEOS},
        "v2+mask":        {v: os.path.join(args.abl_v2mask, f"run_{v}", "violations.csv") for v in VIDEOS},
        "v11+box":        {v: os.path.join(args.abl_v11box, f"run_{v}", "violations.csv") for v in VIDEOS},
        "v2+box(0.20)":   {v: os.path.join(args.sweep_root, f"sweep_{v}_bo0.20", "violations.csv") for v in VIDEOS},
    }

    summary = {}
    detail = {g: {} for g in groups}
    for g, paths in groups.items():
        per_video_results = []
        for v in VIDEOS:
            gt_v = gt.get(v, [])
            r = match_violation_events(confirmed(read_csv(paths[v])), gt_v, min_overlap_s=args.min_overlap_s)
            per_video_results.append(r)
            detail[g][v] = {"TP": r["tp"], "FP": r["fp"], "FN": r["fn"]}
        agg = aggregate(per_video_results)
        summary[g] = {"TP": agg["tp"], "FP": agg["fp"], "FN": agg["fn"],
                      "P": agg["precision"], "R": agg["recall"], "F1": agg["f1"]}

    print(f"{'组合':<16} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>6} {'R':>6} {'F1':>6}")
    for g, s in summary.items():
        print(f"{g:<16} {s['TP']:>3} {s['FP']:>3} {s['FN']:>3} {s['P']:>6.3f} {s['R']:>6.3f} {s['F1']:>6.3f}")

    print("\n=== 逐视频(TP/FP/FN) ===")
    print(f"{'视频':<6} " + " ".join(f"{g:<12}" for g in groups))
    for v in VIDEOS:
        row = []
        for g in groups:
            d = detail[g][v]
            row.append(f"{d['TP']}/{d['FP']}/{d['FN']:<9}")
        print(f"{v:<6} " + " ".join(row))

    # 贡献拆解
    print("\n=== 贡献拆解 ===")
    base = summary["v11+mask(基线)"]["F1"]
    v2dim = summary["v2+mask"]["F1"]
    boxdim = summary["v11+box"]["F1"]
    full = summary["v2+box(0.20)"]["F1"]
    print(f"基线 v11+mask F1={base:.3f}")
    print(f"v2 维度(v11+mask→v2+mask): {base:.3f}→{v2dim:.3f} (Δ{v2dim-base:+.3f})")
    print(f"box 维度(v11+mask→v11+box): {base:.3f}→{boxdim:.3f} (Δ{boxdim-base:+.3f})")
    print(f"联合(v2+box): {base:.3f}→{full:.3f} (Δ{full-base:+.3f})")


if __name__ == "__main__":
    main()
