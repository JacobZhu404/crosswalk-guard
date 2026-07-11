"""读取 diag_*_timeline.csv, 按 light_state 汇总静止车与压线分布。"""
import csv
import os

BASE = os.path.join("data", "output")
for name in ("违章01", "违章02"):
    path = os.path.join(BASE, f"diag_{name}_timeline.csv")
    if not os.path.exists(path):
        print(f"[SKIP] {path}")
        continue
    states = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s = r["light_state"]
            d = states.setdefault(s, {"n": 0, "stat_sum": 0, "stat_frames": 0, "mv": []})
            d["n"] += 1
            ns = int(r["n_stationary"])
            d["stat_sum"] += ns
            if ns > 0:
                d["stat_frames"] += 1
            d["mv"].append(float(r["max_overlap"]))
    print(f"\n===== {name} =====")
    for s, d in sorted(states.items(), key=lambda x: -x[1]["n"]):
        mv = d["mv"]
        ge = sum(1 for x in mv if x >= 0.2)
        print(f"  {s:9s} n={d['n']:4d}  stat_sum={d['stat_sum']:3d}  "
              f"frames_stat>0={d['stat_frames']:3d}  maxov max={max(mv):.2f} "
              f"mean={sum(mv)/len(mv):.2f}  >=0.2count={ge}")
