"""聚合负例质量扫频 JSON + 逐 seed 提取的 diag JSON, 产出最终 5-seed 聚合 JSON。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/_agg_neg_quality.py
输出 data/output/sweep_neg_quality.json
"""
import os, sys, json, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

def _agg(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return None
    return {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4), "n": len(arr)}

# seed 2: parse diag JSON
s2 = json.load(open(os.path.join(ROOT, "data/output/diag_neg_s2.json")))
g2 = s2["gates"]

per_seed = {}
# seeds 0,1: manually read from earlier partial sweep JSON (unchanged models)
# seeds 3,4: from sweep_neg_quality_s3.json, s4.json
for seed, path in [(0, "data/output/diag_neg_s0.json"), (1, "data/output/diag_neg_s1.json")]:
    d = json.load(open(os.path.join(ROOT, path)))
    g = d["gates"]
    per_seed[seed] = {
        "gate1_neg_off": g["gate1_neg_off_ratio"]["value"],
        "gate2_06_train": g["gate2_dark_green"]["06_train"]["value"],
        "gate2_07_val": g["gate2_dark_green"]["07_val"]["value"],
        "gate4_01_neg_off": g["gate4_val_generalization"]["01_neg_off"],
        "gate4_07_pos_recall": g["gate4_val_generalization"]["07_pos_recall"],
        "gate4_11_neg_off": g["gate4_val_generalization"]["11_neg_off"],
        "gate3_probe_04": -1,
        "walk_to_off": d["domain"]["walk_to_off"],
    }
    # gate3 from probe
    p3 = g.get("gate3_probe_04")
    per_seed[seed]["gate3_probe_04"] = p3["value"] if p3 else None

# seed 2
per_seed[2] = {
    "gate1_neg_off": g2["gate1_neg_off_ratio"]["value"],
    "gate2_06_train": g2["gate2_dark_green"]["06_train"]["value"],
    "gate2_07_val": g2["gate2_dark_green"]["07_val"]["value"],
    "gate4_01_neg_off": g2["gate4_val_generalization"]["01_neg_off"],
    "gate4_07_pos_recall": g2["gate4_val_generalization"]["07_pos_recall"],
    "gate4_11_neg_off": g2["gate4_val_generalization"]["11_neg_off"],
    "gate3_probe_04": None,
    "walk_to_off": s2["domain"]["walk_to_off"],
}
p3 = g2.get("gate3_probe_04")
per_seed[2]["gate3_probe_04"] = p3["value"] if p3 else None

# seeds 3,4: from per-seed JSON files
for sfile in ["data/output/sweep_neg_quality_s3.json", "data/output/sweep_neg_quality_s4.json"]:
    d = json.load(open(os.path.join(ROOT, sfile)))
    for s, v in d.get("per_seed", {}).items():
        per_seed[int(s)] = v

metrics = list(next(iter(per_seed.values())).keys())
agg = {m: _agg([per_seed[s].get(m) for s in sorted(per_seed)]) for m in metrics}

out = {
    "config": {"seeds": sorted(per_seed), "note": "负例质量杠杆: 补55假绿off(视频10), Step0正则(dropout=0.3,wd=1e-4)"},
    "per_seed": {str(s): per_seed[s] for s in sorted(per_seed)},
    "agg_mean_std_min": agg,
}
dst = os.path.join(ROOT, "data/output/sweep_neg_quality.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=2)
print(f"-> {dst}")
print("\n=== 聚合 ===")
for m in metrics:
    a = agg[m]
    if a: print(f"  {m:20s} mean={a['mean']:.3f}  std={a['std']:.3f}  min={a['min']:.3f}  max={a['max']:.3f}  n={a['n']}")
