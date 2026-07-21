"""合并分步跑出的 Step0 扫频 JSON(避免重跑已完成的 seed)。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/merge_sweep_step0.py \
      data/output/sweep_step0_seed0.json data/output/sweep_step0.json
把第一个文件里的 seed 并入第二个(覆盖同 seed), 重算 mean±std+min, 写回第二个。
"""
import os
import sys
import json
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _agg(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return None
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "n": len(arr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extra")        # 含额外 seed 的 JSON
    ap.add_argument("target")       # 主 JSON(被并入并写回)
    args = ap.parse_args()

    extra = json.load(open(args.extra, encoding="utf-8"))
    target = json.load(open(args.target, encoding="utf-8"))

    per_seed = dict(target.get("per_seed", {}))
    per_seed.update(extra.get("per_seed", {}))   # 覆盖同 seed

    metrics = list(next(iter(per_seed.values())).keys())
    agg = {m: _agg([per_seed[s].get(m) for s in per_seed]) for m in metrics}

    target["per_seed"] = per_seed
    target["agg_mean_std_min"] = agg
    target.setdefault("config", {})["merged_from"] = [args.extra, args.target]
    target["config"]["seeds"] = sorted(int(s) for s in per_seed.keys())

    with open(args.target, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=2)
    print(f"合并后种子数={len(per_seed)} -> {args.target}")
    print("\n=== 聚合 (mean±std, worst-seed min) ===")
    for m in metrics:
        a = agg[m]
        if a:
            print(f"  {m:20s} mean={a['mean']:.3f}  std={a['std']:.3f}  "
                  f"min={a['min']:.3f}  max={a['max']:.3f}  n={a['n']}")


if __name__ == "__main__":
    main()
