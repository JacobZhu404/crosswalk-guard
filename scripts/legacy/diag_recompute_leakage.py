"""统一口径重算 B-only 与 neg_a 的漏绿/误绿, 供 cc 定 neg_a 是否净收益。

口径严格对齐 cc 裁定(2026-08-04-cc-verify-wb-nega-lovo.md):
  漏绿 = count(gt_green and not(best_conf>=0.30 and best_color=='green'))
  误绿 = count(had_cands and best_cand is not None and best_color=='green' and not gt_green)
两模型均从 *_s1.jsonl(B-only, 主repo) / *_s1A.jsonl(neg_a, wb worktree) 重算,
分别报 全量 与 扣05(违章05 从误绿已扣, 漏绿一致性待定) 两种口径, per-seed + worst-seed + mean。
"""
import json, glob, re, os

BONLY_DIR = "/Users/jacob/personal/crosswalk-guard/models/governing_disc"
NEGA_DIR = "/Users/jacob/personal/crosswalk-guard-wb/models/governing_disc"
EXCLUDE = "违章05"


def load(rows_dir, pat):
    data = {}
    for f in sorted(glob.glob(os.path.join(rows_dir, pat))):
        m = re.search(r"seed(\d+)", f)
        if not m:
            continue
        seed = int(m.group(1))
        rows = [json.loads(l) for l in open(f)]
        data.setdefault(seed, []).extend(rows)
    return data


def leak(rows, exclude=None):
    return sum(
        1
        for r in rows
        if (exclude is None or r.get("video") != exclude)
        and r.get("gt_green")
        and not (r.get("best_conf", 0) >= 0.30 and r.get("best_color") == "green")
    )


def false_green_rate(rows, exclude=None):
    num = sum(
        1
        for r in rows
        if (exclude is None or r.get("video") != exclude)
        and r.get("had_cands")
        and r.get("best_cand") is not None
        and r.get("best_color") == "green"
        and not r.get("gt_green")
    )
    den = sum(1 for r in rows if exclude is None or r.get("video") != exclude)
    return (num / den * 100.0) if den else 0.0


CONFIGS = [
    ("B-only", BONLY_DIR, "rows_*_gw3_s1.jsonl"),
    ("neg_a", NEGA_DIR, "rows_*_gw3_s1A.jsonl"),
]

for label, d, pat in CONFIGS:
    data = load(d, pat)
    n_files = sum(len(v) for v in data.values())
    print(f"=== {label} (rows files={n_files}, seeds={sorted(data)}) ===")
    print(f"{'seed':<6}{'漏绿全':<8}{'漏绿扣05':<10}{'误绿扣05%':<10}")
    lf, le, fg = [], [], []
    for seed in sorted(data):
        full = leak(data[seed])
        excl = leak(data[seed], EXCLUDE)
        rate = false_green_rate(data[seed], EXCLUDE)
        lf.append(full)
        le.append(excl)
        fg.append(rate)
        print(f"{seed:<6}{full:<8}{excl:<10}{rate:.2f}%")
    print(f"{'worst':<6}{max(lf):<8}{max(le):<10}{'':<10}")
    print(f"{'mean':<6}{sum(lf)/len(lf):.1f}{'':<2}{sum(le)/len(le):.1f}{'':<4}{sum(fg)/len(fg):.2f}%")
    print()
