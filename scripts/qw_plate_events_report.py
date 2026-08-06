"""qw 车牌回填逐事件验收(cc 效果 gate 条件⑤): 回填牌 vs violating_plates 全表含负例/盲区。

判定口径(cc f26c91f 送回):
  - GT 违章牌 = events.csv violating_plates(不含 other_plates —— other 是"出现过但不算违章")
  - 回填牌 ∈ violating(ED<=1) -> 命中
  - 回填牌非空 ∉ violating -> 误罚(含 other_plates 的牌: 违章05 京N541E6 即此类)
  - 回填为空 -> 空(miss, 宁缺毋滥, 不算误罚)
  - 负例(has_violation=0)事件 = 事件层 FP(灯态线), 其贴牌计误罚但标注"事件FP所致"
  - 盲区(GT 无牌标注, 如违章11) = 无法判定, 单独标注

用法: python scripts/qw_plate_events_report.py [--videos ...]
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.evaluation.metrics import levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clean(s):
    return [x.strip() for x in (s or "").split(";") if x.strip() and x.strip() not in ("?", "无牌") and not x.strip().startswith("[")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    cfg.output.annotated_video = False
    cfg.output.evidence_images = False

    violating, other = {}, {}
    for r in csv.DictReader(open(os.path.join(ROOT, "datasets", "gt", "events.csv"), encoding="utf-8-sig")):
        vp = set(_clean(r["violating_plates"]))
        op = set(_clean(r["other_plates"]))
        # GT 违章牌只取 is_violation==1 行(cc 口径; is_violation=0 行的 violating_plates 是
        # "该时段车辆"非违章车, 如违章05 红灯段把京N541E6 也列了, 会污染违章牌集)
        if r.get("is_violation") == "1":
            if vp:
                violating.setdefault(r["video"], set()).update(vp)
        if op:
            other.setdefault(r["video"], set()).update(op)
    meta = {r["video"]: r["has_violation"]
            for r in csv.DictReader(open(os.path.join(ROOT, "datasets", "gt", "videos.csv"), encoding="utf-8-sig"))}

    videos = args.videos or [f"违章{i:02d}" for i in range(1, 12)]
    tot_hit = tot_wrong = tot_gt = 0
    print(f"{'视频':<6} {'hv':<4} {'回填牌(多牌)':<26} {'violating_GT':<22} {'判定'}")
    rows = []
    for v in videos:
        events = cli.run(cfg, os.path.join(ROOT, "input_video", f"{v}.mp4"),
                         os.path.join(ROOT, "data", "output", "qw", f"evrep_{v}"), preset="balanced")
        vgt = violating.get(v, set())
        hv = meta.get(v, "1")
        tot_gt += len(vgt)
        for e in events:
            if e["status"] != "confirmed":
                continue
            plates = e.get("plates") or ([e["plate"]] if e.get("plate") else [])
            # 逐牌判定(P1 多牌): 命中=GT 被 plates 覆盖; 误罚=plates 中非 GT 牌
            hit_n = sum(1 for g in vgt if any(levenshtein(p, g) <= 1 for p in plates))
            wrong = [p for p in plates if p and not any(levenshtein(p, g) <= 1 for g in vgt)]
            tot_hit += hit_n
            if wrong:
                if hv == "0":
                    verdict = f"误罚(负例FP事件层){wrong}"
                elif vgt:
                    verdict = f"误罚{wrong}"
                else:
                    verdict = f"盲区(无GT牌){wrong}"
                tot_wrong += len(wrong)
            elif not plates:
                verdict = "空(宁缺毋滥)"
            else:
                verdict = f"命中{hit_n}/{len(vgt)}"
            rows.append((v, hv, "|".join(plates), ",".join(sorted(vgt)), verdict))
            print(f"{v:<6} {hv:<4} {'|'.join(plates):<26} {','.join(sorted(vgt)):<22} {verdict}")
    print(f"\n汇总: 命中={tot_hit}/{tot_gt} 误罚={tot_wrong} (负例/盲区标注见逐行)")
    # 落盘 CSV
    out = os.path.join(ROOT, "data", "output", "qw", "plate_events_report.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "has_violation", "backfilled_plate", "violating_gt", "verdict"])
        for r in rows:
            w.writerow(r)
    print(f"已落盘: {out}")


if __name__ == "__main__":
    main()
