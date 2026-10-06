"""qw b2 验收总脚本(cc 五条硬条件): 全 11 逐 episode 审计 + 聚合 F1/P + 车牌 + 阈值扫描。

用法:
  python scripts/qw_b2_audit.py                     # 全 11, 生产默认阈值 200px/3s, 审计+缓存
  python scripts/qw_b2_audit.py --videos 违章05 违章09
  python scripts/qw_b2_audit.py --sweep             # 用缓存离线扫描 D×gap (不重跑流水线)

输出(data/output/qw/, gitignored):
  b2_effect_<TAG>.txt            逐 episode 审计(span+member+member_all+状态+GT命中+车牌) + 聚合
  _b2_cache/raw_<video>.json      该视频 raw 事件 + track_samples 缓存(离线扫描用)
  b2_effect_sweep.txt             阈值扫描表(D × gap -> 每视频收窄 member 数)
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.evaluation.violation_eval import (
    load_violation_gt, match_violation_events, classify_false_positives,
    aggregate, load_video_metadata, overlap_seconds,
)
from redlight.pipeline.violation_engine import BatchViolationEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "output", "qw", "_b2_cache")


def _clean(s):
    return [x.strip() for x in (s or "").split(";")
            if x.strip() and x.strip() not in ("?", "无牌") and not x.strip().startswith("[")]


def _load_gt():
    gt_win = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "datasets", "gt", "events.csv"), encoding="utf-8-sig")):
        if r.get("is_violation") == "1":
            gt_win.setdefault(r["video"], []).append({
                "start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
                "plates": _clean(r["violating_plates"])})
    meta = {r["video"]: r["has_violation"] for r in
            csv.DictReader(open(os.path.join(ROOT, "datasets", "gt", "videos.csv"), encoding="utf-8-sig"))}
    return gt_win, meta


def _run_one(cfg, v, tag, cache=False):
    """跑一个视频: 返回 (events, raw_pool, track_samples)。

    raw_pool: _dedup 的原始输入(供阈值扫描离线重算)
    track_samples: 引擎逐帧样本(供离线重算 member 归组)
    """
    import redlight.pipeline.violation_engine as ve
    spy = {}
    orig_dedup = ve.BatchViolationEngine._dedup

    def _spy(self, raw_events):
        spy["raw"] = [dict(e) for e in raw_events]
        return orig_dedup(self, raw_events)

    ve.BatchViolationEngine._dedup = _spy
    try:
        events, samples = cli.run(cfg,
                                  os.path.join(ROOT, "input_video", f"{v}.mp4"),
                                  os.path.join(ROOT, "data", "output", "qw", f"_b2audit_{tag}_{v}"),
                                  preset="balanced", return_track_samples=True)
    finally:
        ve.BatchViolationEngine._dedup = orig_dedup
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        slim = {}
        for tid, ss in samples.items():
            slim[str(tid)] = [{"ts": s["ts"], "stationary": s.get("stationary", False),
                               "box": [float(x) for x in s["box"]],
                               "overlap": s.get("overlap", 0.0)} for s in ss]
        with open(os.path.join(CACHE, f"raw_{v}.json"), "w", encoding="utf-8") as f:
            json.dump({"raw": spy.get("raw", []), "samples": slim}, f, ensure_ascii=False)
    return events, spy.get("raw", []), samples


def _fmt(e, k_s="start_ts", k_e="end_ts"):
    return f"[{e[k_s]:.1f}-{e[k_e]:.1f}]"


def audit(cfg, videos, tag, cache=True, out=None):
    gt_win, meta = _load_gt()
    lines, results = [], []
    print(f"=== [{tag}] 生产默认阈值: centroid-d=200 gap-merge=3 ===")
    for v in videos:
        evs, raw, _samples = _run_one(cfg, v, tag, cache=cache)
        conf = [e for e in evs if e["status"] == "confirmed"]
        gts = gt_win.get(v, [])
        r = match_violation_events(conf, gts, min_overlap_s=0.5)
        cls = classify_false_positives(r, conf, gts, is_negative=(meta.get(v, "1") != "1"))
        r["neg_count"], r["oow_count"], r["fragment_count"] = (
            cls["neg_count"], cls["oow_count"], cls["fragment_count"])
        results.append(r)
        hv = meta.get(v, "?")
        line = (f"[{v}] has_viol={hv} confirmed={len(conf)} TP={r['tp']} FP={r['fp']} "
                f"FN={r['fn']} 覆盖={r['mean_coverage']:.2f} 真误报={cls['neg_count'] + cls['oow_count']} "
                f"碎片={cls['fragment_count']}")
        print(line); lines.append(line)
        for i, e in enumerate(conf):
            best = None
            for g in gts:
                ov = overlap_seconds((e["start_ts"], e["end_ts"]), (g["start_s"], g["end_s"]))
                if ov >= 0.5:
                    best = f"GT[{g['start_s']:.0f}-{g['end_s']:.0f}]cov=" \
                           f"{ov / max(g['end_s'] - g['start_s'], 1e-6):.2f}"
                    break
            ep = (f"  ep{i} rep={e['track_id']} span={_fmt(e)} "
                  f"member={len(e.get('member_tracks', []))} "
                  f"member_all={len(e.get('member_tracks_all', []))} "
                  f"状态={e['status']} → {best or '窗外!!'} "
                  f"plate={e.get('plate', '')!r} plates={e.get('plates', [])}")
            print(ep); lines.append(ep)
        for d in cls["detail"]:
            print(f"      [FP:{d['category']}] {d['span']} 灯态={d['light_state'] or '?'}")
            lines.append(f"  FP[{d['category']}] {d['span']}")
    agg = aggregate(results)
    agg_line = (f"\n=== [{tag}] 聚合: TP={agg['tp']} FP={agg['fp']} FN={agg['fn']} "
                f"P={agg['precision']:.3f} R={agg['recall']:.3f} F1={agg['f1']:.3f} "
                f"| 真误报={agg['true_fp_total']} 碎片={agg['fragment_total']} "
                f"P(仅真误报)={agg['p_only_true_fp']:.3f} ===")
    print(agg_line); lines.append(agg_line)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[已存] {out}")
    return agg


def sweep(cfg, videos):
    """离线阈值扫描(硬条件④): 用缓存 raw+samples 重算 _dedup+_narrow_members。"""
    Ds, Gs = [150, 200, 250], [2, 3, 5]
    print("\n=== 阈值扫描(缓存离线, 零重跑流水线): 每视频 confirmed episode 的 member 总数 ===")
    rows = ["video\t" + "\t".join(f"D{d}g{g}" for d in Ds for g in Gs)]
    print(rows[0])
    for v in videos:
        p = os.path.join(CACHE, f"raw_{v}.json")
        if not os.path.exists(p):
            print(f"  [无缓存] {v}: 先不带 --sweep 跑一遍")
            continue
        with open(p, encoding="utf-8") as f:
            cache = json.load(f)
        raw, samples = cache["raw"], {int(k): ss for k, ss in cache["samples"].items()}
        if not raw:
            continue
        eng = BatchViolationEngine("balanced")
        eng._track_samples = samples
        counts = []
        for d in Ds:
            for g in Gs:
                eng.b2_centroid_d, eng.b2_gap_merge = float(d), float(g)
                eps = eng._narrow_members(eng._dedup([dict(e) for e in raw]))
                counts.append(str(sum(len(e.get("member_tracks", [])) for e in eps)))
        row = f"{v}\t" + "\t".join(counts)
        print(row); rows.append(row)
    with open(os.path.join(ROOT, "data", "output", "qw", "b2_effect_sweep.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print("\n[已存] data/output/qw/b2_effect_sweep.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="B2FIX")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--sweep", action="store_true", help="离线阈值扫描(用缓存)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    cfg.output.annotated_video = False
    cfg.output.evidence_images = False

    videos = args.videos or [f"违章{i:02d}" for i in range(1, 12)]
    if args.sweep:
        sweep(cfg, videos)
        return
    out = os.path.join(ROOT, "data", "output", "qw", f"b2_effect_{args.tag}.txt")
    audit(cfg, videos, args.tag, cache=not args.no_cache, out=out)


if __name__ == "__main__":
    main()