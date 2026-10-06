"""qw dirty-bag 只读诊断(cc brief 2026-08-19-cc-brief-qw-b2-dirtybag-rootcause.md)。

目标(§3; 先诊断后动, 禁写生产码):
  1. 全 11 视频量化 _dedup 跨 track 合并: 每 episode 的 member 分 A(同车重编号)/B(异车误并)。
  2. 每个 B 单独判语义: 真违章车(GT 牌/窗认可) vs 搭车/过路车(拆=新记分 FP) vs GT盲盒(球回 Jacob)。
  3. 违章02 错罚单机制链(tid68 京A14672 -> B 类 -> 回填夺魁)。
  4. 阈值敏感性(IoU x 质心距全组合)透明 + 拆开新 confirmed 事件计数(修法草案安全论证输入)。

读数与责任边界:
  - A/B 分类: 以 episode 代表 track(最大 overlap)为锚, 逐共享帧框对框 IoU + 质心距
    (与 cc 2026-08-03 违章07 track20 IoU=0.122 铁证同法)。分类只自我参照, 不依赖 tracking-GT。
  - tracking-GT(datasets/gt/tracking/*.json)当前 frame.box 全为 None(盲盒未建) ->
    无 GT 车牌的窗只能给 'GT盲盒(球回 Jacob)' 判定, 不硬凑(护栏 cc §4)。
  - 判定性结论只在 episode 对 GT 窗覆盖>=0.7 的高覆盖事件上给出; 低覆盖单列(护栏 cc §4)。

用法:
  python scripts/qw_dirtybag_diag.py
  python scripts/qw_dirtybag_diag.py --videos 违章07 违章02
  python scripts/qw_dirtybag_diag.py --iou 0.3 --cd 150

产物(data/output/qw/dirtybag/, gitignored):
  member_stats_<video>.json   逐 episode 逐 member 原始 stat + 分类 + B 判定
  summary.txt                 逐视频表 + 全局聚合 + 阈值敏感性
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.pipeline.violation_engine import BatchViolationEngine
from redlight.pipeline.plate_consensus import PlateConsensus
from redlight.evaluation.violation_eval import overlap_seconds
from redlight.models.crosswalk_v2 import CrosswalkDetectorV2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "output", "qw", "dirtybag")

RAW_EVENTS = []   # _dedup spy: 合并前每个 track 事件(原始)
PLATE_READS = {}  # PlateConsensus.update spy: tid -> [(text, conf, ts)]


def _install_spies():
    global RAW_EVENTS, PLATE_READS
    RAW_EVENTS, PLATE_READS = [], {}
    import redlight.pipeline.violation_engine as ve
    orig_dedup = ve.BatchViolationEngine._dedup

    def _spy_dedup(self, raw_events):
        RAW_EVENTS.extend(dict(e) for e in raw_events)
        return orig_dedup(self, raw_events)

    orig_update = PlateConsensus.update

    def _spy_plate(self, track_id, plate_text, confidence, timestamp):
        if plate_text:
            PLATE_READS.setdefault(track_id, []).append(
                (str(plate_text), float(confidence), float(timestamp)))
        return orig_update(self, track_id, plate_text, confidence, timestamp)

    ve.BatchViolationEngine._dedup = _spy_dedup
    PlateConsensus.update = _spy_plate
    return orig_dedup, orig_update


def _restore_spies(orig_dedup, orig_update):
    import redlight.pipeline.violation_engine as ve
    ve.BatchViolationEngine._dedup = orig_dedup
    PlateConsensus.update = orig_update


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    uni = ua + ub - inter
    return inter / uni if uni > 0 else 0.0


def _centroid(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _clean_plates(s):
    return [x.strip() for x in (s or "").split(";")
            if x.strip() and x.strip() not in ("?", "无牌") and not x.strip().startswith("[")]


def _load_gt(events_csv):
    gt = {}
    with open(events_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("is_violation") or "").strip() != "1":
                continue
            try:
                s, e = float(r["start_s"]), float(r["end_s"])
            except (KeyError, ValueError):
                continue
            gt.setdefault(r["video"], []).append({
                "start_s": s, "end_s": e,
                "plates": _clean_plates(r.get("violating_plates")),
                "note": (r.get("note") or "").strip()})
    for v in gt:
        gt[v].sort(key=lambda x: x["start_s"])
    return gt


def _best_plate(tid):
    """该 track 全部车牌读取 -> PlateConsensus 同款投票(ED<=1 合并, weight=count*avg_conf)。"""
    recs = PLATE_READS.get(tid, [])
    if not recs:
        return None
    pc = PlateConsensus(keep_history=10 ** 6)
    for text, conf, ts in recs:
        pc.update(tid, text, conf, ts)
    return pc.get_best(tid)


def _own_raw_events(tid):
    """该 track 在 _dedup 合并前自己的原始事件(按 track_id 索引)。"""
    out = []
    for e in RAW_EVENTS:
        if e.get("track_id") == tid:
            out.append({
                "span": (e.get("start_s"), e.get("end_s")),
                "status": e.get("status"),
                "max_overlap": e.get("max_overlap", 0.0),
                "light": e.get("light_state"),
            })
    return out


def _shared_frame_stats(rep_ts_boxes, msamples):
    """member vs 代表车: 逐共享帧框对框 IoU + 质心距。

    rep_ts_boxes: 代表车 已按 ts 排序的 [(ts, box), ...]。
    配对: 每 member 样本二分找最近代表车样本, |dt|<=0.5s(4 帧@8fps) 才算同帧对。
    未配到任何帧(代表车当时不存在/被遮挡) -> n_shared=0, 返回质心全局对比灯
    (member 样本均值质心 vs 代表车窗内均值质心), 归 U 类且标注。"""
    import bisect
    rts = [x[0] for x in rep_ts_boxes]
    ious, cds = [], []
    for s in msamples:
        i = bisect.bisect_left(rts, s["ts"])
        cand = []
        if i < len(rts):
            cand.append((abs(rts[i] - s["ts"]), rep_ts_boxes[i][1]))
        if i > 0:
            cand.append((abs(rts[i - 1] - s["ts"]), rep_ts_boxes[i - 1][1]))
        dt, rb = min(cand, key=lambda x: x[0]) if cand else (None, None)
        if rb is None or dt > 0.5:
            continue
        ious.append(_iou(s["box"], rb))
        ca, cb = _centroid(s["box"]), _centroid(rb)
        cds.append(((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5)
    return len(ious), ious, cds


def _global_cd(samples, rep_samples):
    """未配对时的兜底: member 窗内样本均值质心 vs 代表车窗内样本均值质心。"""
    if not samples or not rep_samples:
        return None
    ca = np.mean([_centroid(s["box"]) for s in samples], axis=0)
    cb = np.mean([_centroid(s["box"]) for s in rep_samples], axis=0)
    return round(float(np.linalg.norm(ca - cb)), 1)


def classify(mst, iou_thr, cd_thr):
    """A=同车重编号(空间一致) / B=异车误并 / U=无法判定。
    A: median iou>=iou_thr OR median cd<=cd_thr;  B: iou 低 AND cd 远;  U: 无共享帧或边界。"""
    if mst["n_shared"] == 0 or mst["iou_median"] is None:
        return "U"
    if mst["iou_median"] >= iou_thr or mst["cd_median"] <= cd_thr:
        return "A"
    if mst["iou_median"] < iou_thr and mst["cd_median"] > cd_thr:
        return "B"
    return "U"


def verdict(mst, gt_windows):
    """B 类成员的语义判定(§3.2): 拆开它会产生什么。

    只处理自身有 confirmed 原始事件者(拆开后才会成为新 confirmed episode);
    自身只有 review 事件 -> 拆开也是 review episode, 不记分, P 安全。
    返回 dict:
      emergent_confirmed: bool  拆开是否会产生新 confirmed episode
      emergents: [{span, light, gt_window, overlap_s, kind}]
      kind: gt_car(读牌 in GT 牌) / bystander(读牌但非 GT 车=铁证搭车)
            / gt_unknown(读不到牌 -> GT 窗内的未读牌 GT 车 或 搭车, 盲盒球回 Jacob)
            / oow(自身事件与所有 GT 窗无重叠 -> 窗外, P 致命)
    """
    confirmed = [e for e in mst["own_events"] if e["status"] == "confirmed"]
    if not confirmed:
        return {"emergent_confirmed": False, "emergents": []}
    plate = (mst.get("plate") or "").strip()
    emergents = []
    for e in confirmed:
        best_ov, best_g = 0.0, None
        for g in gt_windows:
            ov = overlap_seconds(e["span"], (g["start_s"], g["end_s"]))
            if ov > best_ov:
                best_ov, best_g = ov, g
        if best_g is None or best_ov < 0.5:
            kind = "oow"  # 自身事件与所有GT窗无重叠: 拆=窗外confirmed, P致命
        elif plate and best_g["plates"] and plate in best_g["plates"]:
            kind = "gt_car"  # 最高票牌 in GT 窗牌 -> GT 认可真违章车(同车重编号或多车窗另一辆)
        elif plate and best_g["plates"]:
            kind = "bystander"  # GT窗有牌信息且可读牌不在其中 -> 铁证非 GT 车(搭车/过路, 拆=FP)
        else:
            kind = "gt_unknown"  # GT窗无牌/读不到牌 -> 盲盒, 球回 Jacob(11 的 '?' GT 即此类)
        emergents.append({
            "span": [round(e["span"][0], 2), round(e["span"][1], 2)],
            "light": e["light"], "max_overlap": round(e["max_overlap"], 3),
            "gt_window": f"[{best_g['start_s']:.0f}-{best_g['end_s']:.0f}]" if best_g else None,
            "overlap_s": round(best_ov, 2), "kind": kind,
        })
    return {"emergent_confirmed": True, "emergents": emergents}


def run_video(cfg, v, iou_thr, cd_thr, gt_win):
    orig = _install_spies()
    try:
        events, samples = cli.run(
            cfg, os.path.join(ROOT, "input_video", f"{v}.mp4"),
            os.path.join(OUT, f"run_{v}"), preset="balanced",
            crosswalk_detector=CrosswalkDetectorV2(cfg),
            occ_denom="box", return_track_samples=True)
    finally:
        _restore_spies(*orig)

    result = {"video": v, "raw_event_count": len(RAW_EVENTS), "episodes": []}
    for ei, ev in enumerate(events):
        rep = ev["track_id"]
        members = ev.get("member_tracks_all", ev.get("member_tracks", []))
        w0, w1 = ev["start_ts"], ev["end_ts"]
        rep_samples = [s for s in samples.get(rep, []) if w0 - 0.5 <= s["ts"] <= w1 + 0.5]
        rep_ts_boxes = sorted((s["ts"], s["box"]) for s in rep_samples)
        best_ov, best_g, cov = 0.0, None, None
        for g in gt_win.get(v, []):
            ov = overlap_seconds((w0, w1), (g["start_s"], g["end_s"]))
            if ov > best_ov:
                best_ov, best_g = ov, g
        if best_g is not None:
            cov = best_ov / max(best_g["end_s"] - best_g["start_s"], 1e-6)
        rep_life = [rep_samples[0]["ts"], rep_samples[-1]["ts"]] if rep_samples else None
        ep = {
            "ep_idx": ei, "status": ev["status"], "rep": rep,
            "rep_life": [round(rep_life[0], 2), round(rep_life[1], 2)] if rep_life else None,
            "span": [round(w0, 2), round(w1, 2)], "member_all": len(members),
            "gt_hit": best_g is not None and best_ov >= 0.5,
            "gt_window": f"[{best_g['start_s']:.0f}-{best_g['end_s']:.0f}]" if best_g else None,
            "coverage": round(cov, 3) if cov is not None else None,
            "members": [],
        }
        for m in members:
            ms = [s for s in samples.get(m, []) if w0 - 0.5 <= s["ts"] <= w1 + 0.5]
            mst = {"tid": m, "is_rep": m == rep,
                   "samples_in_window": len(ms)}
            if ms:
                mst["ts_range"] = [round(ms[0]["ts"], 2), round(ms[-1]["ts"], 2)]
            bp_rep = _best_plate(m)
            if bp_rep and m == rep:
                mst["plate"] = bp_rep["text"]
                mst["plate_count"] = bp_rep["count"]
            if m != rep:
                n, ious, cds = _shared_frame_stats(rep_ts_boxes, ms)
                mst.update({
                    "n_shared": n,
                    "iou_median": round(float(np.median(ious)), 3) if ious else None,
                    "iou_max": round(float(np.max(ious)), 3) if ious else None,
                    "iou_frac_ge03": (round(sum(1 for x in ious if x >= 0.3) / len(ious), 3)
                                      if ious else None),
                    "cd_mean": round(float(np.mean(cds)), 1) if cds else None,
                    "cd_median": round(float(np.median(cds)), 1) if cds else None,
                    "cd_global": _global_cd(ms, rep_samples),
                    "own_events": _own_raw_events(m),
                })
                bp = _best_plate(m)
                if bp:
                    mst["plate"] = bp["text"]
                    mst["plate_count"] = bp["count"]
                    mst["plate_avg_conf"] = bp["avg_conf"]
                mst["class"] = classify(mst, iou_thr, cd_thr)
                if mst["class"] == "B":
                    mst["b_verdict"] = verdict(mst, gt_win.get(v, []))
            ep["members"].append(mst)
        result["episodes"].append(ep)
    return result


def _threshold_analysis(result, gt_win):
    """阈值敏感性: 9 组合下 每视频 A/B/U 数 + B 中会产生新 confirmed 的成员数。"""
    rows = []
    for iou in (0.2, 0.3, 0.4):
        for cd in (100.0, 150.0, 200.0):
            a = b = u = emergent = 0
            for ep in result["episodes"]:
                for m in ep["members"]:
                    if m.get("is_rep"):
                        continue
                    cls = classify(m, float(iou), float(cd))
                    if cls == "A":
                        a += 1
                    elif cls == "B":
                        b += 1
                        vd = verdict(m, gt_win.get(result["video"], []))
                        if vd.get("emergent_confirmed"):
                            emergent += 1
                    else:
                        u += 1
            rows.append(f"{iou:.1f}/{cd:.0f}:A{a}B{b}U{u}新{emergent}")
    return rows


def dump_summary(videos, results, gt_win, iou_thr, cd_thr):
    os.makedirs(OUT, exist_ok=True)
    lines = []
    tot = {"A": 0, "B": 0, "U": 0, "emergent": 0,
           "gt_car": 0, "bystander": 0, "gt_unknown": 0, "oow": 0}
    for v, result in zip(videos, results):
        vsum = {"A": 0, "B": 0, "U": 0, "emergent": 0,
                "gt_car": 0, "bystander": 0, "gt_unknown": 0, "oow": 0}
        lines.append(f"\n===== [{v}] (raw 事件 {result['raw_event_count']}) =====")
        for ep in result["episodes"]:
            ac = sum(1 for m in ep["members"] if m.get("class") == "A")
            bc = sum(1 for m in ep["members"] if m.get("class") == "B")
            uc = sum(1 for m in ep["members"] if m.get("class") == "U")
            em = 0
            for m in ep["members"]:
                vd = m.get("b_verdict") or {}
                if vd.get("emergent_confirmed"):
                    for ex in vd["emergents"]:
                        em += 1
                        vsum[ex["kind"]] += 1
                        tot[ex["kind"]] += 1
            tot["emergent"] += em
            vsum["A"] += ac; vsum["B"] += bc; vsum["U"] += uc
            tot["A"] += ac; tot["B"] += bc; tot["U"] += uc
            vsum["emergent"] += em
            lines.append(f"  ep{ep['ep_idx']}[{ep['status']}] rep={ep['rep']} "
                         f"span=[{ep['span'][0]:.1f}-{ep['span'][1]:.1f}] "
                         f"member_all={ep['member_all']} → A={ac} B={bc} U={uc} "
                         f"(候选新confirmed={em})"
                         f" GT={ep['gt_window'] or '-'} 覆盖={ep['coverage']}")
            for m in ep["members"]:
                if m.get("is_rep"):
                    continue
                det = (f"cls={m['class']} iou_m={m.get('iou_median')} cd={m.get('cd_median')} "
                       f"cdg={m.get('cd_global')} 共享={m.get('n_shared')} 牌={m.get('plate') or '-'} "
                       f"ts={m.get('ts_range')} ")
                if m.get("class") == "B":
                    vd = m.get("b_verdict") or {}
                    det += f"确定={list(vd.get('emergents', [])) if vd.get('emergent_confirmed') else '无新confirmed'}"
                lines.append(f"      tid={m['tid']:>3} {det}")
            lines.append("")
        vsum["emergent"] = (vsum["gt_car"] + vsum["bystander"]
                           + vsum["gt_unknown"] + vsum["oow"])
        lines.insert(-1, f"  视频合计: A={vsum['A']} B={vsum['B']} U={vsum['U']} "
                         f"拆开新confirmed={vsum['emergent']} (GT车={vsum['gt_car']} "
                         f"搭车={vsum['bystander']} "
                         f"盲盒={vsum['gt_unknown']} 窗外={vsum['oow']})")
    lines.append(f"\n===== 聚合(iou>={iou_thr} cd<={cd_thr}) =====")
    lines.append(f"A(同车)={tot['A']}  B(异车)={tot['B']}  U(无法判)={tot['U']}")
    lines.append(f"B 中拆开会出新 confirmed episode 的成员事件数={tot['emergent']}: "
                 f"GT车={tot['gt_car']}  铁证搭车={tot['bystander']}  "
                 f"盲盒={tot['gt_unknown']}  窗外={tot['oow']}")
    lines.append("\n===== 阈值敏感性: (iou,cd) -> A/B/U + 拆新confirmed数 =====")
    for v, result in zip(videos, results):
        rows = _threshold_analysis(result, gt_win)
        lines.append(f"[{v}] " + "  ".join(rows))
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--cd", type=float, default=150.0)
    ap.add_argument("--summarize-only", action="store_true",
                    help="只用已有 member_stats JSON 重出 summary(不重跑流水线)")
    args = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    cfg.output.annotated_video = False
    cfg.output.evidence_images = False
    gt_win = _load_gt(os.path.join(ROOT, "datasets", "gt", "events.csv"))
    videos = args.videos or [f"违章{i:02d}" for i in range(1, 12)]
    results = []
    for v in videos:
        p = os.path.join(OUT, f"member_stats_{v}.json")
        if args.summarize_only and os.path.exists(p):
            print(f"[use-cache] {v}")
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            for ep in r["episodes"]:
                for m in ep["members"]:
                    if m.get("class") == "B":
                        m["b_verdict"] = verdict(m, gt_win.get(v, []))
            results.append(r)
            continue
        print(f"[run] {v} ...", flush=True)
        r = run_video(cfg, v, args.iou, args.cd, gt_win)
        results.append(r)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
    dump_summary(videos, results, gt_win, args.iou, args.cd)


if __name__ == "__main__":
    main()
