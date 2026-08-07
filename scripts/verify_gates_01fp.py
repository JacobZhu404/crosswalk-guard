#!/usr/bin/env python3
# verify_gates_01fp.py — #3 时序门控四道证据门 (plan-gate #5 放行后)
# 只读消费生产管线: 跑 baseline(min_run=0, 等价未修) 与 fixed(min_run=10) 两轮,
# 复用 cli.run 零循环复制。monkeypatch fuse_light 抓带 max_raw_green_run_s 的灯段。
# 红线: 不写生产码 / 禁 select_gtfree / 不碰 light_priors.json·ped_signal.pt。
import os, sys, json, argparse
from redlight.infrastructure.config import load_config, project_root
from redlight.app import cli as cli_mod
import redlight.pipeline.temporal_fusion as tf_mod

_FUSE_LAST = {}

def _patched_fuse(observations, *a, **k):
    segs = _orig_fuse(observations, *a, **k)
    _FUSE_LAST.clear()
    _FUSE_LAST["segs"] = segs
    return segs

_orig_fuse = tf_mod.fuse_light
tf_mod.fuse_light = _patched_fuse

def _green_segs(segs):
    return [(round(s["start_s"], 2), round(s["end_s"], 2), round(s.get("max_raw_green_run_s", 0.0), 3))
            for s in segs if s["state"] in ("green", "flashing")]

def _run_one(cfg, video, out_dir, min_run, input_dir):
    _FUSE_LAST.clear()
    video_path = os.path.join(input_dir, video + ".mp4")
    if not os.path.exists(video_path):
        print(f"[warn] 缺失视频: {video_path}")
        return None
    out = os.path.join(out_dir, "runs", f"{video}_mr{min_run}")
    events, _ts = cli_mod.run(cfg, video_path, out, "balanced",
                              return_track_samples=True, min_persistent_green_run_s=min_run)
    segs = _FUSE_LAST.get("segs")
    green = _green_segs(segs) if segs else []
    by_status = {}
    for e in events:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    return {
        "video": video, "min_run": min_run,
        "events_total": len(events),
        "by_status": by_status,
        "green_segs": green,
        "green_seg_min_maxrun": min([g[2] for g in green], default=None),
        "green_seg_max_maxrun": max([g[2] for g in green], default=None),
    }

def _load_true_green(gt_path):
    tg = set()
    with open(gt_path) as f:
        for line in f:
            p = [x.strip() for x in line.strip().split(",")]
            if len(p) >= 4 and p[3] == "green":
                tg.add(p[0])
    return tg


def _load_gt_segments(gt_path):
    """段级 GT: {video: [(start_s, end_s, state), ...]}。用于 G2/G4 段级对齐,
    区分真绿段与落在 red/unknown 区间的假绿段(同 01 机制)。"""
    segs = {}
    with open(gt_path) as f:
        for line in f:
            p = [x.strip() for x in line.strip().split(",")]
            if len(p) < 4:
                continue
            try:
                s = float(p[1]); e = float(p[2])
            except ValueError:
                continue
            segs.setdefault(p[0], []).append((s, e, p[3]))
    return segs


def _overlap_any(a0, a1, intervals):
    return any(not (a1 < b0 or a0 > b1) for (b0, b1) in intervals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--input-dir", default=os.path.join(project_root(), "input_video"))
    ap.add_argument("--gt", default=os.path.join(project_root(), "datasets", "gt", "events.csv"))
    ap.add_argument("--config", default=os.path.join(project_root(), "configs", "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    videos = [v for v in args.videos.split(",") if v]
    true_green = _load_true_green(args.gt)
    gt_segs = _load_gt_segments(args.gt)
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "gate_evidence.json")
    # 断点续跑: 已有 JSON 里跑过的视频直接跳过, 只对缺失/未完成的补跑
    results = {}
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            prev = old.get("per_video", {})
            for v, rec in prev.items():
                if rec.get("baseline") and rec.get("fixed"):
                    results[v] = rec
            if results:
                print(f"[resume] 复用已跑 {len(results)} 个视频, 跳过")
        except Exception as e:
            print(f"[resume] 读取旧 JSON 失败, 从头跑: {e}")
    for v in videos:
        if v in results and results[v].get("baseline") and results[v].get("fixed"):
            print(f"[skip] {v} 已完成")
            continue
        base = _run_one(cfg, v, args.output, 0.0, args.input_dir)
        fixed = _run_one(cfg, v, args.output, 10.0, args.input_dir)
        results[v] = {"baseline": base, "fixed": fixed}
        # 增量落盘(防长任务 SIGKILL 丢中间结果)
        with open(out_path, "w") as f:
            json.dump({"per_video": results, "true_green_videos": sorted(true_green), "partial": True},
                      f, ensure_ascii=False, indent=2)

    # ---- 聚合四门 ----
    gate = {"G1_negative": {}, "G2_no_regression": {}, "G4_bus_occlusion": {}, "review_delta": {}}
    # G1: 01 误绿段消除(confirmed green 在 fixed 下消失) + 视频10 无新增 confirmed
    for v in ("违章01", "违章10"):
        if v not in results: continue
        b, f = results[v]["baseline"], results[v]["fixed"]
        if b is None or f is None: continue
        b_conf = b["by_status"].get("confirmed", 0)
        f_conf = f["by_status"].get("confirmed", 0)
        gate["G1_negative"][v] = {
            "baseline_confirmed": b_conf, "fixed_confirmed": f_conf,
            "baseline_green_segs": b["green_segs"], "fixed_green_segs": f["green_segs"],
            "eliminated": b_conf > 0 and f_conf == 0,
        }
    # G2: 真绿视频的"真绿段"(与 GT green 区间重叠)不回退 —— 段级对齐排除落在 red 区间的假绿段
    for v in sorted(true_green):
        if v not in results: continue
        b, f = results[v]["baseline"], results[v]["fixed"]
        if b is None or f is None: continue
        gt_green = [(s, e) for (s, e, st) in gt_segs.get(v, []) if st == "green"]
        b_true = sum(1 for seg in b["green_segs"] if _overlap_any(seg[0], seg[1], gt_green))
        f_true = sum(1 for seg in f["green_segs"] if _overlap_any(seg[0], seg[1], gt_green))
        gate["G2_no_regression"][v] = {
            "baseline_true_green_segs": b_true,
            "fixed_true_green_segs": f_true,
            "no_regression": f_true >= b_true,
        }
    # G4: 对抗式扫真绿视频"真绿段"(与 GT green 区间重叠)的 min max_run, 主动证明无 <10s 被误降级
    for v in sorted(true_green):
        if v not in results: continue
        f = results[v]["fixed"]
        if f is None: continue
        gt_green = [(s, e) for (s, e, st) in gt_segs.get(v, []) if st == "green"]
        true_segs = [seg for seg in f["green_segs"] if _overlap_any(seg[0], seg[1], gt_green)]
        min_run = min([s[2] for s in true_segs], default=None)
        gate["G4_bus_occlusion"][v] = {
            "true_green_segs": true_segs,
            "min_max_run": min_run,
            "safe": (min_run is None) or (min_run >= 10.0),
        }
    # review 增量: fixed review - baseline review(全局)
    tot_b_rev = sum(results[v]["baseline"]["by_status"].get("review", 0) for v in results if results[v]["baseline"])
    tot_f_rev = sum(results[v]["fixed"]["by_status"].get("review", 0) for v in results if results[v]["fixed"])
    tot_b_conf = sum(results[v]["baseline"]["by_status"].get("confirmed", 0) for v in results if results[v]["baseline"])
    tot_f_conf = sum(results[v]["fixed"]["by_status"].get("confirmed", 0) for v in results if results[v]["fixed"])
    gate["review_delta"] = {
        "baseline_confirmed_total": tot_b_conf, "fixed_confirmed_total": tot_f_conf,
        "baseline_review_total": tot_b_rev, "fixed_review_total": tot_f_rev,
        "review_increment": tot_f_rev - tot_b_rev,
    }
    # 总体 verdict
    g1_ok = gate["G1_negative"].get("违章01", {}).get("eliminated", False)
    g2_ok = all(d["no_regression"] for d in gate["G2_no_regression"].values())
    g4_ok = all(d["safe"] for d in gate["G4_bus_occlusion"].values())
    gate["verdict"] = {
        "G1_01_eliminated": g1_ok,
        "G2_no_regression": g2_ok,
        "G4_bus_occlusion_safe": g4_ok,
        "ALL_PASS": g1_ok and g2_ok and g4_ok,
    }

    with open(out_path, "w") as f:
        json.dump({"per_video": results, "gates": gate, "true_green_videos": sorted(true_green)},
                  f, ensure_ascii=False, indent=2)

    # ---- 打印表 ----
    print("=== G1 负例(误绿消除) ===")
    for v, d in gate["G1_negative"].items():
        print(f"  {v}: baseline_confirmed={d['baseline_confirmed']} fixed_confirmed={d['fixed_confirmed']} eliminated={d['eliminated']}")
    print("=== G2 真绿视频 TP 不回退 ===")
    for v, d in gate["G2_no_regression"].items():
        print(f"  {v}: base_true={d['baseline_true_green_segs']} fixed_true={d['fixed_true_green_segs']} no_regress={d['no_regression']}")
    print("=== G4 对抗式公交/遮挡真绿段 min max_run (阈值10s) ===")
    for v, d in gate["G4_bus_occlusion"].items():
        print(f"  {v}: min_max_run={d['min_max_run']} safe={d['safe']} segs={d['true_green_segs']}")
    print(f"=== review 增量: base_rev={tot_b_rev} fixed_rev={tot_f_rev} (+{tot_f_rev-tot_b_rev}) ; confirmed {tot_b_conf}->{tot_f_conf} ===")
    print(f"=== VERDICT: G1={g1_ok} G2={g2_ok} G4={g4_ok} ALL_PASS={gate['verdict']['ALL_PASS']} ===")
    print(f"[json] {out_path}")

if __name__ == "__main__":
    main()
