#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_vehicle_track_fragmentation.py — 违章车 track 碎片化 GT-free 量化诊断(只读)

违章车被 tracker 切成多 ID, 但被 _dedup() 合并进 member_tracks 掩盖, F1 看不见。
轨迹 GT 框全 null 跑不了现成 attribution_union, 故走 GT-free:
锚定 confirmed episode 代表 track 的逐帧 box 轨迹, 量真碎片数 + 掩盖缺口 + 成因。

口径(以实际代码为准):
  cli.run(cfg, video_path, out_dir, preset, return_track_samples=True) → (events, track_samples)
  track_samples: tid -> [{ts, stationary, box, overlap, cls, conf}]
  events: confirmed episode 带 track_id(代表) + member_tracks(合并的 ID) + start_ts/end_ts
  tracker 常数: SimpleTracker iou_thresh=0.3, max_disappeared=15

用法: PYTHONPATH=src ./.venv/bin/python scripts/diag_vehicle_track_fragmentation.py
"""
import os, sys, csv, json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.evaluation.module_metrics import iou_box

REPORT = ROOT / "docs" / "reports" / "2026-08-03-qw-vehicle-track-fragmentation.md"
OUT_DIR = ROOT / "data" / "output" / "qw"
VIDEOS_CSV = ROOT / "datasets" / "gt" / "videos.csv"
CFG = ROOT / "configs" / "config.yaml"

MAX_DISAPPEARED = 15
IOU_THRESH = 0.3          # SimpleTracker iou_thresh
T_VALUES = [0.3, 0.5]     # 两档 IoU 阈值


def _load_violation_videos():
    """从 videos.csv 取 has_violation==1 的视频名。"""
    vids = []
    with open(VIDEOS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("has_violation") == "1":
                vids.append(r["video"])
    return sorted(vids)


def _box_iou(a, b):
    """像素框 IoU, 复用 module_metrics.iou_box。"""
    return iou_box(a, b)


def _episode_fragments(track_samples, ev, fps_inference, T):
    """GT-free 真碎片数: 窗内每个采样 ts 取锚 box, 统计所有 IoU≥T 的 track 并集。

    返回 (fragment_set, fragment_details, anchor_coverage)
    fragment_set: set of track IDs
    fragment_details: [{track_id, first_ts, last_ts, n_samples, boxes}]
    anchor_coverage: 锚 track 样本覆盖 episode 窗的比例
    """
    rep_tid = ev["track_id"]
    s, e = ev["start_ts"], ev["end_ts"]
    rep_samples = [sm for sm in track_samples.get(rep_tid, []) if s <= sm["ts"] <= e]

    if not rep_samples:
        return set(), [], 0.0

    # 锚覆盖率
    ts_set = {round(sm["ts"], 3) for sm in rep_samples}
    window_dur = max(e - s, 0.001)
    covered = 0.0
    for i in range(len(rep_samples) - 1):
        covered += rep_samples[i+1]["ts"] - rep_samples[i]["ts"]
    if len(rep_samples) > 0:
        covered += 1.0 / fps_inference  # 最后一个样本至少占一帧
    anchor_coverage = min(1.0, covered / window_dur)

    # 逐 ts 找 IoU≥T 的 track 并集
    # 先建 ts -> samples 索引(所有 track)
    ts_to_samples = defaultdict(list)
    for tid, samples in track_samples.items():
        for sm in samples:
            if s <= sm["ts"] <= e:
                ts_to_samples[round(sm["ts"], 3)].append((tid, sm))

    frag_set = set()
    frag_details_map = {}  # tid -> {first_ts, last_ts, n_samples, boxes}

    for sm in rep_samples:
        ts = round(sm["ts"], 3)
        anchor_box = sm["box"]
        # 同一 ts 的所有 track
        for tid, other_sm in ts_to_samples.get(ts, []):
            iou = _box_iou(anchor_box, other_sm["box"])
            if iou >= T:
                frag_set.add(tid)
                if tid not in frag_details_map:
                    frag_details_map[tid] = {
                        "track_id": tid, "first_ts": other_sm["ts"],
                        "last_ts": other_sm["ts"], "n_samples": 1,
                        "boxes": [other_sm["box"]],
                    }
                else:
                    d = frag_details_map[tid]
                    d["last_ts"] = other_sm["ts"]
                    d["n_samples"] += 1
                    d["boxes"].append(other_sm["box"])

    return frag_set, list(frag_details_map.values()), anchor_coverage


def _classify_causes(fragments, fps_inference):
    """对相邻碎片(按首现 ts 排序)逐对判成因。

    (a) 时间断裂: 间隔 > MAX_DISAPPEARED 帧 = MAX_DISAPPEARED/fps 秒
    (b) 空间跳变: 时间相接(≤阈值)但 IoU < IOU_THRESH
    (c) 并存重叠: 时间重叠且 IoU ≥ 0.5
    """
    if len(fragments) <= 1:
        return []

    frags = sorted(fragments, key=lambda x: x["first_ts"])
    causes = []
    gap_sec = MAX_DISAPPEARED / fps_inference
    time_adj_sec = 2.0 / fps_inference  # 时间相接容忍(2帧)

    for i in range(len(frags) - 1):
        prev = frags[i]
        nxt = frags[i+1]
        gap = nxt["first_ts"] - prev["last_ts"]

        # 取 prev 末 box 和 next 首 box
        prev_box = prev["boxes"][-1] if prev["boxes"] else None
        next_box = nxt["boxes"][0] if nxt["boxes"] else None
        iou = _box_iou(prev_box, next_box) if prev_box and next_box else 0.0

        if gap > gap_sec:
            cause = "a_time_gap"
        elif gap <= time_adj_sec and iou < IOU_THRESH:
            cause = "b_spatial_jump"
        elif gap < 0:  # 时间重叠
            if iou >= 0.5:
                cause = "c_duplicate"
            elif iou < IOU_THRESH:
                cause = "b_spatial_jump"
            else:
                cause = "c_duplicate"  # IoU 0.3-0.5 重叠, 偏重复
        else:
            # 时间相接但 IoU ≥ 0.3 → 正常交接, 不是碎片成因
            if iou >= IOU_THRESH:
                cause = "none"
            else:
                cause = "b_spatial_jump"

        causes.append({
            "prev_tid": prev["track_id"], "next_tid": nxt["track_id"],
            "gap_sec": round(gap, 3), "iou": round(iou, 4), "cause": cause,
        })

    return causes


def run_video(video, cfg, preset):
    """跑单视频, 返回 (episodes_results, frag_csv_rows)。"""
    video_path = str(ROOT / "input_video" / f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return [], []

    out_dir = str(OUT_DIR / f"run_{video}_{preset}")
    events, track_samples = cli.run(
        cfg, video_path, out_dir, preset=preset,
        return_track_samples=True,
    )

    fps_inference = cfg.inference.fps
    confirmed = [ev for ev in events if ev.get("status") == "confirmed"]
    print(f"  [{video}] confirmed episodes={len(confirmed)} track_ids={len(track_samples)}")

    ep_results = []
    frag_rows = []
    for idx, ev in enumerate(confirmed):
        s, e = ev["start_ts"], ev["end_ts"]
        rep_tid = ev["track_id"]
        member_tracks = ev.get("member_tracks", [])
        masked_frag = len(member_tracks)

        # 两档 T 的真碎片数
        frag_data = {}
        for T in T_VALUES:
            frag_set, frag_details, coverage = _episode_fragments(
                track_samples, ev, fps_inference, T)
            frag_data[T] = {
                "frag_set": frag_set,
                "frag_details": frag_details,
                "coverage": coverage,
                "true_frag": len(frag_set),
                "gap": len(frag_set) - masked_frag,
            }

        # 成因分类(用 T=0.3 的碎片)
        fd = frag_data[0.3]
        causes = _classify_causes(fd["frag_details"], fps_inference)
        cause_counts = defaultdict(int)
        for c in causes:
            cause_counts[c["cause"]] += 1

        ep = {
            "video": video, "episode_idx": idx,
            "start_s": round(s, 2), "end_s": round(e, 2),
            "rep_track_id": rep_tid,
            "anchor_coverage": f"{fd['coverage']:.3f}",
            "masked_frag": masked_frag,
            "truefrag_iou03": fd["true_frag"],
            "truefrag_iou05": frag_data[0.5]["true_frag"],
            "gap_iou03": frag_data[0.3]["gap"],
            "gap_iou05": frag_data[0.5]["gap"],
            "cause_a_timegap": cause_counts.get("a_time_gap", 0),
            "cause_b_jump": cause_counts.get("b_spatial_jump", 0),
            "cause_c_dup": cause_counts.get("c_duplicate", 0),
            "cause_none": cause_counts.get("none", 0),
        }
        ep_results.append(ep)

        # 逐碎片 CSV 行
        for fi, frag in enumerate(sorted(fd["frag_details"], key=lambda x: x["first_ts"])):
            prev_cause = causes[fi-1]["cause"] if fi > 0 and fi-1 < len(causes) else "first"
            frag_rows.append({
                "video": video, "episode_idx": idx,
                "frag_track_id": frag["track_id"],
                "first_ts": round(frag["first_ts"], 3),
                "last_ts": round(frag["last_ts"], 3),
                "n_samples": frag["n_samples"],
                "is_rep": int(frag["track_id"] == rep_tid),
                "in_member_tracks": int(frag["track_id"] in member_tracks),
                "cause_vs_prev": prev_cause,
            })

        print(f"    ep{idx} tid={rep_tid} [{s:.1f}-{e:.1f}s] "
              f"masked={masked_frag} true03={fd['true_frag']} true05={frag_data[0.5]['true_frag']} "
              f"cov={fd['coverage']:.2f} causes={dict(cause_counts)}",
              flush=True)

    return ep_results, frag_rows


def main():
    cfg = load_config(str(CFG))
    preset = "balanced"
    videos = _load_violation_videos()
    print(f"违章视频({len(videos)}): {videos}")
    print(f"preset={preset} inference_fps={cfg.inference.fps}")
    print(f"tracker: iou_thresh={IOU_THRESH} max_disappeared={MAX_DISAPPEARED}")
    print()

    all_eps = []
    all_frags = []
    for video in videos:
        ep_results, frag_rows = run_video(video, cfg, preset)
        all_eps.extend(ep_results)
        all_frags.extend(frag_rows)

    # 聚合
    n_eps = len(all_eps)
    if n_eps == 0:
        print("无 confirmed episode, 退出")
        return

    true03 = [e["truefrag_iou03"] for e in all_eps]
    true05 = [e["truefrag_iou05"] for e in all_eps]
    masked = [e["masked_frag"] for e in all_eps]
    gaps03 = [e["gap_iou03"] for e in all_eps]
    coverages = [float(e["anchor_coverage"]) for e in all_eps]

    total_a = sum(e["cause_a_timegap"] for e in all_eps)
    total_b = sum(e["cause_b_jump"] for e in all_eps)
    total_c = sum(e["cause_c_dup"] for e in all_eps)
    total_none = sum(e["cause_none"] for e in all_eps)
    total_pairs = total_a + total_b + total_c + total_none

    print(f"\n=== 聚合 ({n_eps} confirmed episodes) ===")
    print(f"真碎片数(T=0.3): 均值={np.mean(true03):.1f} 中位={np.median(true03):.0f} "
          f"范围=[{min(true03)}, {max(true03)}]")
    print(f"真碎片数(T=0.5): 均值={np.mean(true05):.1f} 中位={np.median(true05):.0f} "
          f"范围=[{min(true05)}, {max(true05)}]")
    print(f"masked 碎片数: 均值={np.mean(masked):.1f} 范围=[{min(masked)}, {max(masked)}]")
    print(f"掩盖缺口(T=0.3): 均值={np.mean(gaps03):.1f} 总和={sum(gaps03)}")
    print(f"锚覆盖率: 均值={np.mean(coverages):.2f} <0.7的={sum(1 for c in coverages if c < 0.7)}")
    print(f"成因: a_time_gap={total_a} b_spatial_jump={total_b} c_duplicate={total_c} none={total_none} "
          f"(总对数={total_pairs})")

    # 三问
    print(f"\n=== 三问 ===")
    print(f"Q1 违章车被切几个 ID: T=0.3 均值 {np.mean(true03):.1f}, "
          f"范围 [{min(true03)}, {max(true03)}]")
    print(f"Q2 _dedup 掩盖/漏多少: masked 均值 {np.mean(masked):.1f}, "
          f"真碎片均值 {np.mean(true03):.1f}, "
          f"缺口总和 {sum(gaps03)}")
    cause_total = max(total_a + total_b + total_c, 1)
    print(f"Q3 主因: a_time_gap={total_a/cause_total*100:.0f}% "
          f"b_jump={total_b/cause_total*100:.0f}% "
          f"c_dup={total_c/cause_total*100:.0f}%")

    # 写 CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ep_csv = OUT_DIR / "vehicle_track_fragmentation_per_episode.csv"
    with open(ep_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_eps[0].keys()))
        w.writeheader()
        w.writerows(all_eps)
    print(f"[csv] {ep_csv}")

    frag_csv = OUT_DIR / "vehicle_track_fragmentation_per_frag.csv"
    if all_frags:
        with open(frag_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_frags[0].keys()))
            w.writeheader()
            w.writerows(all_frags)
        print(f"[csv] {frag_csv}")

    _write_report(all_eps, all_frags, n_eps, true03, true05, masked, gaps03,
                  coverages, total_a, total_b, total_c, total_none, videos)


def _write_report(all_eps, all_frags, n_eps, true03, true05, masked, gaps03,
                  coverages, total_a, total_b, total_c, total_none, videos):
    cause_total = max(total_a + total_b + total_c, 1)
    L = [
        "# 违章车 track 碎片化 GT-free 量化诊断(qw, 只读)\n",
        "> 违章车被 tracker 切成多 ID, 但被 _dedup 合并进 member_tracks 掩盖, F1 看不见。\n",
        "> 轨迹 GT 框全 null 跑不了现成 attribution_union, 走 GT-free: 锚定 confirmed episode 代表 track 的逐帧 box 轨迹, 量真碎片数。\n",
        f"> 口径: cli.run(return_track_samples=True), T=0.3/0.5 两档, tracker iou_thresh={IOU_THRESH} max_disappeared={MAX_DISAPPEARED}\n",
        f"> 违章视频: {', '.join(videos)}\n",
        f"> 单次 cli.run 结果(非多 seed), preset=balanced\n\n",
        "## 三问裁断\n",
        f"**Q1 违章车被切几个 ID**: T=0.3 均值 {np.mean(true03):.1f}, "
        f"中位 {np.median(true03):.0f}, 范围 [{min(true03)}, {max(true03)}]; "
        f"T=0.5 均值 {np.mean(true05):.1f}, 范围 [{min(true05)}, {max(true05)}]\n",
        f"**Q2 _dedup 掩盖/漏多少**: masked 均值 {np.mean(masked):.1f}, "
        f"真碎片(T=0.3)均值 {np.mean(true03):.1f}, "
        f"掩盖缺口总和 {sum(gaps03)} (均值 {np.mean(gaps03):.1f}/episode)\n",
        f"**Q3 碎片化主因**: "
        f"时间断裂(a)={total_a} ({total_a/cause_total*100:.0f}%), "
        f"空间跳变(b)={total_b} ({total_b/cause_total*100:.0f}%), "
        f"并存重复(c)={total_c} ({total_c/cause_total*100:.0f}%)\n",
        f"\n## 聚合 ({n_eps} confirmed episodes)\n",
        f"- 真碎片数(T=0.3): 均值 {np.mean(true03):.1f}, 中位 {np.median(true03):.0f}, "
        f"范围 [{min(true03)}, {max(true03)}]",
        f"- 真碎片数(T=0.5): 均值 {np.mean(true05):.1f}, 中位 {np.median(true05):.0f}, "
        f"范围 [{min(true05)}, {max(true05)}]",
        f"- masked 碎片数(len member_tracks): 均值 {np.mean(masked):.1f}, "
        f"范围 [{min(masked)}, {max(masked)}]",
        f"- 掩盖缺口(真-masked, T=0.3): 均值 {np.mean(gaps03):.1f}, 总和 {sum(gaps03)}",
        f"- 锚覆盖率: 均值 {np.mean(coverages):.2f}, "
          f"<0.7 的 episode {sum(1 for c in coverages if c < 0.7)}/{n_eps} (数字视为下界)",
        f"- 成因分布: a_time_gap={total_a}, b_spatial_jump={total_b}, "
          f"c_duplicate={total_c}, none={total_none} (总对数={total_a+total_b+total_c+total_none})",
    ]

    # per-video
    L.append(f"\n## 逐视频分解\n")
    L.append("| 视频 | episodes | 均值碎片(T=0.3) | 均值masked | 均值缺口 | 主因 |")
    L.append("|---|---|---|---|---|---|")
    for v in videos:
        v_eps = [e for e in all_eps if e["video"] == v]
        if not v_eps:
            L.append(f"| {v} | 0 | - | - | - | - |")
            continue
        vt = [e["truefrag_iou03"] for e in v_eps]
        vm = [e["masked_frag"] for e in v_eps]
        vg = [e["gap_iou03"] for e in v_eps]
        va = sum(e["cause_a_timegap"] for e in v_eps)
        vb = sum(e["cause_b_jump"] for e in v_eps)
        vc = sum(e["cause_c_dup"] for e in v_eps)
        main_cause = max([("时间断裂", va), ("空间跳变", vb), ("并存重复", vc)], key=lambda x: x[1])
        L.append(f"| {v} | {len(v_eps)} | {np.mean(vt):.1f} | {np.mean(vm):.1f} | "
                 f"{np.mean(vg):.1f} | {main_cause[0]}({main_cause[1]}) |")

    # 逐 episode 表
    L.append(f"\n## 逐 episode 明细\n")
    L.append("| video | ep | start | end | rep_tid | coverage | masked | true03 | true05 | gap03 | a | b | c |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in all_eps:
        L.append(f"| {e['video']} | {e['episode_idx']} | {e['start_s']} | {e['end_s']} | "
                 f"{e['rep_track_id']} | {e['anchor_coverage']} | {e['masked_frag']} | "
                 f"{e['truefrag_iou03']} | {e['truefrag_iou05']} | {e['gap_iou03']} | "
                 f"{e['cause_a_timegap']} | {e['cause_b_jump']} | {e['cause_c_dup']} |")

    # 逐碎片表
    L.append(f"\n## 逐碎片明细(供 cc bit-for-bit 复核)\n")
    L.append("| video | ep | frag_tid | first_ts | last_ts | n_samples | is_rep | in_member | cause_vs_prev |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in all_frags:
        L.append(f"| {r['video']} | {r['episode_idx']} | {r['frag_track_id']} | "
                 f"{r['first_ts']} | {r['last_ts']} | {r['n_samples']} | "
                 f"{r['is_rep']} | {r['in_member_tracks']} | {r['cause_vs_prev']} |")

    L.append(f"\n## 方法学\n")
    L.append("- GT-free: 不用 GT 框, 锚定 confirmed episode 代表 track(track_id)的逐帧 box 轨迹。\n")
    L.append(f"- 真碎片数: 窗内每个采样 ts 取锚 box, 统计所有 track ID 中 box IoU≥T 的并集大小(含从未 qualify 的碎片)。\n")
    L.append(f"- masked 碎片数 = len(member_tracks)(_dedup 合并的 ID 数)。\n")
    L.append(f"- 掩盖缺口 = 真碎片数 − masked(_dedup 之外还漏的碎片)。\n")
    L.append(f"- 成因分类: 相邻碎片(按首现 ts 排序)逐对判, "
              f"a=时间断裂(>{MAX_DISAPPEARED}帧), b=空间跳变(IoU<{IOU_THRESH}), c=并存重叠(IoU≥0.5)。\n")
    L.append(f"- 锚覆盖率 < 0.7 的 episode 数字视为下界(锚 track 本身碎 → undercount)。\n")
    L.append(f"- 单次 cli.run 结果(非多 seed), preset=balanced。\n")
    L.append(f"- 逐 episode CSV: `data/output/qw/vehicle_track_fragmentation_per_episode.csv`\n")
    L.append(f"- 逐碎片 CSV: `data/output/qw/vehicle_track_fragmentation_per_frag.csv`\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {REPORT}")


if __name__ == "__main__":
    main()
