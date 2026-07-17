#!/usr/bin/env python3
"""B2 跟踪评测 — GT-free 预览(标"下界")。

⚠️ 这是**代理指标 / 下界**,不是模块真值:
   用系统自身检测出的违章事件(window + 代表 track_id / member_tracks)作"违章车"时间定位代理,
   取自 cli.run(return_track_samples=True) 返回的 track_samples。不依赖 Jacob 标框(GT-anchored 精版另做)。
   它只能反映"已检出违章窗内跟踪稳不稳",不能度量因跟踪失败导致的漏检。报告与所有产物均打"下界"标签。

指标(逐违章事件):
   - 碎片化数 frag_count = len(member_tracks): 物理同车被切成几个 ID。
   - ID 切换数 = 窗内逐帧"主导 tid"(box 最近代表框者)变化次数(跨 None 不计数)。
   - 断裂数 breaks = 窗内"无任何 member_track 在场"的连续间隙段数。
   - 覆盖率 coverage = 窗内"至少 1 个 member_track 在场"的帧占比。

用法:
   python scripts/eval_tracking_gtfree.py                       # 跑全部有 violation_events 的视频
   python scripts/eval_tracking_gtfree.py --videos 违章02 违章03
"""
import os
import sys
import json
import argparse
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.models.crosswalk_v2 import CrosswalkDetectorV2


# ---------- box 工具 ----------
def _box_xyxy(box):
    if box is None:
        return None
    if isinstance(box, dict):
        return [box["x1"], box["y1"], box["x2"], box["y2"]]
    b = list(box)
    if len(b) >= 4:
        if b[2] >= b[0] and b[3] >= b[1]:
            return b[:4]          # xyxy
        return [b[0], b[1], b[0] + b[2], b[1] + b[3]]  # xywh
    return None


def _center(box):
    b = _box_xyxy(box)
    if not b:
        return None
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _median_box(boxes):
    xy = [_box_xyxy(b) for b in boxes if _box_xyxy(b)]
    if not xy:
        return None
    return [statistics.median([b[i] for b in xy]) for i in range(4)]


def _dist(a, b):
    if a is None or b is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ---------- 单事件分析 ----------
def analyze_event(ev, track_samples):
    tids = [str(t) for t in ev.get("member_tracks", [ev.get("track_id")])]
    a, b = float(ev.get("start_ts", ev.get("start_s"))), float(ev.get("end_ts", ev.get("end_s")))
    rep = str(ev.get("track_id"))
    # 归一化 track_samples 的 key 为 str(兼容 int/str), 避免查找全 miss
    ts_norm = {str(k): v for k, v in track_samples.items()}

    # 代表框 = 代表 track 在窗内 box 的中位数(更稳); 退化用首个 member
    rep_samples = [s for s in ts_norm.get(rep, []) if a <= s["ts"] <= b]
    B = _median_box([s["box"] for s in rep_samples]) or _median_box(
        [s["box"] for t in tids for s in ts_norm.get(t, []) if a <= s["ts"] <= b]
    )

    # 窗内逐帧主导 tid
    grid = {}
    for tid in tids:
        for s in ts_norm.get(tid, []):
            if a <= s["ts"] <= b:
                grid.setdefault(round(s["ts"], 2), []).append((tid, _center(s["box"])))
    tss = sorted(grid.keys())
    dom_seq = []
    for ts in tss:
        cands = grid[ts]
        if not cands:
            dom_seq.append((ts, None))
            continue
        best = min(cands, key=lambda c: _dist(c[1], _center(B)))
        dom_seq.append((ts, best[0]))

    switches = 0
    for i in range(1, len(dom_seq)):
        pa, pb = dom_seq[i - 1][1], dom_seq[i][1]
        if pa is not None and pb is not None and pa != pb:
            switches += 1
    breaks = 0
    in_break = False
    for _, d in dom_seq:
        if d is None:
            if not in_break:
                breaks += 1
                in_break = True
        else:
            in_break = False
    n_present = sum(1 for _, d in dom_seq if d is not None)
    coverage = (n_present / len(dom_seq)) if dom_seq else 0.0

    return {
        "track_id": rep,
        "member_tracks": tids,
        "frag_count": len(tids),
        "id_switches": switches,
        "breaks": breaks,
        "coverage": round(coverage, 3),
        "window": [round(a, 1), round(b, 1)],
        "status": ev.get("status"),
        "n_frames": len(dom_seq),
    }


def run_video(video, cfg, preset):
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return None
    # 每视频新建 detector 实例: v2 的 _accum 状态带分辨率, 跨视频复用会 shape 不匹配
    detector = CrosswalkDetectorV2(cfg)
    out_dir = os.path.join(ROOT, "data", "output", f"run_{video}_{preset}")
    events, track_samples = cli.run(
        cfg, video_path, out_dir, preset=preset,
        return_track_samples=True,
        crosswalk_detector=detector, occ_denom="box",
    )
    rows = []
    for ev in events:
        if ev.get("status") in (None, "none"):
            continue
        rows.append(analyze_event(ev, track_samples))
    if not rows:
        return None
    return {"video": video, "events": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--violation-events", default=os.path.join(ROOT, "datasets", "gt", "violation_events"))
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.videos:
        videos = args.videos
    else:
        # 默认: 从 datasets/gt/events.csv 取 is_violation=1 的视频(全量违章视频)
        import csv
        ev_csv = os.path.join(ROOT, "datasets", "gt", "events.csv")
        videos = []
        if os.path.isfile(ev_csv):
            with open(ev_csv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if str(row.get("is_violation", "")).strip() == "1":
                        v = row["video"].strip()
                        if v not in videos:
                            videos.append(v)
        if not videos:
            print("events.csv 无 is_violation=1 行, 改用 --videos 指定")
            return

    print(f"=== B2 跟踪评测 · GT-free 预览(下界) · preset={args.preset} · detector=v2 occ_denom=box ===\n")
    results = []
    for v in videos:
        r = run_video(v, cfg, args.preset)
        if r is None:
            continue
        results.append(r)
        print(f"[{v}]")
        for e in r["events"]:
            flag = "⚠️碎片化" if e["frag_count"] > 1 else "✓"
            print(f"    窗{e['window']} {flag} 代表tid={e['track_id']} "
                  f"碎片数={e['frag_count']} ID切换={e['id_switches']} "
                  f"断裂={e['breaks']} 覆盖={e['coverage']:.3f} status={e['status']}")

    # 聚合
    all_ev = [e for r in results for e in r["events"]]
    if all_ev:
        frag = [e["frag_count"] for e in all_ev]
        cov = [e["coverage"] for e in all_ev]
        sw = [e["id_switches"] for e in all_ev]
        br = [e["breaks"] for e in all_ev]
        n_frag = sum(1 for f in frag if f > 1)
        print(f"\n=== 聚合({len(all_ev)} 个违章事件, {len(results)} 视频) ===")
        print(f"  碎片化(碎片数>1)事件数 = {n_frag}/{len(all_ev)}")
        print(f"  碎片数: mean={statistics.mean(frag):.2f} max={max(frag)}")
        print(f"  ID切换: mean={statistics.mean(sw):.2f} max={max(sw)}")
        print(f"  断裂数: mean={statistics.mean(br):.2f} max={max(br)}")
        print(f"  覆盖率: mean={statistics.mean(cov):.3f} min={min(cov):.3f}")
        print(f"  (对照端到端 eval 命中平均覆盖=0.904)")


if __name__ == "__main__":
    main()
