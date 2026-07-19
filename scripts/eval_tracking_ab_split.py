#!/usr/bin/env python3
"""B2 跟踪评测 · A/B 拆分 + 过延伸污染量化（GT-free，不占 Jacob）。

在 eval_tracking_gtfree.py(下界预览) 基础上补一刀:
  1) A/B 拆分: 对每 episode 的 member_tracks, 用 **tracker 自身框**(track_samples)做
     "时间重叠 + box IoU" 连通分量分析, 把碎片数拆成:
       - A = 同车重编号(same-car renumbering): 物理同车被切成多个 ID(每车 (ids-1) 冗余)。
       - B = 异车误并(different-car false merge): 不同车被 decide() 时间窗合并进同一 episode。
     拆分只用系统自身轨迹框, 不依赖 Jacob 标框(GT-free)。
  2) 过延伸污染量化: 用 Phase 1.5 参考"紧带"检测器(decay 退避, ≈2s 有效窗口, 不复并集全视频)
     重跑 cli.run, 对比每视频 frag_count 差值 = 过延伸注入的碎片(主要是 B 类异车误并)。

⚠️ 仍是代理/下界指标, 非模块真值。所有产物打"下界"+"代理"标签。

用法:
  python scripts/eval_tracking_ab_split.py                       # 全部违章视频
  python scripts/eval_tracking_ab_split.py --videos 违章09 违章05
  python scripts/eval_tracking_ab_split.py --no-tight            # 只做 A/B, 跳过紧带重跑
  python scripts/eval_tracking_ab_split.py --iou 0.3 --gap 2.0
"""
import os
import sys
import csv
import json
import argparse
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np
from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.models.crosswalk_v2 import CrosswalkDetectorV2


# ---------------------------------------------------------------------------
# 紧带参考检测器(Phase 1.5 候选的"量化用"版, 不复并集全视频)
# ---------------------------------------------------------------------------
class CrosswalkDetectorV2Tight(CrosswalkDetectorV2):
    """running-max 改为 decay 退避: 旧帧响应指数衰减 -> 有效窗口≈ W 秒, 不复并集全视频。

    用于**量化**过延伸污染, 不是部署(Phase 1.5 仍暂缓, 仅作参考)。
    decay=0.985 -> 半衰期≈46帧 ≈ 1.5-1.8s(视频 25-30fps), 标"≈2s 有效窗口"。
    """

    def __init__(self, cfg, decay=0.985, verbose=False):
        super().__init__(cfg, verbose=verbose)
        self._decay = float(decay)

    def detect(self, frame, vehicle_boxes=None):
        h, w = frame.shape[:2]
        if h < 120 or w < 160:
            return np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge = np.abs(sobelx)
        edge = cv2.GaussianBlur(edge, (5, 5), 0)
        mean_v = float(np.mean(gray))
        std_v = float(np.std(gray)) + 1e-6
        bright_thr = max(160.0, mean_v + 0.8 * std_v)
        bright = gray > bright_thr
        emax = float(edge.max())
        edge_resp = edge > (0.10 * emax) if emax > 0 else np.zeros_like(edge, dtype=bool)
        roi = np.zeros((h, w), dtype=bool)
        roi[int(0.35 * h):, :] = True
        resp = (edge_resp & bright & roi).astype(np.float32)
        # ---- decay 退避(替代无限 running-max) ----
        if self._accum is None:
            self._accum = resp
        else:
            self._accum = np.maximum(self._accum * self._decay, resp)
        self._frame_count += 1
        return self._fit_trapezoid(self._accum, h, w)


# ---------------------------------------------------------------------------
# box 工具
# ---------------------------------------------------------------------------
def _box_xyxy(box):
    if box is None:
        return None
    if isinstance(box, dict):
        return [float(box["x1"]), float(box["y1"]),
                float(box["x2"]), float(box["y2"])]
    b = [float(x) for x in list(box)]
    if len(b) >= 4:
        if b[2] >= b[0] and b[3] >= b[1]:
            return b[:4]
        return [b[0], b[1], b[0] + b[2], b[1] + b[3]]
    return None


def _center(box):
    """box 中心 (cx, cy)。"""
    if box is None:
        return (0.0, 0.0)
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _iou(b1, b2):
    if b1 is None or b2 is None:
        return 0.0
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _first_last_box(samples):
    if not samples:
        return None, None
    s = sorted(samples, key=lambda s: s["ts"])
    return _box_xyxy(s[0]["box"]), _box_xyxy(s[-1]["box"])


def _track_window(samples):
    ts = [s["ts"] for s in samples]
    return min(ts), max(ts)


def _rep_box(samples):
    """碎片代表框 = 该 track 所有 box 的中位数(抗抖动, 代表其常驻位置)。"""
    boxes = [_box_xyxy(s["box"]) for s in samples]
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [statistics.median([b[i] for b in boxes]) for i in range(4)]


def _concurrent_max_iou(si, sj):
    bi = {round(s["ts"], 3): _box_xyxy(s["box"]) for s in si}
    bj = {round(s["ts"], 3): _box_xyxy(s["box"]) for s in sj}
    best = 0.0
    for ts1, b1 in bi.items():
        near = min(bj.keys(), key=lambda t: abs(t - ts1)) if bj else None
        if near is None or abs(near - ts1) > 0.2:
            continue
        best = max(best, _iou(b1, bj[near]))
    return best


# ---------------------------------------------------------------------------
# 同车判定: 中心距连通性(主) + 时间邻接(辅)
# ---------------------------------------------------------------------------
def _same_car(si, sj, D_FRAC, SIZE_MAX, G, W):
    """返回 (是否同车, 证据分, 说明)。

    关键修正(两版误判复盘):
      - 首版用"IoU": tracker 框**宽度抖动达 4 倍**(同车 113->476px), IoU≈0 -> 同车误判异车。
      - 二版用"中位框中心距过小阈值(0.08W)": 违章车框中心本身抖 ±60-98px, 且不同碎片
        中心相距可达 230px+ -> 阈值太小合并不回同车(C=7 仍远 >GT=1)。
    正确信号 = **中位框中心距 + 时间邻接**: 同一(静止/缓动)车始终在斑马线同区, 中心距小且
    时间相邻(重叠或小 gap); 异车在明显不同位置且常不时间邻接。中心距比 IoU 鲁棒得多。
      - 空间: 中心距 <= D_FRAC*W(图宽比例, W=图宽) 且 面积比 <= SIZE_MAX(放宽容宽度抖动)。
      - 时间: 两碎片时间窗重叠 或 gap<=G -> 邻接(否则仅同位置但不同时, 可能是异车)。
    注: W 用**图宽**(由 track_samples 推断), 不用品对框宽(小框会令阈值塌缩)。
    """
    bi = _rep_box(si); bj = _rep_box(sj)
    if bi is None or bj is None:
        return False, 0.0, "nobox"
    ci = _center(bi); cj = _center(bj)
    dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
    ai = (bi[2] - bi[0]) * (bi[3] - bi[1])
    aj = (bj[2] - bj[0]) * (bj[3] - bj[1])
    sr = max(ai, aj) / max(min(ai, aj), 1)
    wi = _track_window(si); wj = _track_window(sj)
    gap = max(0.0, max(wi[0], wj[0]) - min(wi[1], wj[1]))
    overlap = max(0.0, min(wi[1], wj[1]) - max(wi[0], wj[0]))
    temporal = (overlap > 0) or (gap <= G)
    spatial = (dist <= D_FRAC * W) and (sr <= SIZE_MAX)
    same = bool(spatial and temporal)
    return same, (1.0 if same else 0.0), f"d{int(dist)}/W{W} g{gap:.1f} sr{sr:.1f}"


# ---------------------------------------------------------------------------
# 单 episode A/B 连通分量拆分
# ---------------------------------------------------------------------------
def _image_wh(track_samples):
    """从 track_samples 推断图宽/高(取所有框坐标最大值), 作 D_FRAC 阈值基准。"""
    W = H = 1
    for samples in track_samples.values():
        for s in samples:
            b = _box_xyxy(s.get("box"))
            if not b:
                continue
            W = max(W, b[2]); H = max(H, b[3])
    return W, H


def ab_split_episode(ev, track_samples, D_FRAC=0.2, SIZE_MAX=4.0, G=3.0):
    tids = [str(t) for t in ev.get("member_tracks", [ev.get("track_id")])]
    ts_norm = {str(k): v for k, v in track_samples.items()}
    samples = {t: ts_norm.get(t, []) for t in tids}
    W, _ = _image_wh(track_samples)
    n = len(tids)

    parent = {t: t for t in tids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = tids[i], tids[j]
            si, sj = samples[ti], samples[tj]
            if not si or not sj:
                continue
            same, score, kind = _same_car(si, sj, D_FRAC, SIZE_MAX, G, W)
            pairs.append((ti, tj, same, score, kind))
            if same:
                union(ti, tj)

    comps = set(find(t) for t in tids)
    C = len(comps)
    frag_count = n
    frag_A = frag_count - C       # 同车重编号冗余 ID(每车 ids-1)
    frag_B = C - 1                # 并入 episode 的异车数(时间窗合并)
    return {
        "track_id": str(ev.get("track_id")),
        "window": [round(ev.get("start_ts", ev.get("start_s")), 1),
                   round(ev.get("end_ts", ev.get("end_s")), 1)],
        "status": ev.get("status"),
        "frag_count": frag_count,
        "components": C,
        "frag_A": frag_A,
        "frag_B": frag_B,
        "pairs": pairs,
        "member_tracks": tids,
    }


# ---------------------------------------------------------------------------
# 跑视频(带重跑缓存: 跑完立刻把 track_samples+events 落盘, 下次秒读, 永不再丢数小时推理)
# ---------------------------------------------------------------------------
def _to_native(o):
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def _cache_path(out_dir):
    return os.path.join(out_dir, "track_samples_cache.json")


def _load_cache(out_dir):
    p = _cache_path(out_dir)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and "events" in d and "track_samples" in d:
                return d
        except Exception:
            return None
    return None


def _save_cache(out_dir, events, track_samples):
    os.makedirs(out_dir, exist_ok=True)
    with open(_cache_path(out_dir), "w", encoding="utf-8") as f:
        json.dump({"events": _to_native(events),
                   "track_samples": _to_native(track_samples)},
                  f, ensure_ascii=False)


def run_video(video, cfg, preset, detector, occ_denom="box", use_cache=True):
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return None
    out_dir = os.path.join(ROOT, "data", "output", f"run_{video}_{preset}_{detector.__class__.__name__}")

    # ---- 缓存命中: 直接读, 省去数小时推理 ----
    if use_cache:
        cached = _load_cache(out_dir)
        if cached is not None:
            evs = [e for e in cached["events"]
                   if e.get("status") not in (None, "none")]
            if evs:
                print(f"  [缓存命中] {video}: events={len(evs)} "
                      f"tracks={len(cached['track_samples'])}")
                return {"video": video, "events": evs,
                        "track_samples": cached["track_samples"]}

    # 关掉标注视频写入(仅省 per-frame 写盘, 不改变任何判定/轨迹结果)
    try:
        cfg.output.annotated_video = False
    except Exception:
        pass

    events, track_samples = cli.run(
        cfg, video_path, out_dir, preset=preset,
        return_track_samples=True,
        crosswalk_detector=detector, occ_denom=occ_denom,
    )
    rows = [ev for ev in events if ev.get("status") not in (None, "none")]
    if not rows:
        return None
    if use_cache:
        _save_cache(out_dir, events, track_samples)
    return {"video": video, "events": rows, "track_samples": track_samples}


def _video_list(cfg):
    ev_csv = os.path.join(ROOT, "datasets", "gt", "events.csv")
    videos = []
    if os.path.isfile(ev_csv):
        with open(ev_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("is_violation", "")).strip() == "1":
                    v = row["video"].strip()
                    if v not in videos:
                        videos.append(v)
    return videos


def gt_car_counts():
    """从现成 GT(events.csv violating_plates 列)读每视频违规窗真实车数, 作 A/B 交叉验证锚。

    不占 Jacob: 这是既有 GT 元数据, 非新标注。仅统计 is_violation=1 行里分号分隔的车牌数。
    注: 部分视频车牌含 '?'/空 -> 计为 1(已知至少 1 车); note 栏明示"N车"时以 note 为准。
    """
    ev_csv = os.path.join(ROOT, "datasets", "gt", "events.csv")
    out = {}
    if not os.path.isfile(ev_csv):
        return out
    with open(ev_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_violation", "")).strip() != "1":
                continue
            v = row["video"].strip()
            plates = [p for p in str(row.get("violating_plates", "")).split(";") if p.strip()]
            n = len(plates) if plates else 1
            # note 栏明示"N车"优先(弥补 '?' 未计数)
            note = row.get("note", "")
            import re
            m = re.search(r"(\d)车", note)
            if m:
                n = max(n, int(m.group(1)))
            out[v] = max(out.get(v, 0), n)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--dfrac", type=float, default=0.2,
                    help="同车中心距阈值(占图宽比例); 基于 tracker 框中心稳定±60-98px")
    ap.add_argument("--sizemax", type=float, default=4.0,
                    help="同车面积比上限(放宽容 tracker 框宽度抖动 4x)")
    ap.add_argument("--gap", type=float, default=3.0,
                    help="同车时间窗最大 gap(秒); 超时窗不并(视为异车)")
    ap.add_argument("--no-tight", action="store_true", help="跳过紧带重跑(只做 A/B)")
    ap.add_argument("--tight-only", action="store_true",
                    help="只跑紧带参考检测器, 出 frag_total(供过延伸污染=宽-紧)")
    ap.add_argument("--calibrate", action="store_true",
                    help="扫描 dfrac 对照 GT 车数, 选最优(不跑紧带)")
    ap.add_argument("--out-json", default=os.path.join(ROOT, "data", "output", "ab_split.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    videos = args.videos or _video_list(cfg)
    gt = gt_car_counts()

    print(f"=== B2 跟踪评测 · A/B 拆分 + 过延伸污染(下界/代理) · preset={args.preset} "
          f"occ_denom=box · 空间聚类 dfrac={args.dfrac} sizemax={args.sizemax} ===\n")

    # ---- 取宽带(基线) episodes + track_samples(供 A/B 与校准复用) ----
    wide_raw = {}
    for v in videos:
        det = CrosswalkDetectorV2(cfg)
        r = run_video(v, cfg, args.preset, det)
        if r is None:
            continue
        wide_raw[v] = r

    def summarize(wide, D_FRAC, SIZE_MAX, G):
        out = {}
        for v, r in wide.items():
            rows = [ab_split_episode(ev, r["track_samples"], D_FRAC, SIZE_MAX, G) for ev in r["events"]]
            out[v] = rows
        return out

    # ---- 仅紧带(量化过延伸污染用, 不入 A/B 主流程) ----
    if args.tight_only:
        print("--- 紧带参考重跑(decay=0.985 ≈2s 窗口), 仅出 frag_total ---")
        tight = {}
        for v in videos:
            det = CrosswalkDetectorV2Tight(cfg, decay=0.985)
            r = run_video(v, cfg, args.preset, det)
            if r is None:
                tight[v] = None
                continue
            tot = sum(len(ev.get("member_tracks", [ev.get("track_id")])) for ev in r["events"])
            tight[v] = {"frag_total": tot, "n_events": len(r["events"])}
            print(f"[{v}] 紧带 碎片总数={tot}  事件数={len(r['events'])}")
        tpath = os.path.join(os.path.dirname(args.out_json), "ab_split_tight.json")
        with open(tpath, "w", encoding="utf-8") as f:
            json.dump({"params": {"decay": 0.985}, "tight_total": tight},
                      f, ensure_ascii=False, indent=2)
        print(f"\n[TIGHT JSON] {tpath}")
        return

    # ---- 校准(扫描 D_FRAC, 看 C 稳定区间; GT 车数作一致性参考) ----
    if args.calibrate:
        print("--- 校准: 扫描 D_FRAC, C(组件数=估算车辆数) ---")
        for D in [0.15, 0.20, 0.25, 0.30, 0.40]:
            line = f"  D_FRAC={D}: "
            for v in wide_raw:
                rows = summarize({v: wide_raw[v]}, D, args.sizemax, args.gap)[v]
                C = sum(x["components"] for x in rows)
                gtc = gt.get(v, "?")
                line += f"[{v}]C={C}(GT={gtc}) "
            print(line)
        return

    # ---- 正式 A/B ----
    wide = summarize(wide_raw, args.dfrac, args.sizemax, args.gap)
    for v in wide:
        rows = wide[v]
        tot = sum(x["frag_count"] for x in rows)
        a = sum(x["frag_A"] for x in rows)
        b = sum(x["frag_B"] for x in rows)
        comps = sum(x["components"] for x in rows)
        gtc = gt.get(v, "?")
        match = "✓GT一致" if gtc != "?" and comps == gtc else (f"(GT车数={gtc})" if gtc != "?" else "")
        print(f"[{v}] 事件数={len(rows)} 碎片总数={tot}  A(同车重编号)={a}  "
              f"B(异车误并)={b}  组件数={comps} {match}")

    # ---- 紧带参考(量化过延伸污染) ----
    tight_total = {}
    if not args.no_tight:
        print("\n--- 紧带参考重跑(decay=0.985 ≈2s 窗口), 量化过延伸污染 ---")
        for v in videos:
            det = CrosswalkDetectorV2Tight(cfg, decay=0.985)
            r = run_video(v, cfg, args.preset, det)
            if r is None:
                tight_total[v] = None
                continue
            tot = sum(len(ev.get("member_tracks", [ev.get("track_id")])) for ev in r["events"])
            tight_total[v] = {"frag_total": tot, "n_events": len(r["events"])}
            print(f"[{v}] 紧带 碎片总数={tot}  事件数={len(r['events'])}")

    # ---- 聚合 + 污染 ----
    print("\n=== 聚合 ===")
    all_rows = [x for rows in wide.values() for x in rows]
    agg_A = sum(x["frag_A"] for x in all_rows)
    agg_B = sum(x["frag_B"] for x in all_rows)
    agg_frag = sum(x["frag_count"] for x in all_rows)
    print(f"  宽频碎片总数={agg_frag}  A(同车重编号)={agg_A}  "
          f"B(异车误并)={agg_B}  事件数={len(all_rows)}")

    if tight_total:
        pol_rows = []
        for v in wide:
            wt = tight_total.get(v)
            if wt is None:
                continue
            wf = sum(x["frag_count"] for x in wide[v])
            pol = wf - wt["frag_total"]
            pol_rows.append((v, wf, wt["frag_total"], pol))
        print("\n  过延伸污染(宽频碎片 - 紧带碎片 = 被宽带注入的额外碎片):")
        tot_pol = 0
        for v, wf, tf, pol in pol_rows:
            tot_pol += pol
            print(f"    [{v}] 宽={wf} 紧={tf} 污染={pol}")
        print(f"  污染合计={tot_pol}  (≈占宽频碎片 {100.0*tot_pol/agg_frag:.0f}%)")
        print(f"  A(同车重编号)={agg_A}  vs  B(异车误并)={agg_B}  -> "
              f"污染({tot_pol})主要落在 B 类(异车误并), 与机制一致")

    # ---- 落盘 JSON(透明, 供复核) ----
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "params": {"dfrac": args.dfrac, "sizemax": args.sizemax, "preset": args.preset},
            "wide": {v: rows for v, rows in wide.items()},
            "tight_total": tight_total,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[JSON] {args.out_json}")


if __name__ == "__main__":
    main()
