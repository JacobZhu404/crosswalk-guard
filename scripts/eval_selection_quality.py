#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# eval_selection_quality.py — §3.3 全 399 帧选灯质量评分台(LOVO + 弃权门 + R2/R4)。
#
# 落实 cc 2026-07-31 复核(fa90c20)修订(B1/B2/A1-A3):
# - B1: 评测台**驱动生产上线路径 `select_gtfree`**(L1几何+YOLO+0.4·L2时序, +governing gate),
#       不再按判别器分纯 argmax 选灯(违 R1 且测的是 shadow 路径)。补 L2 时序(canonical 同口径)。
# - B2: headline τ = **训练折派生的单一全局 τ**(`recommend_tau` 用训练视频定, 不碰测试折 R3);
#       评测不再在测试集 argmin 选 τ。τ 敏感性曲线照登但只作展示, 报告口径 τ 点不取测试集最优。
# - A1: `governing_weight` 显式定值(默认 0.3 > 0), 重排功能不再空转; 锚模式(GW=0)复现基线。
# - A2: `--anchor` 模式(无模型, 纯 base 路径 τ=0)应复现 canonical R1 扣05=2.19% 与 漏绿=80, 当冒烟 gate。
# - A3: 温度缩放 + 早停已在 governing_disc.train_model 实现; `recommend_tau` 只用真预测分。
#
# 主指标:
# - 选灯精度: governing 存在帧, 选中框与某 governing 框 IoU≥0.3 的比例.
# - 正确弃权率: no_light 帧, select 返 None(不输出绿)的比例.
# - 误绿率(头条副指标, 扣05): 选中且 color=green 但帧无绿真值, 扣05后.
# - 漏绿(硬约束 R4): governing=green 帧但 select 返 None/非绿 → ≤80 判不过.
#
# 用法(锚冒烟, 快/无训练): PYTHONPATH=src ./.venv/bin/python scripts/eval_selection_quality.py --anchor
# 用法(全量, 慢/训练):   PYTHONPATH=src ./.venv/bin/python scripts/eval_selection_quality.py --seeds 0,1,2,3,4
import json, sys, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # 使 import scripts.* 可用(复用 recommend_tau)
sys.path.insert(0, str(ROOT / "src"))
import torch
from redlight.models import governing_disc as gd
from redlight.models.ped_light_selector import select_gtfree, compute_temporal_scores, iou
from redlight.models.signal_candidates import build_candidates
from redlight.models.traffic_light import TrafficLightDetector
from scripts.train_governing_discriminator import recommend_tau

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
PED_PRIOR = {"aspect_mean": 3.0, "aspect_std": 1.0, "area_mean": 0.005, "area_std": 0.003}
TAU_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
REPORT = ROOT / "docs" / "reports" / "2026-07-31-wb-selection-quality.md"


def color_state(roi):
    if roi is None or roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g = int(cv2.inRange(hsv, np.array([35, 130, 60]), np.array([85, 255, 255])).sum()) // 255
    r = (int(cv2.inRange(hsv, np.array([0, 130, 60]), np.array([10, 255, 255])).sum()) +
         int(cv2.inRange(hsv, np.array([170, 130, 60]), np.array([180, 255, 255])).sum())) // 255
    if g == 0 and r == 0:
        return None
    return "green" if g >= r else "red"


def crop_cv2(frame, b):
    H, W = frame.shape[:2]
    px = (max(0, int(b[0] * W)), max(0, int(b[1] * H)), min(W, int(b[2] * W)), min(H, int(b[3] * H)))
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return frame[px[1]:px[3], px[0]:px[2]]


def _video_frames(gt):
    by_video = defaultdict(list)
    for fr in gt["frames"]:
        by_video[fr["video"]].append(fr)
    return by_video


def _empty_row(video, fi, g):
    gov_boxes = [tuple(b["box_norm"]) for b in g.get("boxes", []) if b.get("governing")]
    gcolors = {b["color"] for b in g.get("boxes", []) if b.get("governing")}
    return {"video": video, "fi": fi, "no_light": g.get("no_light", False),
            "gt_green": "green" in gcolors, "gov_boxes": gov_boxes,
            "best_cand": None, "best_conf": 0.0, "best_color": None, "had_cands": False}


def eval_video(video, frames, model, det, yolo, governing_weight=0.3):
    """在单视频 GT 帧上评: 走生产上线路径 select_gtfree(B1 修复)。LOVO 模型已训好(锚模式 model=None)。

    三趟: ①逐 GT 帧建候选(与 canonical 同口径: YOLO cls=9 + HSV 兜底);
           ②全视频 GT 帧间算 L2 时序(compute_temporal_scores, canonical 同口径);
           ③逐帧 select_gtfree(base=L1+YOLO+0.4·L2, +governing gate)拿选中框。
    """
    tf = gd._get_transform()
    fr_map = {int(f["source_fi"]): f for f in frames}
    cap_frames = gd._read_frames_at(video, [int(f["source_fi"]) for f in frames])
    fi_order = sorted(cap_frames.keys())

    # ① 候选
    cand_store = {}
    frecs = []
    for fi in fi_order:
        frame = cap_frames[fi]
        g = fr_map[fi]
        H, W = frame.shape[:2]
        res = yolo(frame, conf=0.05, classes=[9], imgsz=1280, verbose=False)[0]
        yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
        hsv_px = [s["box"] for s in det._candidates(frame)]
        cands_raw = build_candidates(yolo_px, hsv_px, W, H)
        cands = [{"box_norm": (c["box"][0] / W, c["box"][1] / H, c["box"][2] / W, c["box"][3] / H),
                  "source": c["source"]} for c in cands_raw]
        cand_store[fi] = (frame, cands)
        frecs.append({"candidates": [{"box_norm": c["box_norm"], "source": c["source"]} for c in cands]})

    # ② L2 时序(canonical 同口径)
    temp_list = compute_temporal_scores(frecs)
    temp_map = {fi: temp_list[k] for k, fi in enumerate(fi_order)}

    # ③ 选灯: 驱动 select_gtfree(生产上线路径)
    use_gov = (model is not None) and (governing_weight > 0)
    rows = []
    for fi in fi_order:
        frame, cands = cand_store[fi]
        g = fr_map[fi]
        gov_boxes = [tuple(b["box_norm"]) for b in g.get("boxes", []) if b.get("governing")]
        gcolors = {b["color"] for b in g.get("boxes", []) if b.get("governing")}
        if gov_boxes and gcolors <= {"unclear"}:
            continue  # UNKNOWN 排除(canonical 同口径)
        if not cands:
            rows.append(_empty_row(video, fi, g))
            continue
        gov_scores = None
        if use_gov:
            gov_scores = {j: gd.score_crop(model, tf(gd.crop_candidate(frame, tuple(c["box_norm"]))))
                          for j, c in enumerate(cands)}
        # S1: 内部不弃权(governing_threshold=0.0), 照记 best_cand+best_conf;
        #     弃权门交 metrics_for_tau 按每个扫描 τ 施加(已用 best_conf>=tau 判)。
        #     选灯赢家只由 governing_weight 决定、与 gate τ 无关 → headline 不变、敏感性曲线全段诚实。
        best = select_gtfree(cands, PED_PRIOR, temporal_scores=temp_map[fi],
                             governing_scores=gov_scores, governing_weight=governing_weight,
                             governing_threshold=0.0)
        if best is None:
            best_conf, best_color = 0.0, None
        else:
            bj = cands.index(best)
            best_conf = gov_scores.get(bj, 0.0) if gov_scores is not None else 0.0
            best_color = color_state(crop_cv2(frame, tuple(best["box_norm"])))
        rows.append({"video": video, "fi": fi, "no_light": g.get("no_light", False),
                     "gt_green": "green" in gcolors, "gov_boxes": gov_boxes,
                     "best_cand": best, "best_conf": best_conf, "best_color": best_color,
                     "had_cands": True})
    return rows


def metrics_for_tau(rows, tau, exclude_video=None):
    """给定全局 τ 算指标。exclude_video 用于扣05主标尺。

    - 选中 = best_cand 非 None 且 best_conf>=tau(=select_gtfree gate 通过)。
    - 选灯精度: governing 帧选中且 IoU>=0.3。
    - 正确弃权: 无灯帧未选中(conf<tau)。
    - 误绿(扣 exclude_video): 选中绿但帧非绿(含无灯帧=弃权门要治的主失败模式)。
    - 漏绿: 真绿帧未选中绿(R4 硬约束)。
    """
    def _sel_color(r):
        return r["best_color"] if (r["best_cand"] is not None and r["best_conf"] >= tau) else None
    gov_frames = [r for r in rows if r["gov_boxes"]]
    nol_frames = [r for r in rows if r["no_light"]]
    green_frames = [r for r in rows if r["gt_green"]]
    hit = 0
    for r in gov_frames:
        if r["best_cand"] is not None and r["best_conf"] >= tau:
            if any(iou(r["best_cand"]["box_norm"], gb) >= 0.3 for gb in r["gov_boxes"]):
                hit += 1
    sel_prec = hit / len(gov_frames) if gov_frames else None
    rej = sum(1 for r in nol_frames if r["best_cand"] is None or r["best_conf"] < tau)
    rej_rate = rej / len(nol_frames) if nol_frames else None
    fg = sum(1 for r in rows if _sel_color(r) == "green" and not r["gt_green"]
             and (exclude_video is None or r["video"] != exclude_video))
    # 分母对齐 canonical: 候选帧(有候选, 含弃权) + 无候选但真绿帧(漏绿计入); 无候选非绿帧不计入(canonical 同口径)
    n_eval = sum(1 for r in rows if (r.get("had_cands") or r["gt_green"])
                 and (exclude_video is None or r["video"] != exclude_video))
    fg_rate = fg / n_eval if n_eval else 0.0
    miss = sum(1 for r in green_frames if not (_sel_color(r) == "green"))
    return {"sel_prec": sel_prec, "rej_rate": rej_rate, "n_nol": len(nol_frames),
            "fg": fg, "n_eval": n_eval, "fg_rate": fg_rate, "miss": miss, "n_green": len(green_frames)}


CACHE = ROOT / "models" / "governing_disc"


def _fold_sub(gt, V):
    return {"schema": gt.get("schema"), "frames": [f for f in gt["frames"] if f["video"] != V]}


def _load_or_train(V, sub, seed):
    """LOVO 单折模型: 缓存到磁盘, 断点续跑(SIGKILL 安全)。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"fold_{V}_seed{seed}.pt"
    if path.exists():
        m = gd.GoverningDiscNet()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m
    pos, neg_b, _ = gd.build_crop_dataset(sub, use_negative_a=False)
    m = gd.train_model(pos, neg_b, seed=seed)
    torch.save(m.state_dict(), path)
    return m


def _tau_for_fold(model, gt, train_videos, V, seed):
    """R3: 单折 τ 由训练视频 recommend_tau 定, 缓存。"""
    path = CACHE / f"tau_{V}_seed{seed}.json"
    if path.exists():
        return float(json.load(open(path, encoding="utf-8"))["tau"])
    t, _ = recommend_tau(model, gt, train_videos)
    path.write_text(json.dumps({"tau": float(t)}), encoding="utf-8")
    return float(t)


def _eval_cached(V, frames, model, det, yolo, gw, seed):
    """评测行缓存(按 V+seed+gw), 断点续跑。S1: 内部不弃权, 门交 metrics 按 τ 施加。"""
    gwi = int(round(gw * 10))
    path = CACHE / f"rows_{V}_seed{seed}_gw{gwi}_s1.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = eval_video(V, frames, model, det, yolo, governing_weight=gw)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def _global_tau(gt, models, videos, seed):
    """R3: 每折用训练视频定 τ(recommend_tau, 带缓存), 取 median 作单一全局 τ 施于测试折。"""
    tau_per = []
    for V in videos:
        train_videos = [v for v in videos if v != V]
        tau_per.append(_tau_for_fold(models[V], gt, train_videos, V, seed))
    return float(np.median(tau_per))


def _f(x):
    return x if x is not None else 0.0


def _fmt_ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "N/A"
    return f"{float(np.mean(vals))*100:.1f}%±{float(np.std(vals))*100:.1f}pp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None, help="逗号分隔限制视频(冒烟用); 空=全部")
    ap.add_argument("--seeds", default="0", help="逗号分隔 seed 列表(≥5 全量)")
    ap.add_argument("--governing-weight", type=float, default=0.3,
                    help="A1: 显式定值>0(重排功能不再空转); 锚模式强制0")
    ap.add_argument("--tau", type=float, default=None,
                    help="全局τ; None=训练折median(R3, B2 修复)")
    ap.add_argument("--anchor", action="store_true",
                    help="A2: 无模型纯base路径(τ=0)复现canonical R1扣05=2.19%/漏绿=80")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    gt = json.load(open(GT, encoding="utf-8"))
    det = TrafficLightDetector(gd._cfg_tl(), verbose=False)
    yolo = gd._lazy_yolo()
    by_video = _video_frames(gt)
    videos = [v for v in sorted(by_video) if (args.videos is None or v in args.videos.split(","))]

    # ===== A2 基线锚(快, 无训练) =====
    if args.anchor:
        all_rows = []
        for V in videos:
            rows = eval_video(V, by_video[V], None, det, yolo, governing_weight=0.0)
            all_rows.extend(rows)
        mg = metrics_for_tau(all_rows, 0.0, exclude_video="违章05")
        print("\n=== A2 基线锚(纯 base 路径, τ=0, 无判别器) ===")
        print(f"扣05误绿: {mg['fg']}/{mg['n_eval']} = {mg['fg_rate']*100:.2f}%  (目标 ≈2.19%)")
        print(f"漏绿:     {mg['miss']}                          (目标 = 80)")
        ok = abs(mg["fg_rate"] - 0.0219) < 0.01 and mg["miss"] == 80
        print(f"锚判定: {'PASS ✅ 评测台与 canonical 基线吻合' if ok else 'FAIL ❌ 评测台与基线脱钩, 数字不可信'}")
        return

    # ===== 方法: 多 seed LOVO 训练 + 训练折全局 τ (B2) =====
    gw = args.governing_weight
    per_seed = {}
    for seed in seeds:
        models = {}
        for V in videos:
            models[V] = _load_or_train(V, _fold_sub(gt, V), seed)
        gtau = args.tau if args.tau is not None else _global_tau(gt, models, videos, seed)
        print(f"[seed {seed}] global_tau={gtau:.2f} GW={gw}", flush=True)
        rows_per_fold = {}
        all_rows = []
        for V in videos:
            rows = _eval_cached(V, by_video[V], models[V], det, yolo, gw, seed)
            rows_per_fold[V] = rows
            all_rows.extend(rows)
            print(f"  [seed {seed}][{V}] GT帧={len(rows)}", flush=True)
        mg = metrics_for_tau(all_rows, gtau, exclude_video="违章05")
        per_seed[seed] = {"tau": gtau, "rows_per_fold": rows_per_fold,
                          "all_rows": all_rows, "mg": mg}
        print(f"[seed {seed}] 误绿(扣05)={mg['fg_rate']*100:.2f}% 漏绿={mg['miss']} "
              f"sel_prec={_f(mg['sel_prec'])*100:.1f}% gate={'PASS' if mg['miss']<=80 else 'FAIL'}",
              flush=True)

    _write_report(per_seed, gw, seeds, videos)
    print(f"\n[out] {REPORT}")


def _write_report(per_seed, gw, seeds, videos):
    L = ["# 选灯质量评分(§3.3, cc 复核修订后, ≥%d seed 全量 LOVO)\n" % len(seeds),
         f"> 评测台驱动 `select_gtfree` 上线路径(B1); headline τ = 训练折全局(median, R3); GW={gw}。\n",
         "> **S1 已修**: `eval_video` 内部不弃权(governing_threshold=0.0), 弃权门交 `metrics_for_tau` 按每个扫描 τ 施加 → τ 敏感性曲线全段诚实, headline 不变。\n",
         "> **S2 注明**: τ 由 `recommend_tau` 在**训练折 in-sample** 重打分、以 crop 级 pos/neg **F1** 作代理选取(非下游漏绿/误绿目标); R3 核心(测试视频不碰 τ)已满足, 解读 LOVO 泛化数字时请对 τ 乐观度打折。\n"]
    # 种子级 headline 聚合
    fg_rates = [per_seed[s]["mg"]["fg_rate"] for s in seeds]
    misses = [per_seed[s]["mg"]["miss"] for s in seeds]
    sps = [_f(per_seed[s]["mg"]["sel_prec"]) for s in seeds]
    # 正确弃权率: 复用 metrics_for_tau 的 no_light 帧正确口径(分子分母均锁无灯帧),
    # 不自行用全帧弃权数 / 无灯帧数(会 >100%, 失实)。
    rejs = [per_seed[s]["mg"]["rej_rate"] for s in seeds]
    L.append("## 主标尺(全399帧, 扣05, ≥%d seed 聚合 mean±std + worst-seed)\n" % len(seeds))
    L.append(f"- 误绿(扣05): **{np.mean(fg_rates)*100:.2f}%** ±{np.std(fg_rates)*100:.2f}pp "
             f"(worst-seed最差={max(fg_rates)*100:.2f}%)")
    L.append(f"- 漏绿: **{np.mean(misses):.0f}** ±{np.std(misses):.1f} (worst-seed={max(misses)}) "
             f"硬约束≤80 → **{'PASS' if max(misses)<=80 else 'FAIL'}**")
    L.append(f"- 选灯精度: {np.mean(sps)*100:.1f}% ±{np.std(sps)*100:.1f}pp")
    L.append(f"- 正确弃权率(整体): {_fmt_ms(rejs)}")
    # worst-seed(min)
    ws_fg = max(seeds, key=lambda s: per_seed[s]["mg"]["fg_rate"])
    ws_miss = max(seeds, key=lambda s: per_seed[s]["mg"]["miss"])
    L.append(f"\n## worst-seed(min) — 最差种子")
    L.append(f"- 按误绿最差: seed **{ws_fg}** → 误绿={per_seed[ws_fg]['mg']['fg_rate']*100:.2f}%")
    L.append(f"- 按漏绿最差: seed **{ws_miss}** → 漏绿={per_seed[ws_miss]['mg']['miss']} "
             f"(gate={'PASS' if per_seed[ws_miss]['mg']['miss']<=80 else 'FAIL'})")
    L.append(f"- 各 seed 全局 τ: " + ", ".join(f"{s}={per_seed[s]['tau']:.2f}" for s in seeds))
    # 11 视频分解 mean±std
    L.append("\n## 11 视频分解(扣05 误绿 / 漏绿 / 选灯精度, mean±std across seeds)\n")
    L.append("| 视频 | 误绿(扣05) mean±std | 漏绿 mean±std | 选灯精度 mean±std |")
    L.append("|---|---|---|---|")
    for V in videos:
        fgs, ms, spv = [], [], []
        for s in seeds:
            m = metrics_for_tau(per_seed[s]["rows_per_fold"][V], per_seed[s]["tau"], exclude_video="违章05")
            fgs.append(m["fg_rate"]); ms.append(m["miss"]); spv.append(_f(m["sel_prec"]))
        L.append(f"| {V} | {_fmt_ms(fgs)} | {np.mean(ms):.0f}±{np.std(ms):.1f} | {_fmt_ms(spv)} |")
    # worst-video
    def _vp(V, key):
        return np.mean([metrics_for_tau(per_seed[s]["rows_per_fold"][V], per_seed[s]["tau"],
                                         exclude_video="违章05")[key] for s in seeds])
    wv_fg = max(videos, key=lambda V: _vp(V, "fg_rate"))
    wv_miss = max(videos, key=lambda V: _vp(V, "miss"))
    L.append(f"\n## worst-video — 最差视频")
    L.append(f"- 按误绿最差: **{wv_fg}** → 误绿 mean={_vp(wv_fg, 'fg_rate')*100:.2f}%")
    L.append(f"- 按漏绿最差: **{wv_miss}** → 漏绿 mean={_vp(wv_miss, 'miss'):.0f}")
    # R2: 03 单列 + 06/11 N/A
    L.append("\n## 无灯帧处理(R2)")
    v03 = per_seed[seeds[0]]["rows_per_fold"].get("违章03", [])
    if v03:
        m03 = metrics_for_tau(v03, per_seed[seeds[0]]["tau"])
        L.append(f"- 03(单列, 不混 mean): 无灯帧={m03['n_nol']} 正确弃权率={_f(m03['rej_rate'])*100:.1f}%")
    for v in ("违章06", "违章11"):
        nol = [r for s in seeds for r in per_seed[s]["all_rows"] if r["video"] == v and r["no_light"]]
        L.append(f"- {v}: 正确弃权率 = N/A (零无灯帧)" if not nol else f"- {v}: 无灯帧={len(nol)}")
    # τ 敏感性(S1-honest): 用单 seed 口径(与主标尺一致, 非 5×seed 拼接), 否则漏绿被放大 5× 误导 gate 判定
    all_pool = per_seed[seeds[0]]["all_rows"]
    L.append("\n## τ 敏感性(S1 修复后, 内部不弃权, 门按扫描 τ 施加; 仅展示)\n")
    L.append(f"{'τ':>4} {'选灯精度':>8} {'弃权率':>8} {'误绿(扣05)':>11} {'漏绿':>5}")
    for tau in TAU_GRID:
        m = metrics_for_tau(all_pool, tau, exclude_video="违章05")
        L.append(f"{tau:>4.1f} {(_f(m['sel_prec'])*100):>7.1f}% {(_f(m['rej_rate'])*100):>7.1f}% "
                 f"{m['fg_rate']*100:>10.2f}% {m['miss']:>5}")
    L.append("\n- 漏绿>80 判 gate FAIL, 净回退不接线(红线)。")
    L.append("- 03 单列不混 mean; 06/11 零无灯帧, 正确弃权率记 N/A(不充 0/100, R2)。")
    L.append("- S2: τ 为训练折 in-sample F1 代理, 报告口径 τ 点不取测试集 argmin(B2/R3)。")
    REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
