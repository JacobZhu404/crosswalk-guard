"""灯态快速评测 (Phase-1 红绿灯识别, prior 锁定版)。

直接读 datasets/frames/ 预抽帧(不重新解视频) -> 按 configs/light_priors.json 的
per-video prior 锁定行人信号 -> 逐帧跑检测器 -> 与 datasets/gt/light_states.csv 对比。

输出:
  - 控制台: 逐视频 accuracy / 总体 accuracy / 混淆矩阵 / mismatch 数
  - data/output/light_eval/pred_{video}.csv   逐帧预测(供裁图/回溯)
  - data/output/light_eval/mismatch_all.csv   所有不一致帧(待确认清单)
  - (可选) --regression FILE.csv  只校验固化回归用例(video,t_sec,expected_state), 报 PASS/FAIL

用法:
  python scripts/eval_light_fast.py                      # 默认 02/03/04
  python scripts/eval_light_fast.py --videos 违章02 违章04
  python scripts/eval_light_fast.py --regression datasets/gt/light_regression.csv
"""
import os
import sys
import csv
import json
import argparse
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np
from redlight.infrastructure.config import load_config
from redlight.models.traffic_light import TrafficLightDetector


def load_manifest(frames_dir):
    """返回 {(video, frame_idx): timestamp} 与帧尺寸。"""
    mp = os.path.join(frames_dir, "manifest.csv")
    table = {}
    if os.path.exists(mp):
        with open(mp, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    table[(r["video"], int(r["frame_idx"]))] = float(r["timestamp"])
                except (KeyError, ValueError):
                    pass
    return table


def load_gt(gt_path):
    """返回 {video: [(start, end, state, confidence), ...]} (按 start 升序)。"""
    g = {}
    with open(gt_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = r["video"]
            g.setdefault(v, []).append(
                (float(r["start_s"]), float(r["end_s"]), r["state"], r.get("confidence", "confirmed"))
            )
    for v in g:
        g[v].sort(key=lambda x: x[0])
    return g


def load_priors(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    # 兼容 [cx, cy, roi] 或 {"cx":..,"cy":..,"roi":..}
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = (float(v[0]), float(v[1]), int(v[2]) if len(v) > 2 else 160)
        elif isinstance(v, dict):
            out[k] = (float(v["cx"]), float(v["cy"]), int(v.get("roi", 160)))
    return out


def gt_state_at(segs, t):
    for s, e, st, conf in segs:
        if s <= t <= e:
            return st, conf
    # 兜底: 取最近段
    if segs:
        return segs[-1][2], segs[-1][3]
    return "unknown", "confirmed"


def robust_imread(path):
    with open(path, "rb") as f:
        b = f.read()
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def eval_video(video, frames_dir, manifest, gt_segs, prior, cfg, sample_every=1):
    vdir = os.path.join(frames_dir, video)
    files = sorted(glob.glob(os.path.join(vdir, "frame_*.jpg")))
    if not files:
        return None
    det = TrafficLightDetector(cfg, verbose=False)
    if prior is not None:
        cx, cy, roi = prior
        det.signal_prior = (cx, cy)
        det.prior_roi_px = roi
        print(f"  [{video}] prior=({cx},{cy}) roi={roi}")
    else:
        print(f"  [{video}] 无 prior (自由选灯)")

    rows = []
    for i, fp in enumerate(files):
        idx = int(os.path.basename(fp)[6:-4])  # frame_000123.jpg -> 123
        t = manifest.get((video, idx))
        if t is None:
            # 从文件名推断: 抽帧 step 已知不可靠, 用相邻 manifest 估计; 这里退化为 0
            t = 0.0
        frame = robust_imread(fp)
        if frame is None:
            continue
        res = det.detect(frame)
        state = res.get("state", "unknown")
        g_px = res.get("g_px", 0)
        r_px = res.get("r_px", 0)
        rows.append((t, idx, state, g_px, r_px))
        if i % 50 == 0:
            sys.stderr.write(f"    {video} frame {i}/{len(files)} t={t:.1f}s {state}\n")

    # 逐帧对 GT
    pred_records = []
    mismatches = []
    correct = 0
    total_confirmed = 0
    conf_mat = {}
    for t, idx, state, g_px, r_px in rows:
        gt_st, gt_conf = gt_state_at(gt_segs, t)
        # 混淆矩阵(计入全部)
        conf_mat.setdefault(gt_st, {}).setdefault(state, 0)
        conf_mat[gt_st][state] += 1
        is_confirmed = (gt_conf == "confirmed")
        pred_records.append({
            "video": video, "t_sec": round(t, 2), "frame_idx": idx,
            "pred": state, "gt": gt_st, "gt_conf": gt_conf,
            "g_px": g_px, "r_px": r_px,
        })
        if is_confirmed:
            total_confirmed += 1
            if state == gt_st:
                correct += 1
            else:
                mismatches.append(pred_records[-1])
    acc = (correct / total_confirmed) if total_confirmed else None
    return {
        "video": video, "acc": acc, "total_confirmed": total_confirmed,
        "correct": correct, "mismatches": mismatches, "conf_mat": conf_mat,
        "pred_records": pred_records, "prior": prior,
    }


def compress_timeline(records):
    """把逐帧预测压成连续段: 'red[0.0-20.9] green[20.9-85.0] ...'。"""
    segs = []
    for r in records:
        st = r["pred"]
        t = r["t_sec"]
        if segs and segs[-1][0] == st:
            segs[-1][2] = t
        else:
            segs.append([st, t, t])
    return " ".join(f"{s}[{a:.1f}-{b:.1f}]" for s, a, b in segs)


def boundary_split(mismatches, gt_segs, tol=2.0):
    """拆分 mismatch: 落在 GT 段边界 ±tol 内(过渡) vs 段内部(真实问题)。"""
    bounds = []
    for i in range(len(gt_segs) - 1):
        bounds.append((gt_segs[i][1] + gt_segs[i + 1][0]) / 2.0)  # 段间中点
    interior, boundary = [], []
    for m in mismatches:
        t = m["t_sec"]
        near = any(abs(t - b) <= tol for b in bounds)
        (boundary if near else interior).append(m)
    return interior, boundary


def main():
    ap = argparse.ArgumentParser(description="灯态快速评测 (prior 锁定)")
    ap.add_argument("--videos", nargs="*", default=None, help="默认 02/03/04")
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--priors", default=os.path.join(ROOT, "configs", "light_priors.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "output", "light_eval"))
    ap.add_argument("--regression", default=None, help="固化回归用例 CSV(video,t_sec,expected_state)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    manifest = load_manifest(args.frames_dir)
    gt = load_gt(args.gt)
    priors = load_priors(args.priors)

    videos = args.videos or ["违章02", "违章03", "违章04"]
    videos = [v for v in videos if v in gt]

    print(f"=== 灯态快速评测: {videos} ===")
    results = []
    for v in videos:
        prior = priors.get(v)
        r = eval_video(v, args.frames_dir, manifest, gt[v], prior, cfg)
        if r:
            results.append(r)
            acc = r["acc"]
            acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A (tentative, 仅看预测时间线)"
            print(f"  -> {v}: acc={acc_str} ({r['correct']}/{r['total_confirmed']}) "
                  f"mismatch={len(r['mismatches'])}")
            print(f"     预测时间线: {compress_timeline(r['pred_records'])}")

    # 总体(仅 confirmed)
    tot_c = sum(r["correct"] for r in results)
    tot_n = sum(r["total_confirmed"] for r in results)
    print(f"\n=== 总体 accuracy (confirmed only) = {tot_c/tot_n*100:.1f}% ({tot_c}/{tot_n}) ===")

    # mismatch 边界/段内拆分 (过渡假象 vs 真实问题)
    print("\n=== mismatch 拆解 (边界±2.0s 算过渡, 否则段内真实问题) ===")
    for r in results:
        interior, boundary = boundary_split(r["mismatches"], gt[r["video"]])
        print(f"  {r['video']}: 段内真实问题={len(interior)}  边界过渡={len(boundary)}")
        if interior:
            # 段内错帧的时间分布(取前若干个代表点)
            samp = interior[:12]
            print("     段内错帧代表: " + ", ".join(
                f"t={m['t_sec']:.1f}(预{m['pred']}/真{m['gt']})" for m in samp))

    # 混淆矩阵
    print("\n混淆矩阵 (行=GT, 列=预测):")
    states = sorted({s for r in results for row in r["conf_mat"].values() for s in row})
    print("GT\\PRED".ljust(10) + "".join(s.rjust(10) for s in states))
    for gt_s in states:
        line = gt_s.ljust(10)
        for p_s in states:
            c = sum(r["conf_mat"].get(gt_s, {}).get(p_s, 0) for r in results)
            line += str(c).rjust(10)
        print(line)

    # 落盘逐帧预测 + mismatch
    all_mm = []
    for r in results:
        pred_csv = os.path.join(args.out, f"pred_{r['video']}.csv")
        with open(pred_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video", "t_sec", "frame_idx", "pred", "gt", "gt_conf", "g_px", "r_px"])
            w.writeheader()
            w.writerows(r["pred_records"])
        for m in r["mismatches"]:
            all_mm.append(m)

    mm_csv = os.path.join(args.out, "mismatch_all.csv")
    with open(mm_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "t_sec", "frame_idx", "pred", "gt", "gt_conf", "g_px", "r_px"])
        w.writeheader()
        w.writerows(all_mm)
    print(f"\n逐帧预测: {args.out}/pred_*.csv")
    print(f"mismatch 清单: {mm_csv} ({len(all_mm)} 条)")

    # 回归校验
    if args.regression:
        print(f"\n=== 回归校验: {args.regression} ===")
        reg = {}
        with open(args.regression, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                reg.setdefault(row["video"], []).append((float(row["t_sec"]), row["expected_state"]))
        pred_by_video = {r["video"]: r["pred_records"] for r in results}
        fails = 0
        for v, items in reg.items():
            preds = pred_by_video.get(v, [])
            for t_exp, exp in items:
                # 最近帧
                best = min(preds, key=lambda p: abs(p["t_sec"] - t_exp)) if preds else None
                if best is None:
                    print(f"  [FAIL] {v} t={t_exp}: 无预测"); fails += 1; continue
                ok = (best["pred"] == exp)
                mark = "OK  " if ok else "FAIL"
                if not ok:
                    fails += 1
                print(f"  [{mark}] {v} t={t_exp} 期望={exp} 实际={best['pred']} (帧 t={best['t_sec']})")
        print(f"回归校验: {'全部通过 ✅' if fails == 0 else f'{fails} 条回退 ❌'}")


if __name__ == "__main__":
    main()
