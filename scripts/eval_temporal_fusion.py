#!/usr/bin/env python3
"""时序融合独立评测 (eval-b): 评估第②层 fuse_light 质量。

与 eval_light_fast.py (端到端检测器评测) 的区别:
  - eval-b 使用 TrafficLightDetector.observe() 提取单帧观测(①层),
    再经 fuse_light() 进行时序融合(②层), 独立评估②层输出质量。
  - 不使用时序平滑/迟滞/闪烁判定的 detect()。

用法:
  python scripts/eval_temporal_fusion.py
  python scripts/eval_temporal_fusion.py --videos 违章02 违章03
  python scripts/eval_temporal_fusion.py --unknown-hold 12
"""
import os
import sys
import csv
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.traffic_light import TrafficLightDetector
from redlight.evaluation.gt_lookup import load_light_state_csv, state_at
from redlight.evaluation.frame_dataset import FrameDataset
from redlight.evaluation.video_sampler import VideoSampler
from redlight.evaluation.metrics import light_state_metrics
from redlight.pipeline.temporal_fusion import fuse_light


def load_priors(path):
    """加载 per-video 先验(prior 本身不直接用于 observe, 但保留接口一致性)。"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = (float(v[0]), float(v[1]), int(v[2]) if len(v) > 2 else 160)
        elif isinstance(v, dict):
            out[k] = (float(v["cx"]), float(v["cy"]), int(v.get("roi", 160)))
    return out


def eval_video(video, dataset, gt_segs, cfg, fuse_kwargs, prior=None):
    """对单个视频运行 eval-b: observe -> fuse_light -> 逐帧对比 GT."""
    frames = list(dataset.iter_video(video))
    if not frames:
        return None

    # ①层: 逐帧观测(无状态, 单帧主色)
    det = TrafficLightDetector(cfg, verbose=False)
    if prior is not None:
        det.signal_prior = (float(prior[0]), float(prior[1]))
        det.prior_roi_px = int(prior[2]) if len(prior) > 2 else 160
    observations = []   # [(ts, obs, conf), ...]
    timestamps = []     # [ts, ...]
    frame_indices = []  # [frame_idx, ...]

    for i, (idx, ts, frame) in enumerate(frames):
        if frame is None:
            continue
        res = det.observe(frame)
        obs = res.get("obs", "off")  # green|red|off|None
        conf = res.get("conf", 0.0)
        observations.append((ts, obs, conf))
        timestamps.append(ts)
        frame_indices.append(idx)
        if i % 50 == 0:
            sys.stderr.write(f"    {video} frame {i}/{len(frames)} t={ts:.1f}s obs={obs}\n")

    if not observations:
        return None

    # ②层: 时序融合
    pred_segs = fuse_light(observations, **fuse_kwargs)
    # 转为 state_at 可用的 tuple 格式: (start, end, state, meta)
    pred_segs_typed = [
        (s["start_s"], s["end_s"], s["state"], str(s.get("conf", 0.0)))
        for s in pred_segs
    ]

    # 逐帧对比
    pred_states = []
    gt_states = []
    confirmed_pred = []
    confirmed_gt = []
    pred_records = []
    mismatches = []
    correct = 0
    total_confirmed = 0

    for ts, idx in zip(timestamps, frame_indices):
        pred_state, pred_conf = state_at(pred_segs_typed, ts)
        gt_state, gt_conf = state_at(gt_segs, ts)
        is_confirmed = (gt_conf == "confirmed")
        # conf 数值化(供 make_light_gallery 画廊置信度色块; pred_conf 原始可能是段 meta 字符串)
        try:
            conf_num = float(pred_conf)
        except (TypeError, ValueError):
            conf_num = 0.0
        pred_records.append({
            "video": video, "t_sec": round(ts, 2), "frame_idx": idx,
            "pred": pred_state, "gt": gt_state, "gt_conf": gt_conf,
            "pred_conf": pred_conf, "conf": conf_num,
        })
        pred_states.append(pred_state)
        gt_states.append(gt_state)
        if is_confirmed:
            total_confirmed += 1
            confirmed_pred.append(pred_state)
            confirmed_gt.append(gt_state)
            if pred_state == gt_state:
                correct += 1
            else:
                mismatches.append(pred_records[-1])

    # 全量 metrics (含 tentative)
    full_metrics = light_state_metrics(pred_states, gt_states)
    confirmed_metrics = light_state_metrics(confirmed_pred, confirmed_gt) if confirmed_pred else None

    acc = (correct / total_confirmed) if total_confirmed else None
    return {
        "video": video,
        "acc": acc,
        "total_confirmed": total_confirmed,
        "correct": correct,
        "mismatches": mismatches,
        "pred_records": pred_records,
        "pred_states": pred_states,
        "gt_states": gt_states,
        "full_metrics": full_metrics,
        "confirmed_metrics": confirmed_metrics,
        "pred_segments": pred_segs,
    }


def compress_timeline(records):
    """把逐帧预测压成连续段: 'red[0.0-20.9] green[20.9-85.0] ...'."""
    segs = []
    for r in records:
        st = r["pred"]
        t = r["t_sec"]
        if segs and segs[-1][0] == st:
            segs[-1][2] = t
        else:
            segs.append([st, t, t])
    return " ".join(f"{s}[{a:.1f}-{b:.1f}]" for s, a, b in segs)


def compress_fused_segments(segs):
    """把 fuse_light 产出的段压缩为可读时间线。"""
    return " ".join(
        f"{s['state']}[{s['start_s']:.1f}-{s['end_s']:.1f}]"
        for s in segs
    )


def boundary_split(mismatches, gt_segs, tol=2.0):
    """拆分 mismatch: 落在 GT 段边界 ±tol 内(过渡) vs 段内部(真实问题)。"""
    bounds = []
    for i in range(len(gt_segs) - 1):
        bounds.append((gt_segs[i][1] + gt_segs[i + 1][0]) / 2.0)
    interior, boundary = [], []
    for m in mismatches:
        t = m["t_sec"]
        near = any(abs(t - b) <= tol for b in bounds)
        (boundary if near else interior).append(m)
    return interior, boundary


def main():
    ap = argparse.ArgumentParser(description="时序融合独立评测 (eval-b)")
    ap.add_argument("--videos", nargs="*", default=None, help="默认 GT 中所有视频")
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--priors", default=os.path.join(ROOT, "configs", "light_priors.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "output", "temporal_fusion_eval"))
    ap.add_argument("--gallery-out", default=None,
                     help="同步写 pred_*.csv+mismatch_all.csv 到此目录(供 make_light_gallery 直接消费, 默认 light_eval)")
    # fuse_light 参数覆盖
    ap.add_argument("--window", type=int, default=None, help="覆盖 smoothing_window")
    ap.add_argument("--hysteresis", type=float, default=None, help="覆盖迟滞阈值")
    ap.add_argument("--flicker-toggle", type=int, default=None, help="覆盖闪烁跳变阈值")
    ap.add_argument("--unknown-hold", type=int, default=8, help="unknown 保持阈值(默认8)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    dataset = FrameDataset(args.frames_dir)
    gt = load_light_state_csv(args.gt)

    # fuse_light 参数: 优先命令行, 其次 config
    tl = getattr(cfg, "traffic_light", None)
    fuse_kwargs = {
        "window": args.window if args.window is not None else int(getattr(tl, "smoothing_window", 24)),
        "hysteresis": args.hysteresis if args.hysteresis is not None else float(getattr(tl, "hysteresis", 0.68)),
        "flicker_toggle": args.flicker_toggle if args.flicker_toggle is not None else int(getattr(tl, "flicker_toggle_count", 4)),
        "unknown_hold": args.unknown_hold,
    }

    priors = load_priors(args.priors)

    videos = args.videos or list(gt.keys())
    videos = [v for v in videos if v in gt]

    print(f"=== 时序融合评测 (eval-b): fuse_light({fuse_kwargs}) ===")
    print(f"视频: {videos}")
    results = []
    for v in videos:
        r = eval_video(v, dataset, gt[v], cfg, fuse_kwargs, prior=priors.get(v))
        if r:
            results.append(r)
            acc = r["acc"]
            acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A"
            print(f"\n  [{v}] accuracy(confirmed)={acc_str} ({r['correct']}/{r['total_confirmed']}) "
                  f"mismatch={len(r['mismatches'])}")
            print(f"     融合时间线: {compress_fused_segments(r['pred_segments'])}")
            print(f"     逐帧时间线: {compress_timeline(r['pred_records'])}")
            fm = r["full_metrics"]
            print(f"     全量 macro_f1={fm['macro_f1']:.3f} accuracy={fm['accuracy']:.3f} n={fm['n']}")
            cm = r["confirmed_metrics"]
            if cm:
                print(f"     confirmed macro_f1={cm['macro_f1']:.3f} accuracy={cm['accuracy']:.3f} n={cm['n']}")

    if not results:
        print("无评测结果")
        return

    # 总体 confirmed accuracy
    tot_c = sum(r["correct"] for r in results)
    tot_n = sum(r["total_confirmed"] for r in results)
    print(f"\n=== 总体 accuracy (confirmed only) = {tot_c/tot_n*100:.1f}% ({tot_c}/{tot_n}) ===")

    # 总体全量 metrics
    all_pred = []
    all_gt = []
    for r in results:
        all_pred.extend(r["pred_states"])
        all_gt.extend(r["gt_states"])
    agg = light_state_metrics(all_pred, all_gt)
    print(f"=== 总体全量 macro_f1={agg['macro_f1']:.3f} accuracy={agg['accuracy']:.3f} n={agg['n']} ===")

    # per-class 汇总
    print("\n=== 各类别汇总 (全量) ===")
    classes = sorted(agg["per_class"].keys())
    print("class".ljust(12) + "precision".rjust(12) + "recall".rjust(12) + "f1".rjust(12) + "support".rjust(12))
    for cls in classes:
        d = agg["per_class"][cls]
        print(f"{cls:<12}{d['precision']:>12.3f}{d['recall']:>12.3f}{d['f1']:>12.3f}{d['support']:>12}")

    # mismatch 边界/段内拆分
    print("\n=== mismatch 拆解 (边界±2.0s 算过渡, 否则段内真实问题) ===")
    for r in results:
        interior, boundary = boundary_split(r["mismatches"], gt[r["video"]])
        print(f"  {r['video']}: 段内真实问题={len(interior)}  边界过渡={len(boundary)}")
        if interior:
            samp = interior[:12]
            print("     段内错帧代表: " + ", ".join(
                f"t={m['t_sec']:.1f}(预{m['pred']}/真{m['gt']})" for m in samp))

    # 混淆矩阵
    print("\n混淆矩阵 (行=GT, 列=预测):")
    all_states = set()
    for r in results:
        for rec in r["pred_records"]:
            all_states.add(rec["pred"])
            all_states.add(rec["gt"])
    states = sorted(all_states)
    print("GT\\PRED".ljust(10) + "".join(s.rjust(10) for s in states))
    conf_mat = {gs: {ps: 0 for ps in states} for gs in states}
    for r in results:
        for rec in r["pred_records"]:
            conf_mat[rec["gt"]][rec["pred"]] += 1
    for gs in states:
        line = gs.ljust(10)
        for ps in states:
            line += str(conf_mat[gs][ps]).rjust(10)
        print(line)

    # 落盘: 逐帧预测 + mismatch (字段对齐 make_light_gallery 画廊: 含 conf 数值列)
    _FIELDS = ["video", "t_sec", "frame_idx", "pred", "gt", "gt_conf", "pred_conf", "conf"]
    all_mm = []
    for r in results:
        pred_csv = os.path.join(args.out, f"pred_{r['video']}.csv")
        with open(pred_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            w.writeheader()
            w.writerows(r["pred_records"])
        for m in r["mismatches"]:
            all_mm.append(m)

    mm_csv = os.path.join(args.out, "mismatch_all.csv")
    with open(mm_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(all_mm)
    print(f"\n逐帧预测: {args.out}/pred_*.csv")
    print(f"mismatch 清单: {mm_csv} ({len(all_mm)} 条)")

    # 可选: 同步写一份到 light_eval/ 供 make_light_gallery 直接消费(完整交互画廊)
    if args.gallery_out:
        os.makedirs(args.gallery_out, exist_ok=True)
        for r in results:
            with open(os.path.join(args.gallery_out, f"pred_{r['video']}.csv"), "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDS)
                w.writeheader()
                w.writerows(r["pred_records"])
        with open(os.path.join(args.gallery_out, "mismatch_all.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            w.writeheader()
            w.writerows(all_mm)
        print(f"画廊数据(同步): {args.gallery_out}/pred_*.csv + mismatch_all.csv")

    # 落盘: fuse_light 产出的 segments (JSON)
    segments_json = os.path.join(args.out, "fused_segments.json")
    with open(segments_json, "w", encoding="utf-8") as f:
        json.dump({r["video"]: r["pred_segments"] for r in results}, f, ensure_ascii=False, indent=2)
    print(f"融合段 JSON: {segments_json}")


if __name__ == "__main__":
    main()
