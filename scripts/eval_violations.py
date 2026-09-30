#!/usr/bin/env python3
"""端到端违章事件评测 (eval-e2e): 跑整条流水线 -> 违章事件, 对比 datasets/gt/events.csv。

区别于 eval-b(只评②层灯态) / eval-plate(只评车牌): 本脚本评**最终违章结论**
(灯态×静止×压线×判定的乘积), 用事件级重叠匹配算 P/R/F1, 并报覆盖率(暴露欠检)。

用法:
  python scripts/eval_violations.py                      # 跑全部有 GT 违章的视频(fresh)
  python scripts/eval_violations.py --videos 违章04 违章09
  python scripts/eval_violations.py --reuse              # 复用已有 run_*/violations.csv, 不重跑
  python scripts/eval_violations.py --preset balanced --min-overlap 0.5
"""
import os
import sys
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.crosswalk_v2 import CrosswalkDetectorV2
from redlight.pipeline.tracker import SENSITIVITY_PRESETS
from redlight.evaluation.violation_eval import (
    load_violation_gt, match_violation_events, aggregate,
    load_video_metadata, classify_false_positives,
    load_car_level_gt, match_cars, aggregate_car_level,
)


def _read_violations_csv(path):
    """从已有 violations.csv 读回事件(用于 --reuse)。"""
    events = []
    if not os.path.exists(path):
        return events
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                events.append({
                    "status": (row.get("status") or "").strip(),
                    "start_ts": float(row["start_ts"]),
                    "end_ts": float(row["end_ts"]),
                    "plate": (row.get("plate") or "").strip(),
                    "plates": [p.strip() for p in (row.get("plates") or "").split("|") if p.strip()],
                    "track_id": row.get("track_id", ""),
                    "light_state": (row.get("light_state") or "").strip(),
                })
            except (KeyError, ValueError):
                continue
    return events


def _pred_plates(confirmed_events):
    """从 confirmed 事件收集所有具名预测车牌(主牌 plate + 多牌 plates), 去空去重。"""
    out = set()
    for e in confirmed_events:
        if e.get("plate"):
            out.add(e["plate"].strip())
        for p in (e.get("plates") or []):
            if p and p.strip():
                out.add(p.strip())
    return out


def _run_pipeline(cfg, video, preset, out_root, detector_name="v11", occ_denom="mask", box_overlap=None):
    """跑一遍流水线, 返回事件 list。关标注视频加速(证据截图保留以回填车牌)。

    detector_name: v11(默认) | v2(Plan v6 Phase 1 四边形探针)
    occ_denom: mask(D2 默认) | box(车足迹压线, 推翻 D2)
    box_overlap: 当 occ_denom=box 时覆盖 preset 的 box_overlap 阈值(用于扫描)
    """
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        print(f"  [跳过] 找不到视频 {video_path}")
        return None
    cfg.output.annotated_video = False   # eval 不需要 114MB 标注视频
    out_dir = os.path.join(out_root, f"run_{video}")
    det = CrosswalkDetectorV2(cfg) if detector_name == "v2" else CrosswalkDetector(cfg)
    engine_kwargs = {}
    if occ_denom == "box":
        engine_kwargs["occ_denom"] = "box"
        if box_overlap is not None:
            # 覆盖 preset 阈值(扫描用); 不影响 mask 路径
            SENSITIVITY_PRESETS[preset]["box_overlap"] = float(box_overlap)
    return cli.run(cfg, video_path, out_dir, preset=preset,
                   crosswalk_detector=det, **engine_kwargs)


def _confirmed(events):
    return [e for e in events if e.get("status") == "confirmed"]


def _fmt_span(x, key_s="start_ts", key_e="end_ts"):
    return f"[{x[key_s]:.1f}-{x[key_e]:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None, help="默认: events.csv 里所有含违章的视频")
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--events", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--videos-csv", default=os.path.join(ROOT, "datasets", "gt", "videos.csv"),
                    help="视频元数据(含 has_violation), 用于纳入负例与判定负例")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--min-overlap", type=float, default=0.5, help="判为TP的最小时间重叠(秒)")
    ap.add_argument("--detector", default="v11", choices=["v11", "v2"],
                    help="v11=全宽水平带(基线) | v2=透视梯形(Plan v6 Phase 1 探针)")
    ap.add_argument("--occ-denom", default="mask", choices=["mask", "box"],
                    help="mask=占斑马线比例(D2) | box=车足迹压线(推翻 D2)")
    ap.add_argument("--box-overlap", type=float, default=None,
                    help="occ-denom=box 时覆盖预设阈值(扫描用)")
    ap.add_argument("--reuse", action="store_true",
                    help="复用已有 data/output/run_<video>_<preset>/violations.csv, 不重跑流水线")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "output", "eval_violations"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)
    gt = load_violation_gt(args.events)
    car_gt = load_car_level_gt(args.events)
    meta = load_video_metadata(args.videos_csv)

    videos = args.videos or sorted(meta.keys())
    print(f"=== 端到端违章评测 (eval-e2e): preset={args.preset} min_overlap={args.min_overlap}s "
          f"detector={args.detector} occ_denom={args.occ_denom}"
          f"{(' box_overlap='+str(args.box_overlap)) if args.occ_denom=='box' and args.box_overlap is not None else ''} ===")
    print(f"视频(全 {len(meta)} 个, 含负例 {sum(1 for v in meta if not meta[v])} 个): {videos}\n")

    results, rows, car_results = [], [], []
    for v in videos:
        gt_v = gt.get(v, [])
        is_neg = not meta.get(v, True)  # 未列于 videos.csv 默认当正例
        # 取事件
        if args.reuse:
            csv_path = os.path.join(ROOT, "data", "output", f"run_{v}_{args.preset}", "violations.csv")
            events = _read_violations_csv(csv_path)
            if not events and not os.path.exists(csv_path):
                print(f"  [跳过 {v}] 无 {csv_path} (先不加 --reuse 跑一遍)")
                continue
        else:
            events = _run_pipeline(cfg, v, args.preset, args.out,
                                   detector_name=args.detector, occ_denom=args.occ_denom,
                                   box_overlap=args.box_overlap)
            if events is None:
                continue

        conf = _confirmed(events)
        r = match_violation_events(conf, gt_v, min_overlap_s=args.min_overlap)
        cls = classify_false_positives(r, conf, gt_v, is_negative=is_neg, min_overlap_s=args.min_overlap)
        r["neg_count"] = cls["neg_count"]
        r["oow_count"] = cls["oow_count"]
        r["fragment_count"] = cls["fragment_count"]
        r["_cls"] = cls
        # 车级评测(D2 = Jacob 拍板"按车辆算"): 具名违章车 recall + 误罚
        car_r = match_cars(_pred_plates(conf), car_gt.get(v, {"violating": set(), "non_violating": set()}))
        r["_car"] = car_r
        car_results.append(car_r)
        results.append(r)
        rows.append((v, r, cls, is_neg))

        # 逐视频时间线
        pred_line = " ".join(f"{_fmt_span(e)}{'✓' if any(m['pred_idx']==i for m in r['matches']) else '✗FP'}"
                             for i, e in enumerate(conf)) or "(无confirmed事件)"
        gt_line = " ".join(f"[{g['start_s']:.0f}-{g['end_s']:.0f}]cov={r['gt_coverage'][i]:.2f}"
                           + ("" if i not in r["fn_gts"] else "✗漏")
                           for i, g in enumerate(gt_v)) or "(无GT违章)"
        tag = " [负例]" if is_neg else ""
        print(f"[{v}]{tag} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f} "
              f"(tp={r['tp']} fp={r['fp']} fn={r['fn']}) 覆盖={r['mean_coverage']:.2f} "
              f"车牌={r['plate_hits']}/{r['plate_total']} | 真误报={cls['neg_count']+cls['oow_count']} 碎片={cls['fragment_count']}")
        print(f"    预测confirmed: {pred_line}")
        print(f"    GT违章段:     {gt_line}")
        # 车级明细(D2): 具名违章车命中/漏/误罚
        cr = r["_car"]
        car_bits = f"车级: 命中{cr['car_tp']} 漏{cr['car_fn']} 误罚{cr['car_misfine']} 未登记{cr['car_unknown']}"
        if cr["hit"]:      car_bits += f" | ✓{cr['hit']}"
        if cr["missed"]:   car_bits += f" | ✗漏{cr['missed']}"
        if cr["misfined"]: car_bits += f" | ⚠误罚{cr['misfined']}"
        if cr["unknown"]:  car_bits += f" | ?未登记{cr['unknown']}"
        print(f"    {car_bits}")
        # FP 拆解明细
        for d in cls["detail"]:
            if d["category"] == "neg_true_fp":
                print(f"      [真误报·负例] {d['span']} 灯态={d['light_state'] or '?'}")
            elif d["category"] == "oow_true_fp":
                print(f"      [真误报·窗外] {d['span']} 灯态={d['light_state'] or '?'}")
            else:
                print(f"      [碎片] {d['span']} 重叠GT窗={d['gt_window']} 灯态={d['light_state'] or '?'}")

    if not results:
        print("无评测结果")
        return

    agg = aggregate(results)
    print(f"\n=== 总体事件级(头条 1:1): P={agg['precision']:.3f} R={agg['recall']:.3f} F1={agg['f1']:.3f} "
          f"(tp={agg['tp']} fp={agg['fp']} fn={agg['fn']}) ===")
    print(f"=== 命中违章段平均覆盖率={agg['mean_coverage']:.3f} "
          f"车牌命中={agg['plate_hits']}/{agg['plate_total']} ===")
    print(f"=== 拆解: 真误报={agg['true_fp_total']} (负例+窗外) | 碎片={agg['fragment_total']} "
          f"| 诊断 P(仅真误报)={agg['p_only_true_fp']:.3f} ===")

    # 车级汇总(D2 = Jacob 拍板"按车辆算")
    cagg = aggregate_car_level(car_results)
    print(f"\n=== 车级(D2 按车辆算): 具名违章车召回={cagg['car_recall_named']:.3f} "
          f"(命中{cagg['car_tp']}/漏{cagg['car_fn']}) | 具名精度={cagg['car_precision_named']:.3f} "
          f"| 误罚={cagg['misfine_total']} | 未登记={cagg['car_unknown']} ===")
    print("=== 车级口径: 分母=具名违章车(匿名'?'不计, 是召回下界); 误罚=预测到已知非违章车(必须0) ===")


if __name__ == "__main__":
    main()
