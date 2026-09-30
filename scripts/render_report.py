#!/usr/bin/env python3
"""生成"最终效果"交付物: 逐视频文字报告 + 带标注结果视频。

产出(默认 data/output/report_<ts>/):
  1) REPORT.md            —— 逐视频文字总结(窗级+车级+车牌+最终结论), 全局汇总。
  2) <video>_result.mp4   —— 实时标注视频(车辆框/静止/压线高亮/灯态/斑马线/车牌)
     + 顶部"最终判定"横幅(confirmed/review + 回填车牌, 落在事件窗内显示)。

两遍法(诚实): pass-1 跑生产流水线(cli.run, annotated_video=on)得 annotated.mp4(逐帧
实时推理框)+ 事件(批处理最终判定, 视频结束后才有); pass-2 把最终判定横幅叠加到
annotated.mp4 的对应时间窗上 -> <video>_result.mp4。实时框=推理过程, 横幅=最终结论。

用法:
  python scripts/render_report.py                      # 全 11 视频
  python scripts/render_report.py --videos 违章05 违章11
  python scripts/render_report.py --keep-intermediate  # 保留中间 annotated.mp4
"""
import os
import sys
import csv
import time
import argparse

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.evaluation.violation_eval import (
    load_violation_gt, load_car_level_gt, match_violation_events,
    classify_false_positives, match_cars, load_video_metadata,
)


def _run_one(cfg, video, out_dir):
    """跑生产流水线(annotated_video=on), 返回 events。annotated.mp4 落在 out_dir。"""
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        return None, None
    cfg.output.annotated_video = True
    cfg.output.csv_report = True
    cfg.output.evidence_images = False    # 报告视频不需要证据截图, 省时间/磁盘
    events = cli.run(cfg, video_path, out_dir)   # 默认 detector/occ_denom 从 config(=v2/box)
    return events, video_path


def _confirmed(events):
    return [e for e in (events or []) if e.get("status") == "confirmed"]


def _stamp_verdict(annotated_path, result_path, events, fps_hint=None):
    """pass-2: 把最终判定横幅叠加到 annotated.mp4 的事件时间窗 -> result_path。"""
    cap = cv2.VideoCapture(annotated_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(result_path, fourcc, fps, (W, H))
    # 事件窗 -> 横幅文案
    spans = []
    for e in events or []:
        st = e.get("status")
        if st not in ("confirmed", "review"):
            continue
        plate = e.get("plate") or "车牌未读出"
        label = "违章确认" if st == "confirmed" else "待人工复核"
        color = (0, 0, 255) if st == "confirmed" else (0, 165, 255)
        spans.append((float(e["start_ts"]), float(e["end_ts"]), f"{label}  车牌:{plate}", color))
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = fi / fps
        active = [s for s in spans if s[0] <= ts <= s[1]]
        if active:
            _, _, text, color = active[0]
            # 顶部居中横幅(半透明底 + 文字)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, 54), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
            cv2.putText(frame, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2, cv2.LINE_AA)
        vw.write(frame)
        fi += 1
    cap.release()
    vw.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None, help="默认全部 input_video/*.mp4")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--events", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--videos-csv", default=os.path.join(ROOT, "datasets", "gt", "videos.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()

    out_root = args.out or os.path.join(ROOT, "data", "output",
                                        f"report_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_root, exist_ok=True)

    cfg = load_config(args.config)
    gt = load_violation_gt(args.events)
    car_gt = load_car_level_gt(args.events)
    meta = load_video_metadata(args.videos_csv)

    if args.videos:
        videos = args.videos
    else:
        videos = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "input_video"))
                        if f.endswith(".mp4"))

    report_lines = []
    report_lines.append("# 斑马线违章检测 — 最终效果报告\n")
    report_lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  检测器: v2  |  占道分母: box  |  preset: balanced\n")
    report_lines.append(f"> 视频数: {len(videos)}  |  结果视频: `<视频名>_result.mp4`(实时框+最终判定横幅)\n")
    report_lines.append("\n---\n")

    agg = {"win_tp": 0, "win_fp": 0, "win_fn": 0,
           "car_tp": 0, "car_fn": 0, "car_misfine": 0, "car_unknown": 0,
           "plate_hits": 0, "plate_total": 0}

    for v in videos:
        vdir = os.path.join(out_root, v)
        os.makedirs(vdir, exist_ok=True)
        print(f"\n===== {v} =====")
        events, vpath = _run_one(cfg, v, vdir)
        if events is None:
            print(f"  [跳过] 找不到 {v}.mp4")
            continue
        conf = _confirmed(events)
        gt_v = gt.get(v, [])
        is_neg = not meta.get(v, True)

        r = match_violation_events(conf, gt_v, min_overlap_s=args.min_overlap)
        cls = classify_false_positives(r, conf, gt_v, is_negative=is_neg, min_overlap_s=args.min_overlap)
        pred_plates = set()
        for e in conf:
            if e.get("plate"):
                pred_plates.add(e["plate"].strip())
            for p in (e.get("plates") or []):
                if p.strip():
                    pred_plates.add(p.strip())
        cr = match_cars(pred_plates, car_gt.get(v, {"violating": set(), "non_violating": set()}))

        # pass-2: 叠加最终判定横幅
        annotated = os.path.join(vdir, "annotated.mp4")
        result = os.path.join(out_root, f"{v}_result.mp4")
        stamped = False
        if os.path.isfile(annotated):
            stamped = _stamp_verdict(annotated, result, events)
            if stamped and not args.keep_intermediate:
                os.remove(annotated)

        # 汇总
        agg["win_tp"] += r["tp"]; agg["win_fp"] += r["fp"]; agg["win_fn"] += r["fn"]
        agg["car_tp"] += cr["car_tp"]; agg["car_fn"] += cr["car_fn"]
        agg["car_misfine"] += cr["car_misfine"]; agg["car_unknown"] += cr["car_unknown"]
        agg["plate_hits"] += r["plate_hits"]; agg["plate_total"] += r["plate_total"]

        # 逐视频报告段
        tag = " (负例, 应无违章)" if is_neg else ""
        report_lines.append(f"\n## {v}{tag}\n")
        if conf:
            for e in conf:
                report_lines.append(
                    f"- **违章确认** [{e['start_ts']:.1f}s–{e['end_ts']:.1f}s] "
                    f"车牌: `{e.get('plate') or '未读出'}`"
                    + (f"  其他车: {', '.join(p for p in (e.get('plates') or []) if p != e.get('plate'))}"
                       if len([p for p in (e.get('plates') or []) if p != e.get('plate')]) else "")
                    + f"  灯态: {e.get('light_state','?')}\n")
        else:
            report_lines.append("- 无违章确认事件（输出 0，符合负例/无违章）\n" if is_neg
                                 else "- 无违章确认事件\n")
        # GT 对照
        if gt_v:
            gt_desc = "; ".join(f"[{g['start_s']:.0f}–{g['end_s']:.0f}s] 违章车: "
                                f"{', '.join(g['plates']) or '未具名'}" for g in gt_v)
            report_lines.append(f"- GT 违章窗: {gt_desc}\n")
        report_lines.append(
            f"- 窗级: P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f} "
            f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})\n")
        car_bits = f"命中 {cr['car_tp']} / 漏 {cr['car_fn']} / 误罚 {cr['car_misfine']}"
        if cr["hit"]:      car_bits += f"  ✓{cr['hit']}"
        if cr["missed"]:   car_bits += f"  ✗漏{cr['missed']}"
        if cr["misfined"]: car_bits += f"  ⚠误罚{cr['misfined']}"
        if cr["unknown"]:  car_bits += f"  ?未登记{cr['unknown']}"
        report_lines.append(f"- 车级(按车辆算): {car_bits}\n")
        report_lines.append(f"- 结果视频: `{v}_result.mp4`" + ("" if stamped else " (未生成)") + "\n")

    # 全局汇总
    wt, wf, wn = agg["win_tp"], agg["win_fp"], agg["win_fn"]
    wp = wt / (wt + wf) if (wt + wf) else 0.0
    wr = wt / (wt + wn) if (wt + wn) else 0.0
    wf1 = 2 * wp * wr / (wp + wr) if (wp + wr) else 0.0
    ct, cfn, cm = agg["car_tp"], agg["car_fn"], agg["car_misfine"]
    crecall = ct / (ct + cfn) if (ct + cfn) else 0.0
    cprec = ct / (ct + cm) if (ct + cm) else (1.0 if ct else 0.0)
    summary = [
        "\n---\n\n## 全局汇总\n",
        f"- **窗级**: P={wp:.3f} R={wr:.3f} F1={wf1:.3f}  (tp={wt} fp={wf} fn={wn})\n",
        f"- **车级(按车辆算)**: 具名违章车召回={crecall:.3f}(命中{ct}/漏{cfn}) "
        f"具名精度={cprec:.3f} **误罚={cm}** 未登记={agg['car_unknown']}\n",
        f"- **车牌命中**: {agg['plate_hits']}/{agg['plate_total']}(TP 事件里)\n",
        "- 口径: 窗级=每过街窗1票; 车级=每违章车1票(匿名'?'不计=召回下界); 误罚=预测到已知非违章车(必须0)\n",
    ]
    report_lines += summary
    for ln in summary:
        print(ln.rstrip("\n"))

    report_path = os.path.join(out_root, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\n[报告] {report_path}")
    print(f"[结果视频] {out_root}/<视频名>_result.mp4")


if __name__ == "__main__":
    main()
