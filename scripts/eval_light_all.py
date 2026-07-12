"""红绿灯状态 全量评测 (要求#6 / #10: 标准化指标 + 独立评测集)。

把 datasets/gt/events.csv 的"段级"信号灯真值展开为"逐帧"真值,
对 input_video/ 下全部 违章XX.mp4 跑检测器(仅信号灯, 不跑 YOLO, 秒级),
用 light_state_metrics 计算:
  - 每视频: accuracy / macro_f1 / 各类 P-R-F1
  - 全量合计
  - review 覆盖率: inferred/occluded 段里, 检测器输出 unknown 的比例(进复核队列)

GT 尊重 light_evidence 列:
  - visible 段  -> 期望检测器输出字面 light_state (green/red/flashing)
  - inferred/occluded 段 -> 期望输出 unknown (灯不可见, 进 review, 不逼检测器"看见")
  - unknown 段(无 evidence) -> 期望 unknown

用法:
    python scripts/eval_light_all.py
    python scripts/eval_light_all.py --fill-gap 2.0     # 对短未知间隙前向填充(模拟引擎)
    python scripts/eval_light_all.py --out report.csv
"""
import sys
import os
import csv
import json
import argparse
import glob

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
from redlight.infrastructure.config import load_config, project_root
from redlight.models.traffic_light import TrafficLightDetector
from redlight.evaluation.evaluator import Evaluator


SAMPLE_FPS = 8


def expand_gt(gt_map):
    """返回闭包: ts -> (gt_state_for_detector, evidence)
    - visible 段: 期望检测器输出字面 light_state
    - inferred/occluded 段: 期望输出 unknown (灯不可见, 进 review)
    - unknown 段(无 evidence): 期望 unknown
    """
    def fn(ts):
        for a, b, stt, ev in gt_map:
            if a <= ts <= b:
                if ev in ("inferred", "occluded"):
                    return "unknown", ev
                return stt, (ev or "visible")
        return "unknown", "n/a"
    return fn


def forward_fill(states, ts_list, gap_sec):
    """对 <= gap_sec 的连续 unknown 间隙, 用前一个已知状态填充(模拟引擎行为)。"""
    if gap_sec is None or gap_sec <= 0:
        return states
    out = list(states)
    n = len(out)
    i = 0
    while i < n:
        if out[i] != "unknown":
            i += 1
            continue
        j = i
        while j < n and out[j] == "unknown":
            j += 1
        prev = None
        for k in range(i - 1, -1, -1):
            if out[k] != "unknown":
                prev = out[k]
                break
        if prev is not None and (ts_list[j - 1] - ts_list[i]) <= gap_sec:
            for k in range(i, j):
                out[k] = prev
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill-gap", type=float, default=None,
                    help="未知间隙前向填充秒数(模拟引擎 unknown_light_to_review 前向填充)")
    ap.add_argument("--out", default=None, help="输出逐视频指标 CSV")
    ap.add_argument("--priors", default=None,
                    help="JSON: {视频名: [cx, cy, roi_px]} 行人信号先验(校准过的视频才给)")
    args = ap.parse_args()

    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    ev = Evaluator()

    priors = {}
    if args.priors and os.path.exists(args.priors):
        with open(args.priors, encoding="utf-8") as f:
            priors = json.load(f)
        print(f"[priors] 加载 {len(priors)} 个视频的行人信号先验")

    # 载入段级 GT (含 light_evidence)
    gt_by_video = {}
    with open(os.path.join(ROOT, "datasets", "gt", "events.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt_by_video.setdefault(r["video"], []).append(
                (float(r["start_s"]), float(r["end_s"]),
                 r["light_state"], r.get("light_evidence", "") or ""))

    videos = sorted(glob.glob(os.path.join(ROOT, "input_video", "违章*.mp4")))
    all_pred, all_gt = [], []
    vis_pred_all, vis_gt_all = [], []
    rows = []
    for V in videos:
        name = os.path.splitext(os.path.basename(V))[0]
        cap = cv2.VideoCapture(V)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(round(fps / SAMPLE_FPS)))
        det = TrafficLightDetector(cfg, verbose=False)
        pr = priors.get(name)
        if pr:
            det.signal_prior = (float(pr[0]), float(pr[1]))
            det.prior_roi_px = int(pr[2])
        pred_states, ts_list = [], []
        fi = 0
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if fi % interval == 0:
                pred_states.append(det.detect(fr).get("state"))
                ts_list.append(fi / fps)
            fi += 1
        cap.release()

        gt_fn = expand_gt(gt_by_video.get(name, []))
        pairs = [gt_fn(ts) for ts in ts_list]
        gt_states = [s for s, _ in pairs]
        gt_ev = [ev for _, ev in pairs]
        if args.fill_gap:
            pred_states = forward_fill(pred_states, ts_list, args.fill_gap)

        rep = ev.evaluate_light_states(pred_states, gt_states)

        # 可见段(visible)单独指标 —— Phase-1 主指标: 清楚可见的信号必须判对
        vis_pred = [p for p, e in zip(pred_states, gt_ev) if e == "visible"]
        vis_gt = [g for g, e in zip(gt_states, gt_ev) if e == "visible"]
        vrep = ev.evaluate_light_states(vis_pred, vis_gt) if vis_pred else None
        vis_pred_all += vis_pred
        vis_gt_all += vis_gt

        # review 覆盖率: inferred/occluded 帧里检测器输出 unknown 的比例
        inf = [(p, g) for p, g, e in zip(pred_states, gt_states, gt_ev)
               if e in ("inferred", "occluded")]
        review_cov = (sum(1 for p, _ in inf if p == "unknown") / len(inf)) if inf else None

        rows.append((name, rep, review_cov, len(inf), vrep))
        all_pred += pred_states
        all_gt += gt_states
        rc = f"review_cov={review_cov:.2f}({len(inf)})" if review_cov is not None else "review_cov=n/a"
        vline = (f" vis_acc={vrep['accuracy']:.3f} visF1={vrep['macro_f1']:.3f} vis_n={vrep['n']}"
                 ) if vrep else " vis=n/a"
        print(f"{name}: acc={rep['accuracy']:.3f} macroF1={rep['macro_f1']:.3f} n={rep['n']}"
              f" greenF1={rep['per_class'].get('green',{}).get('f1',0):.2f}"
              f" redF1={rep['per_class'].get('red',{}).get('f1',0):.2f}"
              f" unkF1={rep['per_class'].get('unknown',{}).get('f1',0):.2f} {rc}{vline}")

    rep = ev.evaluate_light_states(all_pred, all_gt)
    # 全量 review_cov 用各视频加权和(逐帧 gt_ev 未保留, 用每视频 review_cov 反推)
    tot_inf = sum(r[3] for r in rows)
    tot_unk_in_inf = 0
    # 用每视频 review_cov 反推
    for name, r, rc, ninf, vrep in rows:
        if rc is not None:
            tot_unk_in_inf += rc * ninf
    all_review_cov = (tot_unk_in_inf / tot_inf) if tot_inf else None

    print(f"\n===== 合计 (n={rep['n']}) =====")
    print(f"  accuracy={rep['accuracy']:.3f}  macro_f1={rep['macro_f1']:.3f}")
    for cls, d in rep["per_class"].items():
        print(f"  {cls:9s} P={d['precision']:.3f} R={d['recall']:.3f} "
              f"F1={d['f1']:.3f} support={d['support']}")
    if vis_pred_all:
        vrep = ev.evaluate_light_states(vis_pred_all, vis_gt_all)
        print(f"\n===== 可见段(visible-only)主指标 合计 (n={vrep['n']}) =====")
        print(f"  accuracy={vrep['accuracy']:.3f}  macro_f1={vrep['macro_f1']:.3f}")
        for cls, d in vrep["per_class"].items():
            print(f"  {cls:9s} P={d['precision']:.3f} R={d['recall']:.3f} "
                  f"F1={d['f1']:.3f} support={d['support']}")
    if all_review_cov is not None:
        print(f"  全量 review 覆盖率={all_review_cov:.3f} (推断/遮挡帧总数={tot_inf})")

    if args.out:
        with open(os.path.join(ROOT, args.out), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "accuracy", "macro_f1", "n",
                        "green_P", "green_R", "green_F1",
                        "red_P", "red_R", "red_F1",
                        "unknown_P", "unknown_R", "unknown_F1",
                        "flashing_P", "flashing_R", "flashing_F1",
                        "review_cov", "inferred_frames",
                        "vis_accuracy", "vis_macro_f1", "vis_n"])
            for name, r, rc, ninf, vrep in rows:
                pc = r["per_class"]
                def g(c, k):
                    return pc.get(c, {}).get(k, 0.0)
                vrow = (round(vrep["accuracy"], 3), round(vrep["macro_f1"], 3), vrep["n"]) if vrep else ("", "", "")
                w.writerow([name, round(r["accuracy"], 3), round(r["macro_f1"], 3), r["n"],
                            g("green", "precision"), g("green", "recall"), g("green", "f1"),
                            g("red", "precision"), g("red", "recall"), g("red", "f1"),
                            g("unknown", "precision"), g("unknown", "recall"), g("unknown", "f1"),
                            g("flashing", "precision"), g("flashing", "recall"), g("flashing", "f1"),
                            (round(rc, 3) if rc is not None else ""), ninf,
                            vrow[0], vrow[1], vrow[2]])
            vtot = (ev.evaluate_light_states(vis_pred_all, vis_gt_all) if vis_pred_all else None)
            vrow = (round(vtot["accuracy"], 3), round(vtot["macro_f1"], 3), vtot["n"]) if vtot else ("", "", "")
            w.writerow(["ALL", round(rep["accuracy"], 3), round(rep["macro_f1"], 3), rep["n"],
                        "", "", "", "", "", "", "", "", "", "", "", "",
                        (round(all_review_cov, 3) if all_review_cov is not None else ""), tot_inf,
                        vrow[0], vrow[1], vrow[2]])
        print(f"\n[CSV] {args.out}")


if __name__ == "__main__":
    main()
