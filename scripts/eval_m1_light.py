"""M1 端到端灯态评测: color vs ped_classifier (M1 调好验证, 要求#6)。

逐帧跑 TrafficLightDetector(两种 traffic_light.method), 切成连续状态段,
与 datasets/gt/light_states.csv 的分段真值比较, 算 accuracy / macro-F1。

- color: 原 v7 HSV 颜色启发式(基线)
- ped_classifier: 行人信号状态分类器(本任务 M1 目标, 需 models/ped_signal.pt)

用法:
    python scripts/eval_m1_light.py                # 评 02/03/04, 两种 method 都跑并对比
    python scripts/eval_m1_light.py --method ped_classifier   # 只跑一类(快)
"""
import os
import sys
import csv
import argparse

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np
from redlight.infrastructure.config import load_config, project_root
from redlight.models.traffic_light import TrafficLightDetector
from redlight.evaluation.evaluator import Evaluator


INPUT = os.path.join(ROOT, "input_video")
GT_CSV = os.path.join(ROOT, "datasets", "gt", "light_states.csv")
VIDEOS = ["违章02", "违章03", "违章04"]


def load_gt():
    gt = {}
    with open(GT_CSV, newline="") as f:
        for r in csv.DictReader(f):
            v = r["video"]
            gt.setdefault(v, []).append(
                (float(r["start_s"]), float(r["end_s"]), r["state"].strip()))
    return gt


def build_segments(timeline):
    """[(ts, state)] -> 连续同态段 [(start_ts, end_ts, state)]。"""
    segs = []
    cur = None
    for ts, st in timeline:
        if cur is None or cur[2] != st:
            if cur is not None:
                segs.append(cur)
            cur = [ts, ts, st]
        else:
            cur[1] = ts
    if cur is not None:
        segs.append(cur)
    return [(s, e, st) for s, e, st in segs]


def run_method(video, cfg, method):
    """逐帧跑检测器(method 覆盖), 返回 [(ts, state)]。"""
    cfg.traffic_light.method = method
    video_path = os.path.join(INPUT, video + ".mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] 打不开 {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / cfg.inference.fps)))
    tl = TrafficLightDetector(cfg, verbose=False)
    timeline = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % interval == 0:
            res = tl.detect(frame)
            st = res.get("state", "unknown") if isinstance(res, dict) else res
            timeline.append((round(fi / fps, 2), st))
        fi += 1
    cap.release()
    return timeline


def eval_vs_gt(pred_segs, gt_segs, step=1.0):
    """在 GT 段内按步长采样, 取该时刻 pred 段的状态, 收集 (pred, gt) 对。

    pred_segs / gt_segs: [(start_ts, end_ts, state)]。pred 在某个采样时刻若落在
    任何 pred 段内则取其状态, 否则记 'unknown'(检测器该时刻未输出有效态)。
    """
    preds, gts = [], []
    max_pred_t = max((pe for _, pe, _ in pred_segs), default=0.0)
    for gs, ge, gst in gt_segs:
        upper = min(ge, max_pred_t) if ge < 900 else min(900.0, max_pred_t)
        t = gs
        while t <= upper + 1e-6:
            cur = None
            for ps, pe, pst in pred_segs:
                if ps - 1e-6 <= t <= pe + 1e-6:
                    cur = pst
                    break
            preds.append(cur if cur else "unknown")
            gts.append(gst)
            t += step
    return preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="both",
                    choices=["both", "color", "ped_classifier"])
    args = ap.parse_args()
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    gt = load_gt()
    ev = Evaluator()

    methods = (["color", "ped_classifier"] if args.method == "both"
               else [args.method])
    summary = {}
    for v in VIDEOS:
        if v not in gt:
            print(f"[跳过] {v}: 无 GT")
            continue
        print(f"\n===== {v} =====")
        vsummary = {}
        for m in methods:
            timeline = run_method(v, cfg, m)
            pred_segs = build_segments(timeline)
            preds, gts = eval_vs_gt(pred_segs, gt[v])
            if not gts:
                continue
            rep = ev.evaluate_light_states(preds, gts)
            vsummary[m] = rep
            print(f"  [{m:14s}] acc={rep['accuracy']:.3f} macro_f1={rep['macro_f1']:.3f} "
                  f"n={rep['n']}")
            for cls, d in rep["per_class"].items():
                print(f"      {cls:9s} P={d['precision']:.3f} R={d['recall']:.3f} "
                      f"F1={d['f1']:.3f} sup={d['support']}")
        summary[v] = vsummary

    # 汇总: 两种 method 的合计 acc / macro_f1
    if args.method == "both":
        print("\n===== 合计对比 =====")
        for m in methods:
            all_p, all_g = [], []
            for v in VIDEOS:
                if v not in summary or m not in summary[v]:
                    continue
                # 重新聚合(简单起见用各视频 n 加权平均已在 rep 内, 这里直接报各视频)
            print(f"  {m}: 见上方各视频明细")


if __name__ == "__main__":
    main()
