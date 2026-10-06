#!/usr/bin/env python3
"""天花板诊断: 把 GT 多边形斑马线当 mask 塞进管线, 看"若斑马线完美"违章 F1 能到多少。

目的: 回答"斑马线检测到底是不是瓶颈"。分两趟对比占道分母:
  - denom=mask (现状 D2): 占斑马线比例 —— 大区域下占道% 变小, 预期偏低。
  - denom=box  (Jacob 第4点): 车足迹压线比例 —— 预期更合理。

代表多边形: 每视频取一个(优先 note 含"@50%"的关键帧, 否则面积最大的已标帧)。
  斑马线静止, 单帧代表可近似全片(相机漂移会略微低估天花板)。

只读诊断: 不改管线默认行为(cli.run/engine 的注入参数均加法、默认不变)。
用法: python scripts/diag_gt_crosswalk_ceiling.py
"""
import os
import sys
import json
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.evaluation.module_metrics import poly_to_mask
from redlight.evaluation.violation_eval import (
    load_violation_gt, match_violation_events, aggregate,
)
from redlight.app import cli


class GtCrosswalkDetector:
    """把固定 GT 多边形光栅化成掩膜返回(忽略帧内容, 仅用尺寸)。"""
    def __init__(self, poly):
        self.poly = poly

    def detect(self, frame, vehicle_boxes=None):
        h, w = frame.shape[:2]
        return poly_to_mask(self.poly, h, w)

    def get_info(self):
        return type("MI", (), {"name": "GtCrosswalkDetector", "version": "gt"})()


def _poly_area(poly):
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _rep_poly(video):
    path = os.path.join(ROOT, "datasets", "gt", "crosswalk", f"{video}.json")
    if not os.path.exists(path):
        return None
    frames = json.load(open(path, encoding="utf-8")).get("frames", [])
    labeled = [f for f in frames if f.get("poly")]
    if not labeled:
        return None
    mid = [f for f in labeled if "@50%" in (f.get("note") or "")]
    pick = mid[0] if mid else max(labeled, key=lambda f: _poly_area(f["poly"]))
    return pick["poly"]


def _confirmed(events):
    return [e for e in events if e.get("status") == "confirmed"]


def run_pass(cfg, videos, gt, denom, preset, out_root):
    results = []
    for v in videos:
        poly = _rep_poly(v)
        if poly is None:
            print(f"  [跳过 {v}] 无 GT 多边形")
            continue
        video_path = os.path.join(ROOT, "input_video", f"{v}.mp4")
        cfg.output.annotated_video = False
        det = GtCrosswalkDetector(poly)
        events = cli.run(cfg, video_path, os.path.join(out_root, f"gtcw_{denom}_{v}"),
                         preset=preset, crosswalk_detector=det, occ_denom=denom)
        conf = _confirmed(events)
        r = match_violation_events(conf, gt.get(v, []), min_overlap_s=0.5)
        results.append(r)
        print(f"  [{v}] P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f} "
              f"(tp={r['tp']} fp={r['fp']} fn={r['fn']}) 覆盖={r['mean_coverage']:.2f}")
    return aggregate(results) if results else None


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    gt = load_violation_gt(os.path.join(ROOT, "datasets", "gt", "events.csv"))
    # 只跑有 GT 多边形标注的违章视频
    videos = sorted(os.path.basename(p)[:-5]
                    for p in glob.glob(os.path.join(ROOT, "datasets", "gt", "crosswalk", "*.json"))
                    if _rep_poly(os.path.basename(p)[:-5]))
    out_root = os.path.join(ROOT, "data", "output", "gt_crosswalk_ceiling")
    os.makedirs(out_root, exist_ok=True)

    print(f"=== GT 斑马线天花板诊断 ({len(videos)} 视频): {videos} ===")
    print("对照基线(当前 v11 检测器, 同 9 视频): P=0.538 R=0.778 F1=0.636\n")

    for denom in ("mask", "box"):
        print(f"--- 趟: GT斑马线 + denom={denom} ---")
        agg = run_pass(cfg, videos, gt, denom, "balanced", out_root)
        if agg:
            print(f"  >>> 总体 P={agg['precision']:.3f} R={agg['recall']:.3f} F1={agg['f1']:.3f} "
                  f"(tp={agg['tp']} fp={agg['fp']} fn={agg['fn']}) 覆盖={agg['mean_coverage']:.3f}\n")


if __name__ == "__main__":
    main()
