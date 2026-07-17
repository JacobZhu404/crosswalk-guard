#!/usr/bin/env python3
"""把标注画廊的反馈 CSV 应用到 GT 骨架 JSON (Part B 标注闭环最后一步)。

- crosswalk: crosswalk_feedback.csv(video,t_sec,y0,y1,verdict) -> datasets/gt/crosswalk/{video}.json 的 frames[].y0/y1
- tracking:  tracking_feedback.csv(video,t_sec,box,verdict)     -> datasets/gt/tracking/{video}.json 的 anchors[].frames[].box

按 (video, ts±tol) 匹配。verdict=no_crosswalk 的帧保持 y0/y1=null(视为无带)。

用法:
  python scripts/apply_annotation_gt.py --kind crosswalk
  python scripts/apply_annotation_gt.py --kind tracking
"""
import os
import sys
import csv
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_feedback(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    return rows


def _to_int(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _match_ts(fb_rows, video, ts, tol=0.15):
    """在反馈里找 (video, ts±tol) 的最近一条。"""
    best, bd = None, tol
    for r in fb_rows:
        if r.get("video", "") != video:
            continue
        try:
            rts = float(r.get("t_sec", ""))
        except (TypeError, ValueError):
            continue
        d = abs(rts - ts)
        if d <= bd:
            bd, best = d, r
    return best


def _parse_poly(raw):
    """'x,y;x,y;x,y;x,y' -> [[x,y],...]; 无效返回 None。"""
    raw = (raw or "").strip()
    if not raw:
        return None
    pts = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        xy = [_to_int(v) for v in pair.split(",")]
        if len(xy) != 2 or any(v is None for v in xy):
            return None
        pts.append(xy)
    return pts if len(pts) >= 3 else None


def apply_crosswalk(fb_rows, gt_dir):
    n_updated = 0
    for fn in sorted(os.listdir(gt_dir)):
        if not fn.endswith(".json"):
            continue
        video = fn[:-5]
        path = os.path.join(gt_dir, fn)
        with open(path, encoding="utf-8") as f:
            gt = json.load(f)
        changed = False
        for fr in gt.get("frames", []):
            r = _match_ts(fb_rows, video, float(fr["ts"]))
            if not r:
                continue
            poly = _parse_poly(r.get("poly"))
            if poly is not None and fr.get("poly") != poly:
                fr["poly"] = poly
                changed = True
                n_updated += 1
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(gt, f, ensure_ascii=False, indent=2)
            print(f"  [{video}] 已回填")
    return n_updated


def apply_tracking(fb_rows, gt_dir):
    n_updated = 0
    for fn in sorted(os.listdir(gt_dir)):
        if not fn.endswith(".json"):
            continue
        video = fn[:-5]
        path = os.path.join(gt_dir, fn)
        with open(path, encoding="utf-8") as f:
            gt = json.load(f)
        changed = False
        for anc in gt.get("anchors", []):
            for fr in anc.get("frames", []):
                r = _match_ts(fb_rows, video, float(fr["ts"]))
                if not r:
                    continue
                box_raw = (r.get("box") or "").strip()
                if not box_raw:
                    continue
                parts = [_to_int(p) for p in box_raw.replace(";", ",").split(",")]
                if len(parts) == 4 and all(p is not None for p in parts):
                    if fr.get("box") != parts:
                        fr["box"] = parts
                        changed = True
                        n_updated += 1
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(gt, f, ensure_ascii=False, indent=2)
            print(f"  [{video}] 已回填")
    return n_updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["crosswalk", "tracking"], required=True)
    ap.add_argument("--feedback-csv", default=None)
    ap.add_argument("--gt-dir", default=None)
    args = ap.parse_args()

    fb = args.feedback_csv or os.path.join(
        ROOT, "data", "output", "annotated", f"{args.kind}_feedback.csv")
    gt_dir = args.gt_dir or os.path.join(ROOT, "datasets", "gt", args.kind)

    fb_rows = _load_feedback(fb)
    if not fb_rows:
        print(f"无反馈或为空: {fb}")
        return
    if not os.path.isdir(gt_dir):
        print(f"无骨架目录: {gt_dir}")
        return

    print(f"=== 应用 {args.kind} 标注: {fb} -> {gt_dir} ===")
    n = apply_crosswalk(fb_rows, gt_dir) if args.kind == "crosswalk" else apply_tracking(fb_rows, gt_dir)
    print(f"共回填 {n} 个标注。")


if __name__ == "__main__":
    main()
