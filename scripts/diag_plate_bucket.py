"""qw 车牌诊断: 失败四桶归因(eval_plate 层) + 生产 consensus 层对比。

桶定义(对每个 GT 车牌):
  a  没检到   : 全视频无任何 plate box, 或该车区域从未被检测(文本代理: 无 ED<=1 且无足够接近的候选)
  b  OCR 读错 : 有 box 且尺寸正常(宽>=80 高>=24) 且 conf>=0.5, 但所有单帧文本 ED>1
  d  图像质量 : 有 box 但 box 小/糊(宽<80 或 高<24) 或 conf<0.5, 文本 ED>1
  c  共识/选帧错: 单帧读到过 ED<=1 的正确文本, 但最终输出(频次 top 匹配)不含 -> 被投票/频次淹没
  OK          : 最终匹配正确(ED<=1 输出)

另跑生产管线(cli.run)输出 ev["plate"], 对比生产口径(event 车牌)与 eval_plate 口径。

用法:
  python scripts/diag_plate_bucket.py [--videos 违章02 违章03 ...] [--fps 8] [--production]
  --production: 额外跑 cli.run 全管线(11 视频, 慢), 出生产口径事件车牌 vs GT。
"""
import argparse
import csv
import os
import sys
from collections import Counter

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer
from redlight.evaluation.metrics import levenshtein


def clean_plates(s):
    out = []
    for p in (s or "").split(";"):
        p = p.strip()
        if p and p not in ("?", "无牌") and not p.startswith("["):
            out.append(p)
    return out


def parse_gt():
    gt = {}
    with open(os.path.join(ROOT, "datasets", "gt", "events.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ps = set(clean_plates(row["violating_plates"]) + clean_plates(row["other_plates"]))
            if ps:
                gt.setdefault(row["video"], set()).update(ps)
    return gt


def collect(video_path, cfg, sample_fps=8):
    """逐帧 plate.detect, 返回 [(frame_idx, text, conf, w, h, x1, y1, x2, y2)]。"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / sample_fps)))
    plate = PlateRecognizer(cfg, verbose=False)
    recs = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            for p in plate.detect(frame):
                txt = p.get("text", "")
                conf = p.get("conf", 0.0)
                if not txt or conf < 0.1:
                    continue
                x1, y1, x2, y2 = [int(v) for v in p.get("xyxy", [0, 0, 0, 0])]
                recs.append((frame_idx, txt, conf, x2 - x1, y2 - y1, x1, y1, x2, y2))
        frame_idx += 1
    cap.release()
    return recs


def bucketize(gt_plates, recs):
    """对每 GT 车牌分桶。"""
    out = {}
    # 全视频所有文本与 box
    all_texts = [r[1] for r in recs]
    for gp in sorted(gt_plates):
        # 单帧最好 ED 与对应帧
        best = None  # (ed, text, conf, w, h, frame)
        for r in recs:
            ed = levenshtein(r[1], gp)
            if best is None or ed < best[0]:
                best = (ed, r[1], r[2], r[3], r[4], r[0])
        if best is None:
            out[gp] = {"bucket": "a_no_detect", "detail": "全视频无任何 plate box"}
            continue
        ed, text, conf, w, h, fr = best
        if ed <= 1:
            # 读到过正确/近正确文本 -> 是否进最终输出?
            out[gp] = {"bucket": "read_ok_single_frame", "detail": f"ED={ed} {text} conf={conf:.2f} box={w}x{h} @f{fr}",
                       "ed": ed, "w": w, "h": h, "conf": conf}
        else:
            # 从未读对 -> b/d 看 box 质量
            if w < 80 or h < 24 or conf < 0.5:
                out[gp] = {"bucket": "d_image_quality", "detail": f"最接近 ED={ed} {text} conf={conf:.2f} box={w}x{h} @f{fr}",
                           "ed": ed, "w": w, "h": h, "conf": conf}
            else:
                out[gp] = {"bucket": "b_ocr_wrong", "detail": f"最接近 ED={ed} {text} conf={conf:.2f} box={w}x{h} @f{fr}",
                           "ed": ed, "w": w, "h": h, "conf": conf}
    return out


def final_top(recs, gt_plates, topn=20):
    """eval_plate 口径: 频次 top 里 ED<=1 匹配。"""
    cnt = Counter(r[1] for r in recs)
    top = [p for p, _ in cnt.most_common(topn)]
    matched = {}
    for gp in gt_plates:
        for p in top:
            if levenshtein(p, gp) <= 1:
                matched[gp] = p
                break
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--production", action="store_true", help="额外跑 cli.run 生产口径")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    gt = parse_gt()
    video_dir = os.path.join(ROOT, "input_video")
    videos = args.videos or sorted(gt.keys())

    summary = Counter()
    per_video = {}
    for v in videos:
        vp = os.path.join(video_dir, f"{v}.mp4")
        if not os.path.isfile(vp):
            print(f"[跳过] {v} 视频缺失")
            continue
        recs = collect(vp, cfg, args.fps)
        matched = final_top(recs, gt.get(v, set()))
        buckets = bucketize(gt.get(v, set()), recs)
        # 用最终匹配回填 OK / c
        for gp, b in buckets.items():
            if gp in matched:
                b["bucket"] = "ok"
            elif b["bucket"] == "read_ok_single_frame":
                b["bucket"] = "c_consensus_or_frame"
        per_video[v] = buckets
        for gp, b in buckets.items():
            summary[b["bucket"]] += 1
            print(f"{v} {gp}: [{b['bucket']}] {b['detail']}")

    print("\n=== 分桶汇总(eval_plate 层) ===")
    for k in ["ok", "a_no_detect", "b_ocr_wrong", "c_consensus_or_frame", "d_image_quality"]:
        print(f"  {k:<22} {summary[k]}")

    if args.production:
        print("\n=== 生产口径(cli.run 事件车牌) ===")
        from redlight.app import cli
        for v in videos:
            vp = os.path.join(video_dir, f"{v}.mp4")
            events = cli.run(cfg, vp, os.path.join(ROOT, "data", "output", "qw", f"plate_prod_{v}"),
                             preset="balanced")
            ev_plates = [e.get("plate", "") for e in events if e.get("status") == "confirmed" and e.get("plate")]
            gt_v = gt.get(v, set())
            hits = [p for p in ev_plates if any(levenshtein(p, g) <= 1 for g in gt_v)]
            print(f"  {v}: 事件车牌={ev_plates} GT={sorted(gt_v)} 命中={len(hits)}/{len(gt_v)}")


if __name__ == "__main__":
    main()
