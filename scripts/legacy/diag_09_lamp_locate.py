"""只读诊断: 用 events.csv 的 09 行人灯态 GT, 不加先验跑 detect 重标定 09 真实灯位。

思路(同 identify_pedestrian_signal.py, 但 GT 源改为 events.csv):
  1. 不加 signal_prior 跑 TrafficLightDetector.detect, 跟踪所有持久信号头(heads)。
  2. 把每头的颜色时间线(每秒) 与 events.csv 的 09 红/绿段对比 -> 匹配率。
  3. 匹配率最高且覆盖足够的头 = 行人信号灯 -> 输出 (cx,cy) 作为建议 signal_prior。

不修改任何生产文件; 仅打印诊断。用法:
  python scripts/diag_09_lamp_locate.py <video> <events.csv> [--grid 0.05]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
import cv2
from redlight.models.traffic_light import TrafficLightDetector
from redlight.infrastructure.config import load_config


def parse_events_gt(csv_path, video):
    """返回 [(start_s, end_s, 'green'|'red'), ...] (仅该视频的 红/绿 段, 跳过 unknown)。"""
    segs = []
    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = [x.strip() for x in line.strip().split(",")]
            if len(p) < 4:
                continue
            if p[0] != video:
                continue
            try:
                s = float(p[1]); e = float(p[2])
            except ValueError:
                continue
            st = p[3]
            if st in ("green", "red"):
                segs.append((s, e, st))
    return segs


def gt_timeline(segs, max_t):
    tl = {}
    for a, b, col in segs:
        b = min(b, max_t)
        for t in range(int(a), int(b) + 1):
            tl[t] = col
    return tl


def identify(video, gt_segs, cfg, max_sec=None, grid=0.05):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = (nframes / fps) if fps else 0
    if max_sec is None or max_sec <= 0:
        max_sec = dur
    det = TrafficLightDetector(cfg, verbose=False)
    fi = 0
    obs = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = fi / fps
        if t > max_sec:
            break
        det.detect(fr)
        for h in det.heads:
            if h.get("frames_seen", 0) < 3:
                continue
            if h.get("last_dom") not in ("green", "red"):
                continue
            rcx = round(h["cx"] / grid) * grid
            rcy = round(h["cy"] / grid) * grid
            obs.append((fi, rcx, rcy, h["last_dom"], h.get("total", 0.0)))
        fi += 1
    cap.release()
    if not obs:
        return None, {"fps": fps, "dur": round(dur, 1), "nframes": nframes}

    groups = {}
    for fi2, rcx, rcy, dom, tot in obs:
        groups.setdefault((rcx, rcy), []).append((fi2, dom, tot))

    gt_tl = gt_timeline(gt_segs, max_t=int(max_sec) + 1)
    results = []
    for (rcx, rcy), recs in groups.items():
        by_sec = {}
        for fi2, dom, tot in recs:
            sec = int(fi2 / fps)
            by_sec[sec] = dom
        match = mismatch = 0
        for sec, g in gt_tl.items():
            hd = by_sec.get(sec)
            if hd is None:
                continue
            if hd == g:
                match += 1
            else:
                mismatch += 1
        denom = match + mismatch
        score = match / denom if denom > 0 else 0.0
        avg_tot = float(np.mean([t for _, _, t in recs])) if recs else 0.0
        coverage = denom / max(1, len(gt_tl))
        results.append({
            "pos": (round(rcx, 3), round(rcy, 3)),
            "n_obs": len(recs), "match": match, "mismatch": mismatch,
            "score": round(score, 3), "coverage": round(coverage, 3),
            "avg_total": round(avg_tot, 1),
        })
    results.sort(key=lambda r: (-r["score"], -r["coverage"], -r["avg_total"]))
    return results, {"fps": fps, "dur": round(dur, 1), "nframes": nframes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gt", default="datasets/gt/events.csv")
    ap.add_argument("--sec", type=float, default=0.0)
    ap.add_argument("--grid", type=float, default=0.05)
    args = ap.parse_args()
    cfg = load_config("configs/config.yaml")
    base = os.path.splitext(os.path.basename(args.video))[0]
    segs = parse_events_gt(args.gt, base)
    if not segs:
        print(f"[跳过] {base}: events.csv 中无红/绿段")
        return
    print(f"===== {base}  GT段={segs} =====")
    results, meta = identify(args.video, segs, cfg, max_sec=args.sec, grid=args.grid)
    if results is None:
        print("  无持久信号头")
        return
    print(f"  视频: fps={meta['fps']:.1f} dur={meta['dur']}s nframes={meta['nframes']}")
    print(f"  {'pos(cx,cy)':<16}{'obs':>5}{'match':>6}{'mis':>5}{'score':>7}{'cover':>7}{'avgTot':>8}")
    for i, r in enumerate(results[:10]):
        mark = " <== 行人信号候选" if i == 0 else ""
        print(f"  {str(r['pos']):<16}{r['n_obs']:>5}{r['match']:>6}{r['mismatch']:>5}"
              f"{r['score']:>7}{r['coverage']:>7}{r['avg_total']:>8}{mark}")
    best = results[0]
    print(f"  >> 建议 signal_prior = {best['pos']}  (score={best['score']}, coverage={best['coverage']})")
    print(f"  (当前 light_priors.json 09 = [0.2, 0.35, 160])")


if __name__ == "__main__":
    main()
