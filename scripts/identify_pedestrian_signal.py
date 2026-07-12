"""数据驱动识别行人信号灯位置 — 用 GT 行人灯态做一次性标定 (非人工猜测).

思路:
  1. 不加先验跑检测器, 跟踪所有持久信号头(self.heads), 记录每头每帧的 (cx,cy,last_dom).
  2. 解析 label_result_01.csv 的行人灯态时段 -> 每视频 GT 行人灯态时间线.
  3. 把每头的颜色时间线(每秒) 与 GT 行人灯态对比 -> 匹配率.
  4. 匹配率最高且覆盖足够的头 = 行人信号灯 -> 输出其 (cx,cy) 作为 signal_prior.

用法:
  python scripts/identify_pedestrian_signal.py input_video/违章02.mp4 \
      --gt input_video/label_result_01.csv --sec 110
  # 一次跑多个:
  python scripts/identify_pedestrian_signal.py input_video/违章02.mp4 input_video/违章03.mp4 input_video/违章04.mp4
"""
import sys, os, csv, re, argparse
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
import cv2
from redlight.models.traffic_light import TrafficLightDetector
from redlight.infrastructure.config import load_config


# ---------------- GT 解析 ----------------
def _to_sec(s):
    s = s.strip()
    m = re.match(r"(\d+):(\d+)", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def parse_gt_light(csv_path):
    """返回 {video_basename: [(start_s, end_s, 'green'|'red'), ...]} (仅行人灯态段)."""
    out = {}
    with open(csv_path, encoding="gbk", errors="ignore") as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:
        if len(row) < 3:
            continue
        vid, desc = row[0].strip(), row[1]
        segs = parse_desc(desc)
        if segs:
            out[vid] = segs
    return out


def parse_desc(desc):
    """从自由文本提取行人灯态段. 支持: 全程红/绿, A-B 红灯, Xs以后变绿/红, X前红X后绿."""
    segs = []
    # 1) 全程
    m = re.search(r"全程.*?行人灯.*?(绿|红)灯", desc)
    if m:
        col = "green" if m.group(1) == "绿" else "red"
        return [(0, 99999, col)]
    # 2) 时间范围段 "A-B ... 行人灯...绿/红灯"
    for mm in re.finditer(r"([\d:]+)\s*[-–]\s*([\d:]+)[^行人灯]*?行人灯[^绿红]*?(绿|红)灯", desc):
        a, b = _to_sec(mm.group(1)), _to_sec(mm.group(2))
        if a is None or b is None:
            continue
        segs.append((a, b, "green" if mm.group(3) == "绿" else "red"))
    # 3) "Xs以后 行人灯变绿/红" 或 "X以后是绿/红灯"
    for mm in re.finditer(r"([\d:]+)\s*以后[^行人灯]*?行人灯[^绿红]*?(绿|红)", desc):
        a = _to_sec(mm.group(1))
        if a is None:
            continue
        segs.append((a, 99999, "green" if mm.group(2) == "绿" else "red"))
    # 4) "Xs之前红灯，Xs之后绿灯" (11 号)
    m2 = re.search(r"([\d:]+)\s*之前.*?红.*?([\d:]+)\s*之后.*?绿", desc)
    if m2:
        a, b = _to_sec(m2.group(1)), _to_sec(m2.group(2))
        if a is not None and b is not None:
            segs.append((0, a, "red"))
            segs.append((b, 99999, "green"))
    return segs


def gt_timeline(segs, max_t):
    """返回 {second: 'green'|'red'} (段覆盖到的秒)."""
    tl = {}
    for a, b, col in segs:
        b = min(b, max_t)
        for t in range(int(a), int(b) + 1):
            tl[t] = col
    return tl


# ---------------- 识别主流程 ----------------
def identify(video, gt_segs, cfg, max_sec=None, grid=0.05):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = (nframes / fps) if fps else 0
    if max_sec is None or max_sec <= 0:
        max_sec = dur
    det = TrafficLightDetector(cfg, verbose=False)
    fi = 0
    obs = []  # (fi, rcx, rcy, dom, total)
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
        return None, []

    # 按 (rcx,rcy) 聚类
    groups = {}
    for fi2, rcx, rcy, dom, tot in obs:
        groups.setdefault((rcx, rcy), []).append((fi2, dom, tot))

    gt_tl = gt_timeline(gt_segs, max_t=int(max_sec) + 1)
    results = []
    for (rcx, rcy), recs in groups.items():
        # 每秒取该组该秒最后一次 dom
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
        # 平均面积(区分真灯泡 vs 噪声)
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
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--gt", default="input_video/label_result_01.csv")
    ap.add_argument("--sec", type=float, default=0.0, help="每视频最多处理秒数(0=整段)")
    ap.add_argument("--grid", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    gt_all = parse_gt_light(args.gt)
    print(f"GT 视频数={len(gt_all)}")
    for v in args.videos:
        base = os.path.splitext(os.path.basename(v))[0]
        segs = gt_all.get(base)
        if not segs:
            print(f"\n[跳过] {base}: GT 中无行人灯态段")
            continue
        print(f"\n===== {base}  GT段={segs} =====")
        results, meta = identify(v, segs, cfg, max_sec=args.sec, grid=args.grid)
        if results is None:
            print("  无持久信号头")
            continue
        print(f"  视频: fps={meta['fps']:.1f} dur={meta['dur']}s nframes={meta['nframes']}")
        print(f"  {'pos(cx,cy)':<16}{'obs':>5}{'match':>6}{'mis':>5}{'score':>7}{'cover':>7}{'avgTot':>8}")
        for r in results[:8]:
            mark = " <== 行人信号候选" if r is results[0] else ""
            print(f"  {str(r['pos']):<16}{r['n_obs']:>5}{r['match']:>6}{r['mismatch']:>5}"
                  f"{r['score']:>7}{r['coverage']:>7}{r['avg_total']:>8}{mark}")
        best = results[0]
        print(f"  >> 建议 signal_prior = {best['pos']}  (score={best['score']}, coverage={best['coverage']})")


if __name__ == "__main__":
    main()
