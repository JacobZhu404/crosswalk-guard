"""Phase A 只读挖矿: 全 11 视频挖 [85-106] 类"持续假绿"样本。

目标: 量化 prior 直采在"灯灭/勿框期"采到车辆绿的规模与分布, 坐实根因。
- 复刻生产 traffic_light._sample_prior_color / _sample_roi(纯 cv2/numpy, 不动生产码)。
- 生产口径: prior ROI=prior_roi_px(160), 紧 ROI 暗(无点亮像素)→ 扩展 prior_roi_expand_factor(2.0) 倍到 320px 重采;
  宽松 HSV 阈(sat_min=60 in inRange 下限) + g_frac>r_frac*1.3 → green。
- "假绿"定义: prior 直采返回 green, 但 GT light_states 该时刻灯态=red/unknown/off(即非真实 green)。
  (GT oracle 仅作诊断, 禁入生产推理 — 红线。)

输出: console 汇总 + data/output/mine_false_green/*.json (供诊断报告引用)。
红线: 只读; 不碰生产 prior/权重/_sample_roi/ped_signal.pt; 不 push/merge。
署名: Co-Authored-By: 白板 <wb@crosswalk-guard.agents>
"""
import os, sys, json, glob, csv
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# input_video/ 是 gitignored(963M), 不在 worktree 内; 只读回退到 canonical main 仓(红线: 仅读不拷不入库)。
_INPUT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "input_video"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "crosswalk-guard", "input_video"),
]
INPUT_DIR = next((p for p in _INPUT_CANDIDATES if os.path.isdir(p)), _INPUT_CANDIDATES[0])
PRIORS_PATH = os.path.join(PROJECT_ROOT, "configs", "light_priors.json")
GT_STATES = os.path.join(PROJECT_ROOT, "datasets", "gt", "light_states.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "mine_false_green")

PRIOR_ROI_PX = 160
EXPAND_FACTOR = 2.0  # 生产 prior_roi_expand_factor; 160*2.0 = 320px

# ---------- 复刻生产 _sample_roi (traffic_light.py:626) ----------
def sample_roi(frame, px, py, roi_px, w, h):
    cx_i, cy_i = int(px * w), int(py * h)
    x1 = max(0, cx_i - roi_px // 2); y1 = max(0, cy_i - roi_px // 2)
    x2 = min(w, cx_i + roi_px // 2); y2 = min(h, cy_i + roi_px // 2)
    if x2 <= x1 or y2 <= y1:
        return None, 0, 0, None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0, 0, None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 生产宽松阈(sat_min=60 在 inRange 下限)
    g_mask = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([95, 255, 255]))
    r1 = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([12, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([158, 60, 40]), np.array([180, 255, 255]))
    r_mask = r1 | r2
    g_n = int(cv2.countNonZero(g_mask)); r_n = int(cv2.countNonZero(r_mask))
    total = (x2 - x1) * (y2 - y1)
    if total == 0:
        return None, 0, 0, None
    g_frac, r_frac = g_n / total, r_n / total
    # 绿像素质心相对 ROI 中心偏移(归一化): 集中灯样(近0) vs 弥散车辆/环境(偏大)
    off = None
    if g_n > 0:
        ys, xs = np.where(g_mask > 0)
        cxs = (xs.mean() / roi_px) - 0.5
        cys = (ys.mean() / roi_px) - 0.5
        off = float((cxs ** 2 + cys ** 2) ** 0.5)
    min_frac = 0.002
    if g_frac < min_frac and r_frac < min_frac:
        return None, g_n, r_n, off
    if g_frac > r_frac * 1.3:
        return "green", g_n, r_n, off
    if r_frac > g_frac * 1.3:
        return "red", g_n, r_n, off
    return ("green" if g_frac >= r_frac else "red"), g_n, r_n, off

# ---------- 复刻生产 _sample_prior_color (traffic_light.py:597) ----------
def sample_prior_color(frame, prior):
    px, py = prior[0], prior[1]
    h, w = frame.shape[:2]
    res, gn, rn, off = sample_roi(frame, px, py, PRIOR_ROI_PX, w, h)
    if res is not None:
        return res, gn, rn, off, False
    # 紧 ROI 暗 -> 扩展重试
    big = int(PRIOR_ROI_PX * EXPAND_FACTOR)
    res, gn, rn, off = sample_roi(frame, px, py, big, w, h)
    return res, gn, rn, off, True

# ---------- GT oracle (仅诊断) ----------
def load_gt_states():
    states = {}
    with open(GT_STATES, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = row["video"].strip()
            try:
                s = float(row["start_s"]); e = float(row["end_s"])
            except ValueError:
                continue
            states.setdefault(v, []).append((s, e, row["state"].strip()))
    return states

def gt_color_at(gt_intervals, t):
    """返回 GT 此时刻灯态分类: green / red / unknown(含 off/occluded/tentative) / None(无标注)。
    仅在 GT 显式覆盖时返回具体态; 否则 None(未标注, 不得当作"非绿"误判假绿)。"""
    for s, e, st in gt_intervals:
        if s <= t < e:
            if st == "green":
                return "green"
            if st == "red":
                return "red"
            return "unknown"  # occluded / other / unknown / off
    return None

# ---------- 主挖矿 ----------
def mine_video(video_name, prior, gt_intervals, fps_target=1.0):
    path = os.path.join(INPUT_DIR, f"{video_name}.mp4")
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / fps_target)))
    # 三类假绿池(仅 GT 显式覆盖时计入): definitive(GT=red) / suspected(GT=unknown/off)
    # 另有 unannotated(GT=None) 单独统计, 不计入"假绿"。
    runs = {"definitive": [], "suspected": []}  # 每个 run: {start_s,end_s,via_expand_frames,total_frames,via}
    cur = {"definitive": None, "suspected": None}
    expanded_frames = 0
    expanded_definitive = 0   # 扩展重采且落在 definitive 假绿
    expanded_suspected = 0
    unannotated_fg = 0        # 未标注区采到绿(排除出假绿计数)
    unannotated_runs = []
    cur_un = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        t = frame_idx / fps
        h, w = frame.shape[:2]
        tight, _, _, _ = sample_roi(frame, prior[0], prior[1], PRIOR_ROI_PX, w, h)
        triggered_expand = tight is None
        if triggered_expand:
            expanded_frames += 1
        res, gn, rn, off, used_expand = sample_prior_color(frame, prior)
        gt = gt_color_at(gt_intervals, t)
        is_green = (res == "green")
        if gt == "red":
            cls = "definitive"
        elif gt in ("unknown",):
            cls = "suspected"
        else:
            cls = None
        if is_green and cls is not None:
            via = "expand" if (triggered_expand or used_expand) else "tight"
            if (triggered_expand or used_expand) and cls == "definitive":
                expanded_definitive += 1
            if (triggered_expand or used_expand) and cls == "suspected":
                expanded_suspected += 1
            c = cur[cls]
            if c is None:
                cur[cls] = {"start_s": t, "end_s": t, "tight": 0, "expand": 0, "off_sum": 0.0, "n": 0}
                c = cur[cls]
            c["end_s"] = t
            c[via] += 1
            if off is not None:
                c["off_sum"] += off; c["n"] += 1
        else:
            for k in ("definitive", "suspected"):
                if cur[k] is not None:
                    runs[k].append(cur[k]); cur[k] = None
        # 未标注区
        if is_green and gt is None:
            unannotated_fg += 1
            if cur_un is None:
                cur_un = {"start_s": t, "end_s": t}
            else:
                cur_un["end_s"] = t
        else:
            if cur_un is not None:
                unannotated_runs.append(cur_un); cur_un = None
        frame_idx += 1
    for k in ("definitive", "suspected"):
        if cur[k] is not None:
            runs[k].append(cur[k])
    if cur_un is not None:
        unannotated_runs.append(cur_un)
    cap.release()
    def span_s(runs_list, step, fps):
        return round(sum((r["end_s"] - r["start_s"]) + step / fps for r in runs_list), 2)
    return {
        "video": video_name, "fps": round(fps, 2), "fps_target": fps_target,
        "frames_sampled": frame_idx // step,
        "expanded_frames": expanded_frames,
        "expanded_definitive_fg_frames": expanded_definitive,
        "expanded_suspected_fg_frames": expanded_suspected,
        "definitive_runs": runs["definitive"], "definitive_total_s": span_s(runs["definitive"], step, fps),
        "suspected_runs": runs["suspected"], "suspected_total_s": span_s(runs["suspected"], step, fps),
        "unannotated_runs": unannotated_runs, "unannotated_fg_frames": unannotated_fg,
    }

def _fmt_runs(runs, n=8):
    out = []
    for r in runs[:n]:
        via = "expand" if r.get("expand", 0) > 0 else "tight"
        off = (r["off_sum"] / r["n"]) if r.get("n") else 0.0
        out.append(f"[{r['start_s']:.1f}-{r['end_s']:.1f}:{via}:off{off:.2f}]")
    return ", ".join(out)

def main():
    with open(PRIORS_PATH, encoding="utf-8") as f:
        priors = json.load(f)
    gt = load_gt_states()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    print(f"{'video':8} {'def_s':>7} {'susp_s':>7} {'expand_fr':>9} {'exp_def':>7} {'exp_susp':>8} {'unann_fg':>8}  definitive_runs")
    print("-" * 120)
    for v in sorted(priors.keys()):
        r = mine_video(v, priors[v], gt.get(v, []))
        if r is None:
            continue
        results[v] = r
        print(f"{v:8} {r['definitive_total_s']:>7} {r['suspected_total_s']:>7} {r['expanded_frames']:>9} "
              f"{r['expanded_definitive_fg_frames']:>7} {r['expanded_suspected_fg_frames']:>8} {r['unannotated_fg_frames']:>8}  {_fmt_runs(r['definitive_runs'])}")
    total_def = sum(r["definitive_total_s"] for r in results.values())
    total_susp = sum(r["suspected_total_s"] for r in results.values())
    total_exp_def = sum(r["expanded_definitive_fg_frames"] for r in results.values())
    summary = {
        "params": {"prior_roi_px": PRIOR_ROI_PX, "expand_factor": EXPAND_FACTOR,
                   "expanded_px": int(PRIOR_ROI_PX * EXPAND_FACTOR), "fps_target": 1.0,
                   "note": "假绿仅计 GT 显式覆盖(red=definitive, unknown/off=suspected); GT=None 不计入"},
        "total_definitive_s": round(total_def, 2),
        "total_suspected_s": round(total_susp, 2),
        "total_expanded_definitive_fg_frames": total_exp_def,
        "per_video": results,
    }
    out = os.path.join(OUTPUT_DIR, "false_green_mining.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("-" * 120)
    print(f"ALL: definitive_s={total_def:.2f}  suspected_s={total_susp:.2f}  expanded_definitive_fg_frames={total_exp_def}")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
