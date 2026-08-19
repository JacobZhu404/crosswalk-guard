"""09 如何赢得 TP (confirmed 绿来源追溯) + 重定位后绿是否保留 (Q3, 只读).

忠实复刻 TrafficLightDetector.observe() 的 color 路径(逐比特调用真实
_sample_prior_color / _sample_box / _candidates), 并记录每帧是走
  - "yolo": YOLO traffic-light 框(离 prior 远→use_yolo) 判色
  - "prior_hsv": 先验 ROI HSV 直采(错位 ROI 抓车辆绿/树叶绿的来源)
  - "global": 无先验全局亮斑
判定 09 当前 [11-72] 的绿到底来自哪个分支; 再模拟"重定位 prior 到真灯位"后重跑,
证明 [11-72] 绿不消失、[72-106] 假绿被消除。

红线: 不改生产码; 仅实例化真实检测器 + 复刻 observe 逻辑(加日志); 不碰
light_priors.json / _sample_roi / ped_signal.pt。
"""
import os
import sys
import json
import cv2
import numpy as np

CANON = "/Users/jacob/personal/crosswalk-guard"
VIDEO = os.path.join(CANON, "input_video", "违章09.mp4")
CANON_GT = os.path.join(CANON, "datasets", "gt", "light_canonical_gt.json")
CFG = os.path.join(CANON, "configs", "config.yaml")
LOC_GT = os.path.join(CANON, "datasets", "light_location_gt.json")
VIDEO_NAME = "违章09"
TARGET_FPS = 3.0

sys.path.insert(0, os.path.join(CANON, "src"))
from redlight.infrastructure.config import load_config  # noqa: E402
from redlight.models.traffic_light import TrafficLightDetector  # noqa: E402
from redlight.models.vehicle import VehicleDetector  # noqa: E402


def faithful_observe(tl, frame, yolo_boxes):
    """逐比特复刻 traffic_light.py observe() color 路径, 返回 (obs, branch, detail)."""
    tl._last_frame = frame
    h, w = frame.shape[:2]
    spots = tl._candidates(frame)
    # ---- YOLO 分支 ----
    if yolo_boxes and frame is not None:
        best = None
        best_dist = None
        for box in yolo_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            bcx = ((x1 + x2) / 2) / w
            bcy = ((y1 + y2) / 2) / h
            if bcy < tl.yolo_cy_min:
                continue
            col, gn, rn = tl._sample_box(box, w, h)
            if col is None:
                continue
            if tl.signal_prior is not None:
                px, py = tl.signal_prior
                dist = ((bcx - px) ** 2 + (bcy - py) ** 2) ** 0.5
                score = -dist
            else:
                dist = None
                score = max(gn, rn)
            if best is None or score > best[1]:
                best = (col, score, gn, rn)
                best_dist = dist
        use_yolo = best is not None and (
            tl.signal_prior is None or best_dist is None or best_dist > tl.yolo_prior_near
        )
        if use_yolo:
            col, _, gn, rn = best
            return col, "yolo", {"box_cx": bcx, "box_cy": bcy,
                                 "dist_to_prior": best_dist, "g_n": gn, "r_n": rn}
    # ---- 先验模式 ----
    if tl.signal_prior is not None and frame is not None:
        sampled = tl._sample_prior_color()
        if sampled is not None:
            g_n, r_n = tl._last_sample
            return sampled, "prior_hsv", {"g_n": g_n, "r_n": r_n}
        px, py = tl.signal_prior
        r = tl.prior_search_radius
        near = [s for s in spots
                if ((s["cx"] - px) ** 2 + (s["cy"] - py) ** 2) ** 0.5 <= r
                and s["color"] in ("green", "red")]
        if near:
            greens = sum(s["area"] for s in near if s["color"] == "green")
            reds = sum(s["area"] for s in near if s["color"] == "red")
            obs = "green" if greens >= reds else "red"
            return obs, "prior_near", {"greens": greens, "reds": reds}
    # ---- 全局 ----
    greens = sum(s["area"] for s in spots if s.get("color") == "green")
    reds = sum(s["area"] for s in spots if s.get("color") == "red")
    if greens == 0 and reds == 0:
        obs = "off"
    elif greens >= reds:
        obs = "green"
    else:
        obs = "red"
    return obs, "global", {}


def gt_state_at(t, gt_timeline):
    """最近邻 GT 状态(green/red/off/unknown)."""
    if not gt_timeline:
        return "unknown"
    best = min(gt_timeline, key=lambda r: abs(r["t"] - t))
    if abs(best["t"] - t) > 3.0:
        return "unknown"
    return best["state"]


def main():
    cfg = load_config(CFG)
    tl = TrafficLightDetector(cfg, verbose=False)
    det = VehicleDetector(cfg, verbose=False)

    # GT timeline (canonical): 每帧 governing ped 状态
    canon = json.load(open(CANON_GT, encoding="utf-8"))
    f09 = [f for f in canon["frames"] if f["video"] == VIDEO_NAME]
    gt_timeline = []
    for f in f09:
        peds = [b for b in f.get("boxes", [])
                if b.get("type") == "pedestrian" and b.get("governing")]
        if not peds:
            peds = [b for b in f.get("boxes", []) if b.get("type") == "pedestrian"]
        if peds:
            color = peds[0]["color"]
            state = "off" if f.get("no_light") else color
        else:
            state = "off" if f.get("no_light") else "unknown"
        gt_timeline.append({"t": f["t"], "state": state})
    gt_timeline.sort(key=lambda r: r["t"])

    # 重定位目标: light_location_gt.json 的 09 derived true_center
    loc = json.load(open(LOC_GT, encoding="utf-8"))
    d09 = loc["derived_per_video"][VIDEO_NAME]
    reloc_cx, reloc_cy = d09["true_center"]

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / TARGET_FPS)))

    rows_cur = []   # (t, obs, branch, gt_state)
    rows_rel = []
    frame_idx = 0
    branch_cur_count = {}
    branch_rel_count = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            t = frame_idx / fps
            # 真实 YOLO 框
            det.detect(frame)
            boxes = det.last_light_boxes

            # ---- 当前 prior (0.2,0.35) ----
            tl.set_video_prior(VIDEO_NAME)   # (0.2,0.35,160)
            obs_c, br_c, _ = faithful_observe(tl, frame, boxes)
            branch_cur_count[br_c] = branch_cur_count.get(br_c, 0) + 1
            rows_cur.append({"t": round(t, 2), "obs": obs_c, "branch": br_c,
                             "gt": gt_state_at(t, gt_timeline)})

            # ---- 重定位 prior (真灯位) ----
            tl.signal_prior = (float(reloc_cx), float(reloc_cy))
            tl.prior_roi_px = 160
            obs_r, br_r, _ = faithful_observe(tl, frame, boxes)
            branch_rel_count[br_r] = branch_rel_count.get(br_r, 0) + 1
            rows_rel.append({"t": round(t, 2), "obs": obs_r, "branch": br_r,
                             "gt": gt_state_at(t, gt_timeline)})
        frame_idx += 1
    cap.release()

    # ---- 分段统计: 按 GT 状态分桶, 看当前/重定位的绿覆盖率 ----
    def cov(rows, gt_filter, obs_val="green"):
        sub = [r for r in rows if r["gt"] in gt_filter]
        if not sub:
            return None, 0, 0
        n = len(sub)
        m = sum(1 for r in sub if r["obs"] == obs_val)
        return round(m / n, 3), m, n

    # GT green 窗口(主): 状态==green
    cur_green_cov, cur_g_m, cur_g_n = cov(rows_cur, ("green",))
    rel_green_cov, rel_g_m, rel_g_n = cov(rows_rel, ("green",))
    # GT unknown/off 窗口(假绿风险区): 状态 off/unknown
    cur_off_cov, cur_o_m, cur_o_n = cov(rows_cur, ("off", "unknown"))
    rel_off_cov, rel_o_m, rel_o_n = cov(rows_rel, ("off", "unknown"))

    # 当前 prior 假绿(obs=green 但 GT != green)主要来自哪个分支?
    fake_green_branches = {}
    for r in rows_cur:
        if r["obs"] == "green" and r["gt"] != "green":
            fake_green_branches[r["branch"]] = fake_green_branches.get(r["branch"], 0) + 1
    # 当前 prior [11-72] 绿来源分支(GT green 窗口内 obs=green 的分支)
    tp_green_branches = {}
    for r in rows_cur:
        if r["gt"] == "green" and r["obs"] == "green":
            tp_green_branches[r["branch"]] = tp_green_branches.get(r["branch"], 0) + 1

    out = {
        "video": VIDEO_NAME,
        "reloc_target": [reloc_cx, reloc_cy],
        "branch_counts_current_prior": branch_cur_count,
        "branch_counts_relocated_prior": branch_rel_count,
        "coverage": {
            "gt_green_window": {
                "current_prior_green_cov": cur_green_cov, "current_green_m_n": [cur_g_m, cur_g_n],
                "relocated_prior_green_cov": rel_green_cov, "relocated_green_m_n": [rel_g_m, rel_g_n],
            },
            "gt_off_unknown_window": {
                "current_prior_green_cov": cur_off_cov, "current_green_m_n": [cur_o_m, cur_o_n],
                "relocated_prior_green_cov": rel_off_cov, "relocated_green_m_n": [rel_o_m, rel_o_n],
            },
        },
        "current_prior_fake_green_branches": fake_green_branches,
        "current_prior_tp_green_branches": tp_green_branches,
        "rows_current": rows_cur,
        "rows_relocated": rows_rel,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output", "diag_09")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "q3_tp_source.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- 控制台摘要 ----
    print("=== Q3: 09 TP 绿来源 + 重定位影响 ===")
    print(f"重定位目标(真灯位): ({reloc_cx:.3f}, {reloc_cy:.3f})")
    print(f"当前 prior 分支计数: {branch_cur_count}")
    print(f"重定位 prior 分支计数: {branch_rel_count}")
    print()
    print(f"[GT green 窗口] 当前 prior 绿覆盖={cur_green_cov} ({cur_g_m}/{cur_g_n})  "
          f"| 重定位后 绿覆盖={rel_green_cov} ({rel_g_m}/{rel_g_n})")
    print(f"[GT off/unknown 窗口] 当前 prior 假绿覆盖={cur_off_cov} ({cur_o_m}/{cur_o_n})  "
          f"| 重定位后 绿覆盖={rel_off_cov} ({rel_o_m}/{rel_o_n})")
    print()
    print(f"当前 prior 假绿(obs绿但GT非绿)分支分布: {fake_green_branches}")
    print(f"当前 prior TP绿(GT绿且obs绿)分支分布: {tp_green_branches}")
    print(f"\n输出: {out_path}")


if __name__ == "__main__":
    main()
