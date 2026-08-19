"""09 真行人灯像素搜索 + 漂移测量 + IoU 位置级验证 (Q1/Q2, 只读).

对齐 datasets/light_location_gt.json + datasets/gt/light_canonical_gt.json,
回答:
  Q1: 09 是否存在可用的真行人灯像素? 该区域在多少帧里可见? (IoU 位置级)
  Q2: 09 固定机位还是运动? 隔离单灯帧测真实漂移(避免多灯聚合虚高).

红线: 不改任何生产文件; 只读视频 + GT + 复用 TrafficLightDetector._candidates(纯cv2).
不碰 light_priors.json / _sample_roi / _sample_prior_color / ped_signal.pt.
"""
import os
import sys
import json
import cv2
import numpy as np

# ---- 路径(只读访问 canonical 资产; worktree 无 input_video) ----
CANON = "/Users/jacob/personal/crosswalk-guard"
VIDEO = os.path.join(CANON, "input_video", "违章09.mp4")
LOC_GT = os.path.join(CANON, "datasets", "light_location_gt.json")
CANON_GT = os.path.join(CANON, "datasets", "gt", "light_canonical_gt.json")
CFG = os.path.join(CANON, "configs", "config.yaml")
VIDEO_NAME = "违章09"

sys.path.insert(0, os.path.join(CANON, "src"))
from redlight.infrastructure.config import load_config  # noqa: E402
from redlight.models.traffic_light import TrafficLightDetector  # noqa: E402

TARGET_FPS = 3.0          # 采样率(诊断足够; 秒级)
NEAR_R = 0.06             # 真灯区域半径(归一化): 距 GT 中心 < 此值算"落在真灯区"
MIN_SPOT_AREA = 20        # 最小亮斑面积(像素)


def iou(box_a, box_b):
    """box = [x1,y1,x2,y2] 归一化."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def main():
    cfg = load_config(CFG)
    tl = TrafficLightDetector(cfg, verbose=False)  # 仅用 _candidates(纯cv2)

    # ---- GT: light_location_gt.json 的 09 derived ----
    loc = json.load(open(LOC_GT, encoding="utf-8"))
    d09 = loc["derived_per_video"][VIDEO_NAME]
    true_cx, true_cy = d09["true_center"]
    true_w, true_h = d09["true_box_wh"]
    true_box = [true_cx - true_w / 2, true_cy - true_h / 2,
                true_cx + true_w / 2, true_cy + true_h / 2]

    # ---- GT: canonical 逐帧 governing ped 框(权威位置 + 状态) ----
    canon = json.load(open(CANON_GT, encoding="utf-8"))
    f09 = [f for f in canon["frames"] if f["video"] == VIDEO_NAME]
    # 每帧: 取所有 pedestrian(governing) 框的质心 + 状态
    gt_rows = []
    for f in f09:
        peds = [b for b in f.get("boxes", [])
                if b.get("type") == "pedestrian" and b.get("governing")]
        if not peds:
            peds = [b for b in f.get("boxes", []) if b.get("type") == "pedestrian"]
        if peds:
            cxs = [(b["box_norm"][0] + b["box_norm"][2]) / 2 for b in peds]
            cys = [(b["box_norm"][1] + b["box_norm"][3]) / 2 for b in peds]
            cx = float(np.mean(cxs)); cy = float(np.mean(cys))
        else:
            cx = cy = None
        color = peds[0]["color"] if peds else ("off" if f.get("no_light") else "unknown")
        gt_rows.append({"t": f["t"], "cx": cx, "cy": cy,
                        "no_light": f.get("no_light", False), "color": color})
    gt_rows.sort(key=lambda r: r["t"])

    # ---- Q2: 漂移(隔离单灯: 用 GT governing 质心, 不含多灯聚合) ----
    valid = [r for r in gt_rows if r["cx"] is not None]
    cx_arr = np.array([r["cx"] for r in valid])
    cy_arr = np.array([r["cy"] for r in valid])
    drift = {
        "n_frames": len(valid),
        "cx_mean": float(cx_arr.mean()), "cx_std": float(cx_arr.std()),
        "cx_min": float(cx_arr.min()), "cx_max": float(cx_arr.max()),
        "cx_spread": float(cx_arr.max() - cx_arr.min()),
        "cy_mean": float(cy_arr.mean()), "cy_std": float(cy_arr.std()),
        "cy_min": float(cy_arr.min()), "cy_max": float(cy_arr.max()),
        "cy_spread": float(cy_arr.max() - cy_arr.min()),
    }

    # GT 推导的"真灯质心均值"作为重定位目标(比单点更稳)
    reloc_cx, reloc_cy = float(cx_arr.mean()), float(cy_arr.mean())
    reloc_box = [reloc_cx - true_w / 2, reloc_cy - true_h / 2,
                 reloc_cx + true_w / 2, reloc_cy + true_h / 2]

    # ---- 当前 prior ROI 框(center 0.2,0.35, roi_px 160) ----
    PRIOR_CX, PRIOR_CY = 0.2, 0.35
    ROI_PX = 160
    cap = cv2.VideoCapture(VIDEO)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    half_x = (ROI_PX / 2) / W
    half_y = (ROI_PX / 2) / H
    cur_prior_box = [PRIOR_CX - half_x, PRIOR_CY - half_y,
                     PRIOR_CX + half_x, PRIOR_CY + half_y]
    reloc_prior_box = [reloc_cx - half_x, reloc_cy - half_y,
                       reloc_cx + half_x, reloc_cy + half_y]

    # ---- Q1: 可见帧(独立检测器验证真灯区是否有点亮像素) ----
    # 用 _candidates 找亮斑, 统计落在真灯区(距 GT 质心 < NEAR_R)的彩色亮斑帧数
    stride = max(1, int(round(fps / TARGET_FPS)))
    n_sampled = 0
    visible_green = 0
    visible_red = 0
    visible_any = 0
    # 真灯区质心轨迹(独立检测器): 每帧取距 reloc 中心最近的亮斑
    det_cx, det_cy = [], []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            n_sampled += 1
            spots = tl._candidates(frame)
            # 落在真灯区的彩色亮斑
            near = [s for s in spots
                    if s.get("color") in ("green", "red")
                    and ((s["cx"] - reloc_cx) ** 2 + (s["cy"] - reloc_cy) ** 2) ** 0.5 < NEAR_R]
            if near:
                visible_any += 1
                greens = sum(1 for s in near if s["color"] == "green")
                reds = sum(1 for s in near if s["color"] == "red")
                if greens >= reds:
                    visible_green += 1
                else:
                    visible_red += 1
                # 距 reloc 中心最近的亮斑质心(漂移轨迹)
                best = min(near, key=lambda s: (s["cx"] - reloc_cx) ** 2 + (s["cy"] - reloc_cy) ** 2)
                det_cx.append(best["cx"]); det_cy.append(best["cy"])
        frame_idx += 1
    cap.release()

    det_cx = np.array(det_cx); det_cy = np.array(det_cy)
    det_drift = {
        "n": int(len(det_cx)),
        "cx_mean": float(det_cx.mean()) if len(det_cx) else None,
        "cx_std": float(det_cx.std()) if len(det_cx) else None,
        "cy_mean": float(det_cy.mean()) if len(det_cy) else None,
        "cy_std": float(det_cy.std()) if len(det_cy) else None,
    }

    # ---- IoU 位置级 ----
    iou_cur = iou(cur_prior_box, true_box)
    iou_reloc = iou(reloc_prior_box, true_box)
    # 真灯是否被 ROI 包含(交集=真灯框面积比例)
    inter_reloc = iou_reloc * (max(0.0, reloc_prior_box[2] - reloc_prior_box[0]) * max(0.0, reloc_prior_box[3] - reloc_prior_box[1]))
    true_area = true_w * true_h
    contains_reloc = (inter_reloc / true_area) if true_area > 0 else 0.0
    # 当前 prior 中心 -> 真灯中心 距离
    dist_cur = ((PRIOR_CX - true_cx) ** 2 + (PRIOR_CY - true_cy) ** 2) ** 0.5
    dist_reloc = ((reloc_cx - true_cx) ** 2 + (reloc_cy - true_cy) ** 2) ** 0.5

    # GT 可见帧统计(真灯存在)
    gt_present = sum(1 for r in gt_rows if not r["no_light"])

    out = {
        "video": VIDEO_NAME,
        "frame_size": [W, H], "fps": fps,
        "true_light_location_gt": {
            "true_center": [true_cx, true_cy], "true_box_wh": [true_w, true_h],
            "true_box_norm": true_box,
        },
        "reloc_target_from_canonical_mean": {"cx": reloc_cx, "cy": reloc_cy},
        "Q1_visibility": {
            "sampled_frames": n_sampled,
            "gt_present_frames": gt_present,
            "detector_visible_any": visible_any,
            "detector_visible_green": visible_green,
            "detector_visible_red": visible_red,
            "visible_fraction": round(visible_any / n_sampled, 3) if n_sampled else 0,
            "note": "可见=独立检测器在真灯区(NEAR_R=%.3f)检测到彩色亮斑的帧" % NEAR_R,
        },
        "Q2_drift_gt_centroid": drift,
        "Q2_drift_detector_centroid": det_drift,
        "position_iou": {
            "current_prior_box": cur_prior_box,
            "relocated_prior_box": reloc_prior_box,
            "true_box": true_box,
            "iou_current_prior_vs_true": round(iou_cur, 4),
            "iou_relocated_prior_vs_true": round(iou_reloc, 4),
            "true_box_contained_in_relocated_roi_frac": round(contains_reloc, 4),
            "dist_current_prior_to_true_center": round(dist_cur, 4),
            "dist_relocated_prior_to_true_center": round(dist_reloc, 4),
        },
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "output", "diag_09"),
                exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "output",
                            "diag_09", "q1q2_true_light.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- 控制台摘要 ----
    print("=== Q1: 09 真行人灯可见性 ===")
    print(f"  真灯中心(GT derived): ({true_cx:.3f}, {true_cy:.3f})  框wh=({true_w:.3f},{true_h:.3f})")
    print(f"  采样帧={n_sampled}  GT存在帧={gt_present}  检测器可见(任何)={visible_any} "
          f"({visible_any / n_sampled * 100:.1f}%) 绿={visible_green} 红={visible_red}")
    print("=== Q2: 09 机位/灯漂移 (隔离单灯质心) ===")
    print(f"  GT质心: cx={drift['cx_mean']:.3f}±{drift['cx_std']:.3f} "
          f"spread={drift['cx_spread']:.3f} | cy={drift['cy_mean']:.3f}±{drift['cy_std']:.3f} "
          f"spread={drift['cy_spread']:.3f} (n={drift['n_frames']})")
    print(f"  检测器质心: cx={det_drift['cx_mean']}±{det_drift['cx_std']} "
          f"cy={det_drift['cy_mean']}±{det_drift['cy_std']} (n={det_drift['n']})")
    print("=== 位置级 IoU (gate③) ===")
    print(f"  当前 prior(0.2,0.35) ROI vs 真灯框: IoU={iou_cur:.4f}  "
          f"中心距={dist_cur:.3f}  -> 真灯完全在 ROI 外(错位)")
    print(f"  重定位后 prior ROI vs 真灯框: IoU={iou_reloc:.4f}  "
          f"中心距={dist_reloc:.4f}  真灯被ROI包含={contains_reloc*100:.1f}%")
    print(f"\n输出: {out_path}")


if __name__ == "__main__":
    main()
