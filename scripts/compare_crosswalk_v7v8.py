"""v7 vs v8 斑马线掩膜对比 (2026-07-11)。

对诊断 CSV 标识的 4 个关键帧, 分别用 v7(灰度阈值) 和 v8(HSV白色检测)
生成带掩膜叠加的可视化, 用于肉眼确认修复效果。
"""
import sys
import os
import cv2
import numpy as np

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config, project_root
from redlight.models.vehicle import VehicleDetector
from redlight.models.crosswalk import CrosswalkDetector
from redlight.infrastructure.geometry import compute_overlap_ratio


TARGETS = {
    "违章01": [28.56, 61.02],
    "违章02": [7.54, 109.91],
}


def overlay_mask(vis, mask, color=(0, 200, 0), alpha=0.35):
    if mask is None:
        return vis
    m = (mask > 0).astype(np.uint8)
    o = vis.copy()
    o[m == 1] = color
    cv2.addWeighted(o, alpha, vis, 1 - alpha, 0, vis)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, 2)


def compare_one(video, ts, cfg, out_dir, det):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    name = os.path.splitext(os.path.basename(video))[0]

    cw_v7 = CrosswalkDetector(cfg, verbose=False)
    cw_v7._cv = cw_v7._cv_v7  # force v7
    cw_v7.detect = lambda frame: cw_v7._cv_v7(frame)

    cw_v8 = CrosswalkDetector(cfg, verbose=False)

    idx = int(round(ts * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  SKIP {name} ts={ts}")
        return

    # 车辆检测 (共享)
    dets = det.detect(frame)

    # v7
    m7 = cw_v7.detect(frame)
    vis7 = frame.copy()
    overlay_mask(vis7, m7, (0, 180, 0))   # green for v7
    max_ov7 = 0.0
    for d in dets:
        ov = compute_overlap_ratio(d["xyxy"], m7) if m7 is not None else 0.0
        max_ov7 = max(max_ov7, ov)
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        col = (0, 0, 255) if ov >= 0.15 else (128, 128, 0)
        cv2.rectangle(vis7, (x1, y1), (x2, y2), col, 2)

    # v8
    m8 = cw_v8.detect(frame)
    vis8 = frame.copy()
    overlay_mask(vis8, m8, (255, 100, 0))  # orange for v8
    max_ov8 = 0.0
    for d in dets:
        ov = compute_overlap_ratio(d["xyxy"], m8) if m8 is not None else 0.0
        max_ov8 = max(max_ov8, ov)
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        col = (0, 0, 255) if ov >= 0.15 else (128, 128, 0)
        cv2.rectangle(vis8, (x1, y1), (x2, y2), col, 2)

    # labels
    area7 = int(np.count_nonzero(m7)) if m7 is not None else 0
    area8 = int(np.count_nonzero(m8)) if m8 is not None else 0
    cv2.putText(vis7, f"[v7] ts={ts:.1f} max_ov={max_ov7:.3f} mask_px={area7}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(vis8, f"[v8] ts={ts:.1f} max_ov={max_ov8:.3f} mask_px={area8}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 150, 0), 2)

    p7 = os.path.join(out_dir, f"cmp_{name}_ts{ts}_v7.jpg")
    p8 = os.path.join(out_dir, f"cmp_{name}_ts{ts}_v8.jpg")
    cv2.imwrite(p7, vis7)
    cv2.imwrite(p8, vis8)
    print(f"  [{name}] ts={ts}  v7:ov={max_ov7:.3f}px={area7}  v8:ov={max_ov8:.3f}px={area8}")
    print(f"    -> {p7}\n    -> {p8}")


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    base_in = r"E:\BaiduNetdiskDownload"
    out_dir = os.path.join(project_root(), "data", "output", "cw_debug")
    os.makedirs(out_dir, exist_ok=True)
    det = VehicleDetector(cfg, verbose=False)
    for vid, targets in TARGETS.items():
        video = os.path.join(base_in, f"{vid}.mp4")
        if not os.path.exists(video):
            continue
        print(f"\n===== {vid} =====")
        for ts in targets:
            compare_one(video, ts, cfg, out_dir, det)


if __name__ == "__main__":
    main()
