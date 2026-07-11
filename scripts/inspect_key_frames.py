"""关键帧实证核查 (2026-07-11, Task #2/#3)。

针对诊断 CSV 给出的关键帧时间戳, 抽取帧并叠加:
  - 斑马线掩膜 (绿色半透明填充 + 轮廓)
  - 车辆框 (红色) + 各车 overlap%
  - 信号灯状态标签
用于肉眼判断: 掩膜是否准、车是否真压线、绿灯为何缺失。
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
from redlight.models.traffic_light import TrafficLightDetector
from redlight.pipeline.tracker import TrackStateManagerV2
from redlight.infrastructure.geometry import compute_overlap_ratio

TARGETS = {
    "违章01": [28.56, 61.02],   # 28.56=红/压线0.499静止; 61.02=绿/压线0.12
    "违章02": [7.54, 109.91],   # 7.54=unknown/压线0.598静止x2; 109.91=flashing/0.339
}


def inspect(video, targets, cfg, out_dir):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[SKIP] {video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det = VehicleDetector(cfg, verbose=False)
    cw = CrosswalkDetector(cfg)
    tl = TrafficLightDetector(cfg, verbose=False)
    track = TrackStateManagerV2("balanced")
    name = os.path.splitext(os.path.basename(video))[0]

    for ts in targets:
        idx = int(round(ts * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  [{name}] ts={ts}: 无法读取帧@{idx}")
            continue
        mask = cw.detect(frame)
        dets = det.detect(frame)
        states = track.update(dets, ts)
        lres = tl.detect(frame)
        lstate = lres.get("state", "unknown") if isinstance(lres, dict) else lres
        vis = frame.copy()
        # 掩膜叠加
        if mask is not None:
            m = (mask > 0).astype(np.uint8)
            overlay = vis.copy()
            overlay[m == 1] = (0, 200, 0)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        # 车辆框 + overlap
        max_ov = 0.0
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            ov = compute_overlap_ratio(d["xyxy"], mask) if mask is not None else 0.0
            max_ov = max(max_ov, ov)
            col = (0, 0, 255) if ov >= 0.15 else (255, 128, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"{ov:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        # 标签
        cv2.putText(vis, f"light={lstate} ts={ts:.1f} max_ov={max_ov:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        out = os.path.join(out_dir, f"inspect_{name}_ts{ts}.jpg")
        cv2.imwrite(out, vis)
        print(f"  [{name}] ts={ts}: light={lstate} max_ov={max_ov:.3f} "
              f"n_active={len(dets)} mask={'yes' if mask is not None else 'NO'} -> {out}")
    cap.release()


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    base_in = r"E:\BaiduNetdiskDownload"
    out_dir = os.path.join(project_root(), "data", "output", "cw_debug")
    os.makedirs(out_dir, exist_ok=True)
    for vid, targets in TARGETS.items():
        video = os.path.join(base_in, f"{vid}.mp4")
        if not os.path.exists(video):
            print(f"[SKIP] {video}")
            continue
        print(f"===== {vid} =====")
        inspect(video, targets, cfg, out_dir)


if __name__ == "__main__":
    main()
