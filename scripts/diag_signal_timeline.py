"""诊断脚本 (2026-07-11): 打开"0 违规"黑盒。

逐帧(与 cli 同 interval)跑 车辆/斑马线/信号灯 三个检测器, 记录时序:
  ts, light_state, light_reason, n_active, n_stationary,
  max_overlap, n_on_crosswalk_stationary, crosswalk_present, crosswalk_occluded

输出每视频一个 CSV + 汇总直方图, 用于判断 0 违规到底是:
  (a) 真没有  (b) 信号灯检测器几乎看不到灯  (c) 斑马线/压线判定失效

用法:
    python scripts/diag_signal_timeline.py
"""
import sys
import os
import csv

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np

from redlight.infrastructure.config import load_config, project_root
from redlight.models.vehicle import VehicleDetector
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.traffic_light import TrafficLightDetector
from redlight.pipeline.tracker import TrackStateManagerV2, SENSITIVITY_PRESETS
from redlight.pipeline.violation_engine import ViolationEngineV2
from redlight.infrastructure.geometry import compute_overlap_ratio


def diagnose(video, out_csv, cfg, preset="balanced"):
    overlap_thr = SENSITIVITY_PRESETS[preset]["overlap"]
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[SKIP] 无法打开 {video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    interval = max(1, int(round(fps / cfg.inference.fps)))

    det = VehicleDetector(cfg, verbose=False)
    cw = CrosswalkDetector(cfg)
    tl = TrafficLightDetector(cfg, verbose=False)
    track = TrackStateManagerV2(preset)

    rows = []
    frame_idx = 0
    proc = 0
    light_hist = {}
    reason_hist = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            proc += 1
            dets = det.detect(frame)
            states = track.update(dets, frame_idx / fps)
            mask = cw.detect(frame)
            lres = tl.detect(frame)
            lstate = lres.get("state", "unknown") if isinstance(lres, dict) else lres
            lreason = lres.get("reason", "") if isinstance(lres, dict) else ""
            light_hist[lstate] = light_hist.get(lstate, 0) + 1
            if lreason:
                reason_hist[lreason] = reason_hist.get(lreason, 0) + 1
            n_active = sum(1 for s in states.values() if s.get("active"))
            n_stat = sum(1 for s in states.values() if s.get("active") and s.get("stationary"))
            max_ov = 0.0
            n_ocs = 0
            for s in states.values():
                if not s.get("active"):
                    continue
                ov = compute_overlap_ratio(s["box"], mask)
                max_ov = max(max_ov, ov)
                if s.get("stationary") and ov >= overlap_thr:
                    n_ocs += 1
            cw_present = mask is not None
            cw_occ = ViolationEngineV2._is_occluded(mask)
            rows.append({
                "ts": round(frame_idx / fps, 2),
                "light_state": lstate,
                "light_reason": lreason,
                "n_active": n_active,
                "n_stationary": n_stat,
                "max_overlap": round(max_ov, 3),
                "n_on_crosswalk_stationary": n_ocs,
                "crosswalk_present": int(cw_present),
                "crosswalk_occluded": int(cw_occ),
            })
        frame_idx += 1

    cap.release()
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    # 汇总: 各 light_state 下, 出现"静止压线"的帧数
    by_state = {}
    for r in rows:
        st = r["light_state"]
        by_state.setdefault(st, {"frames": 0, "ocs_frames": 0, "cw_present": 0, "cw_occ": 0})
        by_state[st]["frames"] += 1
        by_state[st]["ocs_frames"] += (1 if r["n_on_crosswalk_stationary"] > 0 else 0)
        by_state[st]["cw_present"] += r["crosswalk_present"]
        by_state[st]["cw_occ"] += r["crosswalk_occluded"]

    print(f"\n===== {os.path.basename(video)} =====")
    print(f"总帧={total} 推理帧={proc} fps={fps:.1f} interval={interval} overlap_thr={overlap_thr}")
    print("信号灯状态直方图 (帧数):", dict(sorted(light_hist.items(), key=lambda x: -x[1])))
    print("信号灯 reason 直方图:", dict(sorted(reason_hist.items(), key=lambda x: -x[1])))
    print("各状态下: 帧数 / 出现静止压线帧数 / 斑马线可见帧 / 斑马线遮挡帧:")
    for st, d in sorted(by_state.items(), key=lambda x: -x[1]["frames"]):
        print(f"  {st:10s} frames={d['frames']:4d}  ocs_frames={d['ocs_frames']:4d}  "
              f"cw_present={d['cw_present']:4d}  cw_occ={d['cw_occ']:4d}")
    print(f"[CSV] {out_csv}")


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    base_in = r"E:\BaiduNetdiskDownload"
    base_out = os.path.join(project_root(), "data", "output")
    for fname in ("违章01.mp4", "违章02.mp4"):
        video = os.path.join(base_in, fname)
        if not os.path.exists(video):
            print(f"[SKIP] {video}")
            continue
        out_csv = os.path.join(base_out, f"diag_{os.path.splitext(fname)[0]}_timeline.csv")
        diagnose(video, out_csv, cfg, "balanced")


if __name__ == "__main__":
    main()
