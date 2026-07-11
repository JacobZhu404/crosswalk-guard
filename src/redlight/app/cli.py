"""L6 应用层: CLI 入口与视频处理主循环。

运行方式 (工程根目录):
    python -m redlight.app.cli --video <mp4> --output <dir> --preset balanced
或简用:
    python scripts/run_video.py <mp4> <dir> [--preset balanced]
"""
import os
import sys
import argparse
import time
import csv

import cv2

from ..infrastructure.config import load_config, ensure_dir, project_root
from ..models.vehicle import VehicleDetector
from ..models.crosswalk import CrosswalkDetector
from ..models.traffic_light import TrafficLightDetector
from ..models.plate import PlateRecognizer
from ..pipeline.tracker import TrackStateManagerV2
from ..pipeline.violation_engine import ViolationEngineV2
from ..pipeline.visualizer import Visualizer
from ..pipeline.dag import build_default_dag
from ..pipeline.plate_consensus import PlateConsensus


def run(cfg, video_path, output_dir, preset="balanced", mode="red_light"):
    ensure_dir(output_dir)
    evidence_dir = os.path.join(output_dir, "evidence")
    ensure_dir(evidence_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    interval = max(1, int(round(fps / cfg.inference.fps)))

    video_writer = None
    if cfg.output.annotated_video:
        out_path = os.path.join(output_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    comp = {
        "vehicle": VehicleDetector(cfg),
        "crosswalk": CrosswalkDetector(cfg),
        "light": TrafficLightDetector(cfg),
        "plate": PlateRecognizer(cfg),
        "trackstate": TrackStateManagerV2(preset),
        "engine": ViolationEngineV2(
            preset, unknown_to_review=cfg.output.unknown_light_to_review, mode=mode),
        "viz": Visualizer(cfg),
    }
    dag = build_default_dag(cfg, comp)

    ctx = {
        "proc": 0, "frame": None, "ts": 0.0, "dets": [], "states": {},
        "mask": None, "light": "unknown", "light_state": "unknown", "plates": [], "new_events": [],
        "csv_rows": [], "video_writer": video_writer, "evidence_dir": evidence_dir,
        "cfg": cfg, "disp": None,
    }
    last_disp = None
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        ctx["frame"] = frame
        ctx["ts"] = ts
        if frame_idx % interval == 0:
            ctx["proc"] += 1
            dag.run(ctx)
            last_disp = ctx.get("disp")
        if video_writer is not None:
            video_writer.write(last_disp if last_disp is not None else frame)
        frame_idx += 1

    cap.release()
    if video_writer:
        video_writer.release()

    if cfg.output.csv_report:
        csv_path = os.path.join(output_dir, "violations.csv")
        cols = ["event_id", "track_id", "status", "start_ts", "end_ts",
                "vehicle_class", "confidence", "light_state", "signal_assumption",
                "plate", "evidence_image"]
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in ctx["csv_rows"]:
                w.writerow(row)

    elapsed = time.time() - t0
    events = comp["engine"].events
    confirmed = sum(1 for e in events if e["status"] == "confirmed")
    review = sum(1 for e in events if e["status"] == "review")
    print(f"[完成] 帧数={frame_idx} 推理帧={ctx['proc']} 耗时={elapsed:.1f}s "
          f"({ctx['proc'] / max(elapsed, 1e-3):.1f} 推理帧/秒)")
    print(f"[结果] 确认违规={confirmed}  待复核={review}  preset={preset}")
    print(f"[输出] {os.path.abspath(output_dir)}")
    return events


def main():
    ap = argparse.ArgumentParser(description="斑马线行人绿灯压线检测 (v2.0 分层架构)")
    ap.add_argument("--config", default=os.path.join(project_root(), "configs", "config.yaml"))
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--output", default=None, help="输出目录 (默认 data/output/run_<name>)")
    ap.add_argument("--preset", default="balanced",
                    choices=["strict", "balanced", "loose"],
                    help="违规判定灵敏度预设")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output is None:
        name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(project_root(), "data", "output", f"run_{name}_{args.preset}")
    run(cfg, args.video, args.output, args.preset)


if __name__ == "__main__":
    main()
