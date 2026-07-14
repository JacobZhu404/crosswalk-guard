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
import json

import cv2

from ..infrastructure.config import load_config, ensure_dir, project_root
from ..models.vehicle import VehicleDetector
from ..models.crosswalk import CrosswalkDetector
from ..models.traffic_light import TrafficLightDetector
from ..models.plate import PlateRecognizer
from ..pipeline.tracker import TrackStateManagerV2
from ..pipeline.violation_engine import BatchViolationEngine
from ..pipeline.visualizer import Visualizer
from ..pipeline.dag import build_default_dag
from ..pipeline.plate_consensus import PlateConsensus
from ..pipeline.analysis import AnalysisAccumulator, CotReporter


def run(cfg, video_path, output_dir, preset="balanced", cot=False):
    ensure_dir(output_dir)
    evidence_dir = os.path.join(output_dir, "evidence")
    ensure_dir(evidence_dir)
    cot_dir = os.path.join(output_dir, "cot") if cot else None
    if cot:
        ensure_dir(cot_dir)

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

    # ②③ 接线: fuse_light 参数来自 cfg.traffic_light
    tl_cfg = getattr(cfg, "traffic_light", None)
    fuse_kwargs = {
        "window": int(getattr(tl_cfg, "smoothing_window", 24)),
        "hysteresis": float(getattr(tl_cfg, "hysteresis", 0.68)),
        "flicker_toggle": int(getattr(tl_cfg, "flicker_toggle_count", 4)),
        "unknown_hold": int(getattr(tl_cfg, "anchor_hold", 30)),
    }
    comp = {
        "vehicle": VehicleDetector(cfg),
        "crosswalk": CrosswalkDetector(cfg),
        "light": TrafficLightDetector(cfg),
        "plate": PlateRecognizer(cfg),
        "trackstate": TrackStateManagerV2(preset),
        "engine": BatchViolationEngine(
            preset=preset,
            sample_fps=cfg.inference.fps,
            unknown_to_review=cfg.output.unknown_light_to_review,
            fuse_kwargs=fuse_kwargs,
        ),
        "viz": Visualizer(cfg, preset=preset),
        "plate_consensus": PlateConsensus(keep_history=180),
    }
    # 按 video 名加载 per-video 行人信号位置先验 (light_priors.json), 接入 observe() prior 直采。
    # 命中 -> observe 用先验 ROI 直采排除环境绿/树叶干扰; 未命中 -> 回退全局亮斑 (保持原行为)。
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if comp["light"].set_video_prior(video_name):
        print(f"[红绿灯] 已加载视频先验 {video_name}: prior={comp['light'].signal_prior} roi={comp['light'].prior_roi_px}px")
    dag = build_default_dag(cfg, comp)

    # COT 可解释分析累积器 (设计需求 v2 §4): 增量累积中间态, 循环结束后渲染小作文+截图
    acc = None
    if cot:
        acc = AnalysisAccumulator(video_name, fps, (total / fps) if total else 0.0)

        def n_record(ctx):
            light = ctx.get("light")
            lstate = ctx.get("light_state", "unknown")
            lconf = light.get("confidence", 0.0) if isinstance(light, dict) else 0.0
            acc.on_frame(ctx["ts"], lstate, lconf, ctx.get("states", {}),
                         ctx.get("mask"), ctx.get("consensus_plates", {}),
                         ctx.get("frame"), ctx.get("dets", []))
            # on_event 延迟到视频结束后统一调用(批处理决策完成后)

        dag.add_node("record", n_record)
        dag.add_edge("accumulate", "record")
        dag.add_edge("record", "visualize")

    ctx = {
        "proc": 0, "frame": None, "ts": 0.0, "dets": [], "states": {},
        "mask": None, "light": "unknown", "light_state": "unknown", "plates": [],
        "video_writer": video_writer, "evidence_dir": evidence_dir,
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

    # ②③ 接线: 视频结束后批处理决策
    events = comp["engine"].decide()

    # 证据截图 + CSV 输出(批处理后)
    _write_outputs(events, video_path, evidence_dir, output_dir, cfg,
                   comp.get("plate_consensus"))

    cot_path = None
    if cot and acc is not None:
        # 批处理完成后统一回填事件到 COT
        for ev in events:
            acc.on_event(ev)
        analysis = acc.build()
        # 组装截图素材(灯态证据帧 / 占用峰值帧 / 车牌清晰帧)
        frames = {
            "light": {s: fr for s, (_, fr) in acc.light_evidence.items()},
            "tracks": {},
        }
        for tid, (ratio, fr, box, mask) in acc.occ_peak.items():
            frames["tracks"].setdefault(tid, {})["occ"] = fr
        for tid, (conf, fr, pb, text) in acc.plate_best.items():
            frames["tracks"].setdefault(tid, {})["plate"] = fr
        reporter = CotReporter(cfg)
        cot_path, _ = reporter.render(analysis, cot_dir, frames)
        # 同时落盘结构化中间态 JSON(便于后续模块化评测/回溯)
        with open(os.path.join(cot_dir, f"analysis_{analysis['video']}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    confirmed = sum(1 for e in events if e["status"] == "confirmed")
    review = sum(1 for e in events if e["status"] == "review")
    print(f"[完成] 帧数={frame_idx} 推理帧={ctx['proc']} 耗时={elapsed:.1f}s "
          f"({ctx['proc'] / max(elapsed, 1e-3):.1f} 推理帧/秒)")
    print(f"[结果] 确认违规={confirmed}  待复核={review}  preset={preset}")
    print(f"[输出] {os.path.abspath(output_dir)}")
    if cot_path:
        print(f"[COT] 可解释报告: {os.path.abspath(cot_path)}")
    return events


def _write_outputs(events, video_path, evidence_dir, output_dir, cfg, consensus):
    """批处理后生成 CSV 与证据截图。"""
    # 收集车牌
    consensus_plates = consensus.get_all() if consensus else {}

    # 证据截图: 重新打开视频 seek 到事件 start_ts
    if cfg.output.evidence_images and events:
        cap2 = cv2.VideoCapture(video_path)
        for ev in events:
            ts = ev["start_ts"]
            cap2.set(cv2.CAP_PROP_POS_MSEC, int(ts * 1000))
            ret, frame = cap2.read()
            if not ret:
                continue
            plate_text = ""
            tid = ev["track_id"]
            if tid in consensus_plates:
                plate_text = consensus_plates[tid]["text"]
            fname = f"ev{ev['event_id']:04d}_tid{tid}"
            if plate_text:
                fname += f"_{plate_text}"
            fname += ".jpg"
            fpath = os.path.join(evidence_dir, fname)
            cv2.imwrite(fpath, frame)
            ev["evidence_image"] = fpath
            ev["plate"] = plate_text
        cap2.release()

    if cfg.output.csv_report:
        csv_path = os.path.join(output_dir, "violations.csv")
        cols = ["event_id", "track_id", "status", "start_ts", "end_ts",
                "vehicle_class", "confidence", "light_state", "signal_assumption",
                "plate", "evidence_image"]
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for ev in events:
                w.writerow({
                    "event_id": ev["event_id"], "track_id": ev["track_id"],
                    "status": ev["status"], "start_ts": ev["start_ts"],
                    "end_ts": ev["end_ts"], "vehicle_class": ev.get("vehicle_class", ""),
                    "confidence": ev.get("confidence", 0.0),
                    "light_state": ev["light_state"],
                    "signal_assumption": cfg.output.signal_assumption,
                    "plate": ev.get("plate", ""),
                    "evidence_image": ev.get("evidence_image", ""),
                })


def main():
    ap = argparse.ArgumentParser(description="斑马线行人绿灯压线检测 (v2.0 分层架构)")
    ap.add_argument("--config", default=os.path.join(project_root(), "configs", "config.yaml"))
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--output", default=None, help="输出目录 (默认 data/output/run_<name>)")
    ap.add_argument("--preset", default="balanced",
                    choices=["strict", "balanced", "loose", "very_loose"],
                    help="违规判定灵敏度预设")
    ap.add_argument("--cot", action="store_true",
                    help="输出 COT 可解释小作文+截图 (设计需求 v2 §4)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output is None:
        name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(project_root(), "data", "output", f"run_{name}_{args.preset}")
    run(cfg, args.video, args.output, args.preset, cot=args.cot)


if __name__ == "__main__":
    main()
