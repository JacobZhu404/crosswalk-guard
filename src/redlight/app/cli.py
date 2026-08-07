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
from ..models.crosswalk_v2 import CrosswalkDetectorV2
from ..models.traffic_light import TrafficLightDetector
from ..models.plate import PlateRecognizer
from ..pipeline.tracker import TrackStateManagerV2
from ..pipeline.violation_engine import BatchViolationEngine
from ..pipeline.visualizer import Visualizer
from ..pipeline.dag import build_default_dag
from ..pipeline.plate_consensus import PlateConsensus
from ..pipeline.analysis import AnalysisAccumulator, CotReporter
from ..evaluation.metrics import levenshtein


def run(cfg, video_path, output_dir, preset="balanced", cot=False, return_track_samples=False,
        crosswalk_detector=None, occ_denom=None, min_persistent_green_run_s=None):
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
    # #3 时序门控 (plan-gate #5): 持久绿阈值, 段内最长 raw 绿 run < 此值 -> 降级 review(非 confirmed)
    # 显式传参(诊断/评测注入, 如 G1-G4 门验证)优先于 config 默认值
    _cfg_min_run = float(getattr(tl_cfg, "min_persistent_green_run_s", 6.0))
    _min_run = float(min_persistent_green_run_s) if min_persistent_green_run_s is not None else _cfg_min_run
    # 接线(C3, Jacob 拍板 2026-08-04): 默认 occ_denom 从 config 读(接线后默认 "box");
    # 显式传参仍优先(诊断/评测注入路径不变, 见 eval_violations/sweep/diag 显式传参)。
    _occ_denom = occ_denom if occ_denom is not None else getattr(cfg.crosswalk, "occ_denom", "mask")
    _engine_kwargs = dict(
        preset=preset,
        sample_fps=cfg.inference.fps,
        unknown_to_review=cfg.output.unknown_light_to_review,
        fuse_kwargs=fuse_kwargs,
        occ_denom=_occ_denom,
        min_persistent_green_run_s=_min_run,
    )
    # 接线(C3): 默认检测器从 config 读(接线后默认 "v2"); 显式传参优先。
    _cw_version = getattr(cfg.crosswalk, "version", "v11")
    _crosswalk_det = crosswalk_detector if crosswalk_detector is not None else (
        CrosswalkDetectorV2(cfg) if _cw_version == "v2" else CrosswalkDetector(cfg))
    comp = {
        "vehicle": VehicleDetector(cfg),
        "crosswalk": _crosswalk_det,
        "light": TrafficLightDetector(cfg),
        "plate": PlateRecognizer(cfg),
        "trackstate": TrackStateManagerV2(preset),
        "engine": BatchViolationEngine(**_engine_kwargs),
        "viz": Visualizer(cfg, preset=preset, occ_denom=_occ_denom),
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
                   comp.get("plate_consensus"), comp["engine"]._track_samples,
                   plate_recog=comp["plate"])

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
    if return_track_samples:
        # 纯加法: 仅返回引擎已逐帧累积的 track 样本(tid -> [{ts,stationary,box,...}]),
        # 不触发任何额外计算、不改变判定/输出行为。默认 False 时返回值与旧版完全一致。
        return events, comp["engine"]._track_samples
    return events


def _episode_plate(consensus, ev, track_samples=None):
    """从 episode 的 member_tracks 按"违章车组+全局真实性+空间聚集"三重约束回填主车牌。

    修复(2026-08-05, cc 效果 gate 送回 f26c91f 后 v7):
      旧逻辑(2026-07-16 扩展, 遍历 member_tracks 取 weight 最高)被 b2 过合并污染——
      误并进 episode 的过路/别车 track 可夺魁 → 开错罚单(违章02 京A14672 / 违章05 京N541E6)。
      约束见 _constrained_agg; 主牌 = 约束后 weight 最高, 无候选 → 空串(宁缺毋滥)。
    """
    main, _ = _episode_plate_all(consensus, ev, track_samples)
    return main


def _episode_plate_all(consensus, ev, track_samples=None):
    """三重约束回填(共享内层): 返回 (主牌, 多牌列表[ED<=1 变体系去重, weight 降序])。

    约束:
      (1) 违章车组: 候选归属 tid 在事件窗口内 stationary 占比 >= 0.6(挡过路/移动车);
      (2) 全局真实性: 候选 text 全视频读取帧数 >= 5(挡孤证/幻觉牌);
      (3) 空间聚集: 候选"ED<=1 变体系"归属 tid(窗口内)质心 x 范围 <= 0.25×帧宽
          (挡跨车关联污染: 05 京N541E6 系 607px; 真牌同车碎片 07 京Q5D2N8 69px)。
      时间语义: 牌读取帧可落在事件窗口外(产品语义"车牌需视频全局读取")。
      兼容: consensus 为 None / dict 旧接口 / 无 track_records / 无 track_samples → 回退旧聚合逻辑。
    """
    tids = list(dict.fromkeys([ev["track_id"]] + list(ev.get("member_tracks", []))))
    if consensus is None:
        return "", []
    if isinstance(consensus, dict):
        # 旧接口兼容(dict: tid -> {text, weight}, 2026-07-16 行为): 按聚合 weight 选最高
        best_text, best_w = "", -1.0
        for t in tids:
            p = consensus.get(t)
            if p and p.get("text") and p.get("weight", 0.0) > best_w:
                best_text, best_w = p["text"], p.get("weight", 0.0)
        return best_text, ([best_text] if best_text else [])
    records = getattr(consensus, "track_records", None)
    if not records or not track_samples:
        # 回退旧逻辑: 按聚合 weight 选最高
        cp = consensus.get_all()
        best_text, best_w = "", -1.0
        for t in tids:
            p = cp.get(t)
            if p and p.get("text") and p.get("weight", 0.0) > best_w:
                best_text, best_w = p["text"], p.get("weight", 0.0)
        return best_text, ([best_text] if best_text else [])

    STATIONARY_RATIO = 0.6
    GLOBAL_MIN_FRAMES = 5
    SPAN_RATIO = 0.25  # 质心 x 范围占帧宽比例上限
    eps = 1.0
    t0, t1 = ev["start_ts"] - eps, ev["end_ts"] + eps

    # 帧宽: 从 track_samples 全量 box 推(统一 1920 场景; 防 0 除)
    frame_w = max((s["box"][2] for _tid in track_samples for s in track_samples[_tid]), default=1920) or 1920

    # 窗口内各 tid 的 stationary 占比 + 质心
    sta_ratio, tid_cx = {}, {}
    for tid in tids:
        samples = [s for s in track_samples.get(tid, []) if t0 <= s["ts"] <= t1]
        if samples:
            sta_ratio[tid] = sum(1 for s in samples if s.get("stationary")) / len(samples)
            tid_cx[tid] = sum((s["box"][0] + s["box"][2]) / 2 for s in samples) / len(samples)

    # 全局真实性: 候选 text 在 consensus 全量(所有 tid)records 中的帧数
    global_count = {}
    for _tid, _recs in records.items():
        for _r in _recs:
            global_count[_r["text"]] = global_count.get(_r["text"], 0) + 1

    # 回填聚合: text -> conf 总和(weight = count×avg_conf = conf 总和)
    # P1: 同 tid(同一车)只取 weight 最高 text(一车一牌, 挡同车误读变体,
    # 如违章08 tid92 上 京ACW6553(13帧) vs 京J00542(误读) 互斥取前者)。
    agg = {}
    for tid in tids:
        if sta_ratio.get(tid, 0.0) < STATIONARY_RATIO:
            continue  # 非窗口内违章车组(过路/移动/窗口外车), 其牌不参与回填
        tid_agg = {}
        for rec in records.get(tid, []):
            text = rec["text"]
            if global_count.get(text, 0) < GLOBAL_MIN_FRAMES:
                continue  # 孤证/幻觉牌, 宁缺毋滥
            tid_agg[text] = tid_agg.get(text, 0.0) + rec.get("conf", 0.0)
        if tid_agg:
            best_text = max(tid_agg, key=tid_agg.get)  # 一车一牌: tid 内取 weight 最高
            agg[best_text] = agg.get(best_text, 0.0) + tid_agg[best_text]

    # 空间聚集: 牌的"ED<=1 变体系"归属 tid(窗口内)质心 x 范围 <= 0.25×帧宽,
    # 否则判为跨车关联污染(真牌同车碎片聚集: 07 京Q5D2N8 69px / 09 京AC63971 221px;
    # 污染系跨车分散: 05 京N541E6 系 607px, 含变体 京N541E61 等 —— 按系合并挡, 防变体绕过)。
    text_tid_span = {}
    for text in agg:
        tset = {tid for tid, _recs in records.items()
                if any(levenshtein(r["text"], text) <= 1 for r in _recs)}
        cxs = [tid_cx[t] for t in tset if t in tid_cx]
        if cxs:
            text_tid_span[text] = max(cxs) - min(cxs)
    agg = {t: w for t, w in agg.items()
           if t not in text_tid_span or text_tid_span[t] <= SPAN_RATIO * frame_w}
    return _pick_plates(agg, global_count, ev["track_id"], records)


def _pick_plates(agg, global_count=None, rep_tid=None, records=None):
    """从 P2 约束后的候选(agg: text->weight)选主牌与多牌列表(P1)。

    主牌 = weight 最高(兼容 ev["plate"]); 次牌须同时满足:
      - 全局帧数 >= 10(挡低帧过路车: 08 京PK9B77 6帧 / 06 京WPM966 6帧);
      - 与主牌 ED > 1(非主牌变体);
      - 归属 tid 不含代表 track(代表车的牌应为主牌; 挂在代表 track 上的其他牌=
        代表车误读/污染, 如 03 京FJQ279 挂代表 tid103 → 挡; 而多车事件第二违章车
        08 京ACW6553 tid92 / 09 京NNM526 tid99 均非代表 → 保留)。
      至多 2 个次牌。空 -> ("", [])。
    """
    if not agg:
        return "", []
    main_text = max(agg, key=lambda k: agg[k])
    plates = [main_text]
    if global_count and records is not None:
        for t in sorted(agg, key=lambda k: -agg[k]):
            if t == main_text or levenshtein(t, main_text) <= 1:
                continue
            if global_count.get(t, 0) < 10:
                continue
            t_tids = {tid for tid, _recs in records.items()
                      if any(r["text"] == t for r in _recs)}
            if rep_tid in t_tids:
                continue  # 代表车上的其他牌 = 误读/污染, 不作次牌
            plates.append(t)
            if len(plates) >= 3:
                break
    return main_text, plates


def _p3_roi_retry(video_path, ev, track_samples, plate_recog):
    """P3: 代表车全图读牌为空时, 事件窗口内车框 ROI 放大 2x 重试 HyperLPR3。

    目标: 低曝光违章车牌(违章02 京LNE560 / 违章05 京ADH9206, 全图检测读不出)。
    安全: ROI 限定代表 track 车框(读的是代表车自己的牌, 非别车), conf>=0.6 + 格式校验。
    返回 text 或 ""。
    """
    if plate_recog is None or not getattr(plate_recog, "use_hl", False):
        return ""
    rep = ev["track_id"]
    samples = [s for s in (track_samples or {}).get(rep, [])
               if ev["start_ts"] <= s["ts"] <= ev["end_ts"]]
    if not samples:
        return ""
    from ..models.plate import _is_valid_plate
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""
    best_text, best_conf = "", 0.0
    mid = (ev["start_ts"] + ev["end_ts"]) / 2.0
    for dt in (-3.0, 0.0, 3.0):
        ts = max(0.0, mid + dt)
        cap.set(cv2.CAP_PROP_POS_MSEC, int(ts * 1000))
        ret, frame = cap.read()
        if not ret:
            continue
        nb = min(samples, key=lambda s: abs(s["ts"] - ts))["box"]
        x1, y1, x2, y2 = [int(v) for v in nb]
        H, W = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - 0.2 * bw)), max(0, int(y1 - 0.2 * bh))
        cx2, cy2 = min(W, int(x2 + 0.2 * bw)), min(H, int(y2 + 0.2 * bh))
        roi = frame[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            continue
        roi2 = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        for p in plate_recog.detect(roi2):
            t, c = p.get("text", ""), p.get("conf", 0.0)
            if t and _is_valid_plate(t) and c >= 0.6 and c > best_conf:
                best_text, best_conf = t, c
    cap.release()
    return best_text


def _write_outputs(events, video_path, evidence_dir, output_dir, cfg, consensus,
                   track_samples=None, plate_recog=None):
    """批处理后生成 CSV 与证据截图。"""
    # 收集车牌: 按 episode 的 member_tracks 回填(不依赖 evidence 开关)
    for ev in events:
        ev["plate"], ev["plates"] = _episode_plate_all(consensus, ev, track_samples)
        if not ev["plate"]:
            # P3: 代表车全图读牌为空 -> 事件窗口内 ROI 放大重试(低曝光车牌)
            retry = _p3_roi_retry(video_path, ev, track_samples, plate_recog)
            if retry:
                ev["plate"], ev["plates"] = retry, [retry]

    # 证据截图: 重新打开视频 seek 到事件 start_ts
    if cfg.output.evidence_images and events:
        cap2 = cv2.VideoCapture(video_path)
        for ev in events:
            ts = ev["start_ts"]
            cap2.set(cv2.CAP_PROP_POS_MSEC, int(ts * 1000))
            ret, frame = cap2.read()
            if not ret:
                continue
            tid = ev["track_id"]
            fname = f"ev{ev['event_id']:04d}_tid{tid}"
            if ev["plate"]:
                fname += f"_{ev['plate']}"
            fname += ".jpg"
            fpath = os.path.join(evidence_dir, fname)
            cv2.imwrite(fpath, frame)
            ev["evidence_image"] = fpath
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
