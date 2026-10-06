#!/usr/bin/env python3
"""违章01 FP — 结构信号可分性横测 (plan-gate #4 结构版前置, wb, 只读)。

cc 强制门升级版: 不再预锁单一外观/结构信号。同时横测两个候选**结构**信号,
且只用生产真能消费的信号(非 GT/oracle 遮挡):

  (a) 生产遮挡信号 = ViolationEngineV2._is_occluded(mask)
      —— 注意: 它度量的是「斑马线掩膜缺失/过小/被底边截断」(看不到完整斑马线),
         不是「绿车挡住信号灯」。这是 cc 预警的语义陷阱, 必须实测它在 01 假绿段
         到底 fire 不 fire, 以及是否在真绿段也 fire(若也 fire 则降级会屠真绿)。
  (b) YOLO 灯形在场/缺席 = observe 入参 yolo_light_boxes 的数量
      —— 01 窗口 n_yolo_light_boxes=0 是「信号灯被绿车挡住/不在画面」的直接结构证据;
         真绿视频的灯应在画面, YOLO 应框到 ≥1 个灯。

判定规则(cc 指定): 修法#3「遮挡时 prior 直采不得输出 confident green」只在
「该结构信号在 01 假绿段 fire 且 在真绿段(尤其 06 低饱和真绿)不 fire」时才成立。
任一信号在真绿段也 fire → 无 margin → 该路线死; 在 01 假绿段不 fire → 抓不到 FP → 也死。

红线: 只读, 不写生产码, 不建 worktree。禁用 select_gtfree。产物仅诊断输出(gitignored)。

方法: 复用生产入口 cli.run(零循环复制), 仅用只读 monkeypatch 抓逐帧
  - TrafficLightDetector.observe: 记录 n_yolo_light_boxes(=len(yolo_light_boxes))。
  - BatchViolationEngine.accumulate: 记录 (ts, obs, conf, occ_fire=_is_occluded(mask),
    n_yolo_light_boxes, signal_prior)。
  - BatchViolationEngine.decide: 抓引擎实例(_light_obs/_occ_samples)。
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config, project_root
from redlight.app import cli as cli_mod
from redlight.models.traffic_light import TrafficLightDetector
from redlight.pipeline.violation_engine import BatchViolationEngine, ViolationEngineV2
from redlight.pipeline.decision import decide_violations

# ---------- 只读钩子状态 ----------
OBS_LATEST = {}          # 最近一次 observe 归因(前向填充)
ACC_LOG = []             # 每推理帧归因
_ENGINE_REF = {}
_TL_INSTANCES = []

_orig_tl_init = TrafficLightDetector.__init__


def _patched_tl_init(self, cfg, verbose=True):
    _orig_tl_init(self, cfg, verbose)
    _TL_INSTANCES.append(self)


_orig_observe = TrafficLightDetector.observe


def _patched_observe(self, frame, yolo_light_boxes=None):
    res = _orig_observe(self, frame, yolo_light_boxes)
    OBS_LATEST.clear()
    OBS_LATEST.update({
        "n_yolo_light_boxes": len(yolo_light_boxes or []),
        "yolo_light_boxes": [[int(v) for v in b] for b in (yolo_light_boxes or [])],
        "signal_prior": list(self.signal_prior) if self.signal_prior else None,
    })
    return res


_orig_acc = BatchViolationEngine.accumulate


def _patched_accumulate(self, track_states, mask, light_observation, timestamp):
    obs = light_observation.get("obs") if isinstance(light_observation, dict) else None
    conf = light_observation.get("conf") if isinstance(light_observation, dict) else None
    # 生产遮挡信号(真实消费口径): 与 ViolationEngineV2.evaluate 同源
    occ_fire = bool(ViolationEngineV2._is_occluded(mask))
    ACC_LOG.append({
        "ts": round(float(timestamp), 3),
        "obs": obs, "conf": conf,
        "occ_fire": occ_fire,
        "n_yolo_light_boxes": OBS_LATEST.get("n_yolo_light_boxes", 0),
        "yolo_light_boxes": OBS_LATEST.get("yolo_light_boxes", []),
        "signal_prior": OBS_LATEST.get("signal_prior"),
    })
    return _orig_acc(self, track_states, mask, light_observation, timestamp)


_orig_decide = BatchViolationEngine.decide


def _patched_decide(self):
    _ENGINE_REF["eng"] = self
    return _orig_decide(self)


TrafficLightDetector.__init__ = _patched_tl_init
TrafficLightDetector.observe = _patched_observe
BatchViolationEngine.accumulate = _patched_accumulate
BatchViolationEngine.decide = _patched_decide


def _run_one_video(cfg, video, out_dir, preset):
    """跑单个视频, 返回该视频的逐帧诊断行 + 生产事件列表。"""
    OBS_LATEST.clear(); ACC_LOG.clear()
    video_path = os.path.join(project_root(), "input_video", video + ".mp4")
    if not os.path.exists(video_path):
        print(f"[warn] 缺失视频: {video_path}")
        return [], []
    events, _ts = cli_mod.run(cfg, video_path, out_dir, preset,
                              return_track_samples=True)
    rows = [dict(r, video=video) for r in ACC_LOG]
    return rows, events


def _green_label(rows, video, false_video):
    """标注每个 obs=='green' 帧是 false(01) 还是 true(真绿视频)。"""
    out = []
    for r in rows:
        if r["obs"] != "green":
            continue
        label = "false_green" if video == false_video else "true_green"
        out.append({
            "video": video, "ts": r["ts"], "label": label,
            "occ_fire": r["occ_fire"], "n_yolo": r["n_yolo_light_boxes"],
        })
    return out


def _agg(green_rows):
    """按 label 聚合两个结构信号。"""
    res = {}
    for lab in ("false_green", "true_green"):
        sub = [r for r in green_rows if r["label"] == lab]
        n = len(sub)
        if n == 0:
            res[lab] = {"n": 0}
            continue
        occ_true = sum(1 for r in sub if r["occ_fire"])
        yolo0 = sum(1 for r in sub if r["n_yolo"] == 0)
        # 任一 YOLO 灯框在场
        yolo_ge1 = n - yolo0
        res[lab] = {
            "n": n,
            "occ_fire_n": occ_true,
            "occ_fire_frac": round(occ_true / n, 3),
            "yolo0_n": yolo0,
            "yolo0_frac": round(yolo0 / n, 3),
            "yolo_ge1_n": yolo_ge1,
        }
    return res


def main():
    ap = argparse.ArgumentParser(description="结构信号可分性横测 (遮挡 + YOLO 双候选)")
    ap.add_argument("--config", default=os.path.join(project_root(), "configs", "config.yaml"))
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--videos", required=True,
                    help="逗号分隔视频名, 例: 违章01,违章05,违章06,违章07,违章08,违章09")
    ap.add_argument("--false-video", default="违章01",
                    help="已知假绿视频(全程红, 任何绿=误绿)")
    ap.add_argument("--output", default=os.path.join(project_root(), "data", "output", "diag_struct_sep"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.output.annotated_video = False
    cfg.output.evidence_images = False
    cfg.output.csv_report = True

    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    os.makedirs(args.output, exist_ok=True)

    all_rows = []
    per_video_events = {}
    for v in videos:
        out_dir = os.path.join(args.output, "runs", v)
        rows, events = _run_one_video(cfg, v, out_dir, args.preset)
        all_rows.extend(rows)
        per_video_events[v] = events

    green_rows = []
    for v in videos:
        vrows = [r for r in all_rows if r["video"] == v]
        green_rows.extend(_green_label(vrows, v, args.false_video))

    agg = _agg(green_rows)

    # 逐视频分解(真绿视频各自的 occ/yolo, 用于 06 石检验)
    per_video = {}
    for v in videos:
        sub = [r for r in green_rows if r["video"] == v]
        n = len(sub)
        if n:
            per_video[v] = {
                "n_green": n,
                "occ_fire_frac": round(sum(1 for r in sub if r["occ_fire"]) / n, 3),
                "yolo0_frac": round(sum(1 for r in sub if r["n_yolo"] == 0) / n, 3),
            }
        else:
            per_video[v] = {"n_green": 0}

    # 分离度指标
    fg = agg.get("false_green", {})
    tg = agg.get("true_green", {})
    occ_sep = None
    yolo_sep = None
    if fg.get("n") and tg.get("n"):
        occ_sep = round(abs(fg["occ_fire_frac"] - tg["occ_fire_frac"]), 3)
        yolo_sep = round(abs(fg["yolo0_frac"] - tg["yolo0_frac"]), 3)

    report = {
        "videos": videos,
        "false_video": args.false_video,
        "aggregation": agg,
        "per_video": per_video,
        "separability": {
            "occ_signal": {
                "false_green_occ_frac": fg.get("occ_fire_frac"),
                "true_green_occ_frac": tg.get("occ_fire_frac"),
                "abs_gap": occ_sep,
                "verdict": _occ_verdict(fg, tg),
            },
            "yolo_signal": {
                "false_green_yolo0_frac": fg.get("yolo0_frac"),
                "true_green_yolo0_frac": tg.get("yolo0_frac"),
                "abs_gap": yolo_sep,
                "verdict": _yolo_verdict(fg, tg),
            },
        },
        "production_confirmed_events": {
            v: [{"status": e["status"], "start_ts": e.get("start_ts"),
                 "end_ts": e.get("end_ts"), "light_state": e.get("light_state")}
                for e in per_video_events[v] if e.get("status") in ("confirmed", "review")]
            for v in videos
        },
    }

    out_json = os.path.join(args.output, "diag_struct_sep.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台表
    print("=" * 78)
    print("[sep] 结构信号可分性横测 — 仅在 obs=='green' 决策帧上比较")
    print(f"[sep] 候选(a) 生产遮挡 _is_occluded(mask) | 候选(b) YOLO 灯形在场/缺席")
    print("-" * 78)
    print(f"[sep] {'label':<12} {'n_green':>8} {'occ_fire%':>10} {'yolo0%':>8} {'yolo>=1':>8}")
    for lab, d in agg.items():
        if not d.get("n"):
            continue
        print(f"[sep] {lab:<12} {d['n']:>8} {d['occ_fire_frac']*100:>9.1f}% "
              f"{d['yolo0_frac']*100:>7.1f}% {d['yolo_ge1_n']:>8}")
    print("-" * 78)
    print(f"[sep] 逐视频(真绿) occ/yolo0 占比(06 石重点看):")
    for v, d in per_video.items():
        if d["n_green"]:
            print(f"[sep]   {v:<8} n_green={d['n_green']:>4} "
                  f"occ_fire={d['occ_fire_frac']*100:>5.1f}% "
                  f"yolo0={d['yolo0_frac']*100:>5.1f}%")
        else:
            print(f"[sep]   {v:<8} (无 obs==green 帧)")
    print("-" * 78)
    sv = report["separability"]
    print(f"[sep] 候选(a) 遮挡: 假绿 occ={sv['occ_signal']['false_green_occ_frac']} "
          f"真绿 occ={sv['occ_signal']['true_green_occ_frac']} gap={occ_sep} "
          f"=> {sv['occ_signal']['verdict']}")
    print(f"[sep] 候选(b) YOLO: 假绿 yolo0={sv['yolo_signal']['false_green_yolo0_frac']} "
          f"真绿 yolo0={sv['yolo_signal']['true_green_yolo0_frac']} gap={yolo_sep} "
          f"=> {sv['yolo_signal']['verdict']}")
    print("=" * 78)
    print(f"[sep] 产物: {out_json}")


def _occ_verdict(fg, tg):
    if not (fg.get("n") and tg.get("n")):
        return "数据不足"
    f, t = fg["occ_fire_frac"], tg["occ_fire_frac"]
    # 死法1: 假绿段不 fire → 抓不到 FP
    if f < 0.5:
        return "死: 假绿段不fire(抓不到FP)"
    # 死法2: 真绿段也大面积 fire → 无 margin, 降级会屠真绿(尤其06)
    if t >= 0.5:
        return "死: 真绿段也fire(无margin, 降级屠真绿)"
    # 中间: gap 越大越干净
    return f"活: 假绿fire且真绿不fire (gap={round(abs(f-t),3)})"


def _yolo_verdict(fg, tg):
    if not (fg.get("n") and tg.get("n")):
        return "数据不足"
    f, t = fg["yolo0_frac"], tg["yolo0_frac"]
    # 假绿段期望 yolo0 高(无灯框); 真绿段期望 yolo0 低(有灯框)
    if f < 0.5:
        return "死: 假绿段也有灯框(抓不到FP)"
    if t >= 0.5:
        return "死: 真绿段也无灯框(无margin, 降级屠真绿)"
    return f"活: 假绿无灯框且真绿有灯框 (gap={round(abs(f-t),3)})"


if __name__ == "__main__":
    main()
