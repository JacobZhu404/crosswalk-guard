"""L7 可解释分析层 (设计需求 v2 §4/§5):

把规则系统的**结构化中间态**渲染为可审阅的 COT(Chain-of-Thought) 小作文 + 截图,
便于定位"哪一步出错"(VQA 式事后可解释, 不引入 LLM, 确定性模板, 零外部依赖).

数据流:
  analyze_video() 在每个推理帧累积:
    - light_timeline: [(ts, state, conf)]  逐帧灯态
    - track_records:  tid -> {class, occ_peak/occ_first/occ_last/occ_peak_box,
                               plate_reads:[(ts,text,conf)], violation}
    - events:         违规引擎事件
  build_analysis()   折叠为 AnalysisResult (中间态 JSON)
  CotReporter.render() 渲染 COT_xxx.md + 截图目录

截图策略 (设计需求 v2 §5.4):
  ① 灯态证据帧: 每个不同灯态段取一帧(画灯候选框)
  ② 占用峰值帧: 每违规/占道车 track 上 overlap 最大那帧(画车框+斑马线掩膜)
  ③ 车牌清晰帧: 该车 track 上 OCR 置信最高那帧(画车牌框+文字)
"""
import os
import csv
import json
import time
import collections

import cv2
import numpy as np

from ..infrastructure.config import ensure_dir
from ..infrastructure.geometry import compute_overlap_ratio


def fmt_ts(sec):
    """秒 -> mm:ss 便于 COT 阅读。"""
    if sec is None:
        return "??:??"
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


class AnalysisAccumulator:
    """在主循环里增量累积中间态。

    调用方(analyze_video)在每个推理帧调用:
        acc.on_frame(ts, light_state, light_conf, states, mask, consensus_plates, frame, dets)
    并在违规事件产生时调用 acc.on_event(ev)。
    """

    def __init__(self, video_name, fps, duration):
        self.video = video_name
        self.fps = fps
        self.duration = duration
        self.light_timeline = []          # (ts, state, conf)
        self.track_records = {}            # tid -> record
        self.events = []
        # 截图缓存: 灯态证据帧 / 占用峰值帧 / 车牌清晰帧
        self.light_evidence = {}          # state -> (ts, frame_bgr)
        self.occ_peak = {}                # tid -> (ratio, frame_bgr, box, mask)
        self.plate_best = {}              # tid -> (conf, frame_bgr, plate_box, text)

    # ---- 每推理帧调用 ----
    def on_frame(self, ts, light_state, light_conf, states, mask, consensus_plates,
                 frame, dets):
        self.light_timeline.append((ts, light_state, light_conf))
        # 灯态证据帧(每状态取首个)
        if light_state not in self.light_evidence and frame is not None:
            self.light_evidence[light_state] = (ts, frame.copy())
        # 每车占道 / 车牌
        for tid, st in states.items():
            if not st.get("active"):
                continue
            rec = self.track_records.setdefault(tid, {
                "tid": tid, "vehicle_class": st.get("vehicle_class", "car"),
                "occ_peak": 0.0, "occ_first": None, "occ_last": None,
                "occ_peak_box": None, "plate_reads": [], "violation": None,
            })
            box = st.get("box")
            if box is not None and mask is not None:
                ratio = compute_overlap_ratio(box, mask, footprint=0.5, denom="mask")
                if ratio > 0.02:   # 有实质性占道才记录
                    if rec["occ_first"] is None:
                        rec["occ_first"] = ts
                    rec["occ_last"] = ts
                    if ratio > rec["occ_peak"]:
                        rec["occ_peak"] = ratio
                        rec["occ_peak_box"] = list(box)
                        if frame is not None and mask is not None:
                            self.occ_peak[tid] = (ratio, frame.copy(), list(box), mask.copy())
            # 车牌读取(全局聚合, 见 E19)
            text = consensus_plates.get(tid, {}).get("text") if consensus_plates else None
            conf = consensus_plates.get(tid, {}).get("conf", 0.0) if consensus_plates else 0.0
            if text:
                rec["plate_reads"].append((ts, text, conf))
                if frame is not None and conf >= self.plate_best.get(tid, (0,))[0]:
                    # 找该车车牌框(从 dets 里匹配 track box 重叠的 plate)
                    pb = self._find_plate_box(dets, box)
                    self.plate_best[tid] = (conf, frame.copy(),
                                            list(pb) if pb else None, text)

    @staticmethod
    def _find_plate_box(dets, veh_box):
        if not veh_box:
            return None
        vx1, vy1, vx2, vy2 = veh_box
        best, biou = None, 0.0
        for d in dets:
            pb = d.get("xyxy")
            if not pb:
                continue
            px1, py1, px2, py2 = pb
            ix1, iy1 = max(px1, vx1), max(py1, vy1)
            ix2, iy2 = min(px2, vx2), min(py2, vy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                continue
            pa = (px2 - px1) * (py2 - py1)
            va = (vx2 - vx1) * (vy2 - vy1)
            iou = inter / min(pa, va) if min(pa, va) > 0 else 0.0
            if iou > biou:
                biou, best = iou, pb
        return best

    # ---- 违规事件 ----
    def on_event(self, ev):
        self.events.append(ev)
        tid = ev.get("track_id")
        if tid in self.track_records:
            self.track_records[tid]["violation"] = {
                "status": ev["status"], "start": ev["start_ts"],
                "end": ev["end_ts"], "light_state": ev.get("light_state", "unknown"),
            }

    # ---- 折叠为结构化中间态 ----
    def build(self):
        # 灯态段: 合并相邻同状态
        light_segments = []
        for ts, state, conf in self.light_timeline:
            if light_segments and light_segments[-1]["state"] == state:
                light_segments[-1]["end"] = ts
                light_segments[-1]["conf"] = max(light_segments[-1]["conf"], conf)
            else:
                light_segments.append({"start": ts, "end": ts, "state": state, "conf": conf})
        # 车牌读取去重(同车取最高置信那次作为代表, 但保留读取时刻列表)
        tracks = {}
        for tid, rec in self.track_records.items():
            reads = sorted(rec["plate_reads"], key=lambda x: -x[2])
            best_text = reads[0][1] if reads else ""
            best_conf = reads[0][2] if reads else 0.0
            tracks[tid] = {
                "tid": tid, "vehicle_class": rec["vehicle_class"],
                "occupancy": {
                    "peak": round(rec["occ_peak"], 3),
                    "first_ts": rec["occ_first"], "last_ts": rec["occ_last"],
                    "peak_box": rec["occ_peak_box"],
                },
                "plate": {"text": best_text, "best_conf": round(best_conf, 3),
                          "reads": [{"ts": t, "text": x, "conf": round(c, 3)}
                                    for t, x, c in rec["plate_reads"]]},
                "violation": rec["violation"],
            }
        return {
            "video": self.video, "fps": self.fps, "duration": self.duration,
            "light_segments": light_segments,
            "tracks": tracks,
            "events": self.events,
        }


class CotReporter:
    """把 AnalysisResult 渲染为 COT markdown + 截图目录 (确定性模板, 无 LLM)。"""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def render(self, analysis, out_dir, frames=None):
        """frames: 可选 {state: bgr, tid: {occ: bgr, plate: bgr}} 截图素材。"""
        ensure_dir(out_dir)
        md = self._render_markdown(analysis)
        md_path = os.path.join(out_dir, f"COT_{analysis['video']}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        # 截图
        shot_dir = os.path.join(out_dir, "screenshots")
        ensure_dir(shot_dir)
        shots = self._save_screenshots(analysis, shot_dir, frames)
        return md_path, shots

    def _render_markdown(self, a):
        L = []
        L.append(f"# 视频 {a['video']} 解析 (COT 可解释报告)\n")
        L.append(f"- 时长: {fmt_ts(a['duration'])}  帧率: {a['fps']:.1f}fps\n")

        # ① 灯态时间线
        L.append("\n## 一、红绿灯状态时间线\n")
        if not a["light_segments"]:
            L.append("- (无信号灯检测结果)\n")
        for seg in a["light_segments"]:
            st = seg["state"]
            note = ""
            if st == "unknown":
                note = "  (被遮挡/未捕获信号灯, 推测为绿灯但需复核)"  # D1
            L.append(f"- {fmt_ts(seg['start'])} - {fmt_ts(seg['end'])} "
                     f"行人灯为 **{self._zh(st)}**{note}\n")

        # ② 占道车辆与判定
        L.append("\n## 二、占道车辆与违章判定\n")
        tracks = a["tracks"]
        if not tracks:
            L.append("- (未检测到占道车辆)\n")
        for tid, tr in tracks.items():
            occ = tr["occupancy"]
            plate = tr["plate"]["text"] or "(未识别车牌)"
            L.append(f"\n### 车辆 #{tid} ({plate})\n")
            if occ["peak"] <= 0.02 or occ["first_ts"] is None:
                L.append(f"- 未检测到该车占据斑马线, 不构成压线。\n")
            else:
                occ_pct = f"{occ['peak']*100:.0f}%"
                L.append(f"- 在 {fmt_ts(occ['first_ts'])} - {fmt_ts(occ['last_ts'])} "
                         f"占据斑马线约 **{occ_pct}** (峰值), 符合\"静止+压线\"特征。\n")
            # 车牌读取时刻
            if tr["plate"]["text"]:
                r0 = tr["plate"]["reads"][0]
                L.append(f"- 车牌 **{tr['plate']['text']}** 在 {fmt_ts(r0['ts'])} 秒"
                         f"(置信 {tr['plate']['best_conf']:.2f}) 读到。\n")
            else:
                L.append(f"- 车牌: 全程未清晰识别。\n")
            # 违章结论
            v = tr["violation"]
            if v:
                if v["status"] == "confirmed":
                    L.append(f"- **结论: {plate} 违章 (confirmed)** "
                             f"— 行人灯为{v['light_state']}且静止压线。\n")
                elif v["status"] == "review":
                    L.append(f"- **结论: {plate} 待复核 (review)** "
                             f"— 灯态未知/遮挡, 推测绿灯, 安全侧交人复核。\n")
                else:
                    L.append(f"- 结论: {plate} 不违章。\n")
            else:
                L.append(f"- 结论: 该车不满足违规三条件(灯态/静止/压线不全), 不违章。\n")

        # ③ 最终结论
        L.append("\n## 三、最终结论\n")
        confirmed = [t for t in tracks.values()
                     if t["violation"] and t["violation"]["status"] == "confirmed"]
        review = [t for t in tracks.values()
                  if t["violation"] and t["violation"]["status"] == "review"]
        if confirmed:
            names = ", ".join(t["plate"]["text"] or f"#{t['tid']}" for t in confirmed)
            L.append(f"- 确认违规: {names} ({len(confirmed)} 起)\n")
        else:
            L.append(f"- 确认违规: 0 起\n")
        if review:
            names = ", ".join(t["plate"]["text"] or f"#{t['tid']}" for t in review)
            L.append(f"- 待复核: {names} ({len(review)} 起)\n")
        L.append("\n---\n*本报告由规则系统结构化中间态确定性渲染, 非 LLM 生成, 可复现。*\n")
        return "".join(L)

    @staticmethod
    def _zh(state):
        return {"green": "绿色", "red": "红色", "flashing": "闪烁(清空相位)",
                "unknown": "未知/遮挡"}.get(state, state)

    @staticmethod
    def _save_screenshots(a, shot_dir, frames):
        saved = []
        if not frames:
            return saved
        # 灯态证据帧
        for state, bgr in frames.get("light", {}).items():
            p = os.path.join(shot_dir, f"light_{state}.jpg")
            cv2.imwrite(p, bgr)
            saved.append(p)
        # 占用峰值帧 / 车牌帧
        for tid, d in frames.get("tracks", {}).items():
            if "occ" in d and d["occ"] is not None:
                p = os.path.join(shot_dir, f"tid{tid}_occupancy.jpg")
                cv2.imwrite(p, d["occ"])
                saved.append(p)
            if "plate" in d and d["plate"] is not None:
                p = os.path.join(shot_dir, f"tid{tid}_plate.jpg")
                cv2.imwrite(p, d["plate"])
                saved.append(p)
        return saved
