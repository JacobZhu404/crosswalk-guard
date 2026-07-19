#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 GT · Phase 2 重建脚本（build_gt.py）

把画廊修正（feedback/）+ 迭代 badcase（badcases/）merge 进 canonical：
  - light_states.csv  : 帧级 → 段级 merge（本 Phase 唯一提交的 canonical）
  - crosswalk/*.json  : 仅 dry-run 报告，不提交（决策 2）
  - events.csv/videos.csv : 保留 base，不重建（决策 3）

红线（见 design doc）：
  - 输入只读（feedback/source/badcases 绝不回写）
  - 每处变更可溯源到具体 feedback/badcase 行（build_audit.json）
  - 旧 canonical 留快照 _snapshot_pre_build/ + git tag gt-pre-phase2（决策 4）
  - 测量尺变更透明：旧/新 F1 对比由 Task #29 完成；本脚本只产 GT + 审计
  - 冲突守卫（§2.5）：green↔red 翻转且视频 has_violation=True → 高危待签，默认不写 canonical；
                      负例(01/10) merge 后仍不得出现 green，否则 abort 零产出

用法（默认 dry-run，不碰 canonical）：
    python scripts/build_gt.py                      # 计算 + 写审计 + 写 _proposed（不覆盖 canonical）
    python scripts/build_gt.py --write             # 安全时覆盖 light_states.csv（先快照+tag）
    python scripts/build_gt.py --write --accept-high-risk   # 高危变更也强制提交（需人工已签）

本脚本为纯标准库实现，无第三方依赖。
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

# ---------- 可调超参 ----------
MERGE_GAP_TOL = 0.5      # 同 gt 相邻覆盖点间隔 <= tol 合并为一段连续覆盖
MARGIN = 0.1             # 合并后区间向两侧各扩 margin（半帧），对齐 design 聚合-1 用例
DENSITY_K = 3            # 覆盖区间由 >= K 个采样点支撑 → confidence=feedback，否则 feedback-tentative
CROSSWALK_IOU_OK = 0.8   # dry-run 一致性阈值
SENTINEL_END = 999       # base 段 end_s 哨兵（"到视频末尾"）
ENABLING = {"green"}     # 撑起违章资格的灯态集合（green 窗口 = 车辆占道违章判定窗）

# ---------- 纯函数（可单测） ----------

def _fmt_num(x):
    """整数去尾零，浮点保留 3 位。"""
    if x == int(x):
        return int(x)
    return round(x, 3)


def _state_enables(state):
    return state in ENABLING


def merge_override_points(points, gap_tol=MERGE_GAP_TOL, margin=MARGIN):
    """把帧级覆盖点 [(t, gt, src, video, reason, row), ...] 合并成连续区间。

    返回区间列表：{t0,t1,gt,src,video,reason,t_repr,n,rows}
    同 gt 且相邻间隔 <= gap_tol 的点合并；区间两侧各扩 margin。
    """
    if not points:
        return []
    # 按 t 排序
    pts = sorted(points, key=lambda p: p["t"])
    groups = []
    cur = [pts[0]]
    for p in pts[1:]:
        last = cur[-1]
        if p["gt"] == last["gt"] and (p["t"] - last["t"]) <= gap_tol:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)

    intervals = []
    for g in groups:
        gt = g[0]["gt"]
        src = g[0]["src"]
        video = g[0]["video"]
        t_min = min(p["t"] for p in g) - margin
        t_max = max(p["t"] for p in g) + margin
        if t_min < 0:
            t_min = 0.0
        mid = (t_min + t_max) / 2.0
        # 代表行：t 最接近区间中点的那一行
        rep = min(g, key=lambda p: abs(p["t"] - mid))
        intervals.append({
            "t0": t_min, "t1": t_max, "gt": gt, "src": src, "video": video,
            "reason": rep.get("reason", ""), "t_repr": rep["t"],
            "raw_lo": min(p["t"] for p in g), "raw_hi": max(p["t"] for p in g),
            "n": len(g), "rows": g,
        })
    intervals.sort(key=lambda iv: iv["t0"])
    return intervals


def clamp_interval_to_same_state_base(iv, base_segs):
    """钳制覆盖区间，避免 margin 把区间越过 base 段边界、蹭出伪翻转。

    规则：若 feedback 簇（raw_lo..raw_hi）整体落在某段 *同态* base 内
    （即该 base 段 state==gt 且完整包含簇），则把区间钳制到该 base 段边界
    （允许 margin 在内，但不越出到 foreign-state base）。
    若簇本身跨 base 边界（真校正，无单一同态 base 包含它）→ 不钳制，照常覆盖。

    效果：feedback 仅"确认"已有 base 时零变更；真校正仍正常生效并被守卫捕获。
    """
    raw_lo, raw_hi = iv["raw_lo"], iv["raw_hi"]
    containing = [s for s in base_segs
                  if s["state"] == iv["gt"] and s["start"] <= raw_lo and s["end"] >= raw_hi]
    if not containing:
        return iv
    seg = containing[0]
    iv["t0"] = max(iv["t0"], seg["start"])
    iv["t1"] = min(iv["t1"], seg["end"])
    return iv


def _make_base(seg, s, e):
    return {"start": s, "end": e, "state": seg["state"], "confidence": seg["confidence"],
            "note": seg["note"], "src": "base"}


def _make_override(seg, s, e, iv, k=DENSITY_K):
    if iv["src"] == "badcase":
        conf = "badcase"
    else:
        conf = "feedback" if iv["n"] >= k else "feedback-tentative"
    note = "{base} | merge({src}): {video}@{t} gt={gt} (n={n}, reason={reason})".format(
        base=seg["note"], src=iv["src"], video=iv["video"], t=_fmt_num(iv["t_repr"]),
        gt=iv["gt"], n=iv["n"], reason=iv["reason"])
    return {"start": s, "end": e, "state": iv["gt"], "confidence": conf, "note": note,
            "src": iv["src"], "_old": {"state": seg["state"], "confidence": seg["confidence"],
                                       "note": seg["note"]}, "_rows": iv["rows"]}


def _apply_interval(segs, iv, k=DENSITY_K):
    """把一个覆盖区间应用到当前段列表，返回新段列表（仅在 gt != base_state 时改写）。"""
    out = []
    for seg in segs:
        # 无重叠
        if seg["end"] <= iv["t0"] or seg["start"] >= iv["t1"]:
            out.append(seg)
            continue
        # 左段（base）
        if seg["start"] < iv["t0"]:
            out.append(_make_base(seg, seg["start"], iv["t0"]))
        # 中段（覆盖区间与段的交集）
        in_s = max(seg["start"], iv["t0"])
        in_e = min(seg["end"], iv["t1"])
        if in_s < in_e:
            if seg["state"] == iv["gt"]:
                # gt == base_state → 不改写，保持 diff 干净（对齐 §2.1/§2.2）
                out.append(_make_base(seg, in_s, in_e))
            else:
                out.append(_make_override(seg, in_s, in_e, iv, k))
        # 右段（base）
        if seg["end"] > iv["t1"]:
            out.append(_make_base(seg, iv["t1"], seg["end"]))
    out = [p for p in out if p["end"] - p["start"] > 1e-9]
    return out


def _merge_identical_adjacent(segs):
    """合并相邻且 (state,confidence,note,src) 全同的段，吸收内部 _rows/_old。"""
    if not segs:
        return segs
    merged = [dict(segs[0])]
    for seg in segs[1:]:
        top = merged[-1]
        if (seg["state"] == top["state"] and seg["confidence"] == top["confidence"]
                and seg["note"] == top["note"] and seg["src"] == top["src"]):
            # 合并：扩展 end，合并 _rows
            top["end"] = seg["end"]
            if "_rows" in seg and seg["_rows"] is not None:
                top.setdefault("_rows", []).extend(seg["_rows"])
        else:
            merged.append(dict(seg))
    return merged


def build_light_video(base_segs, intervals, has_violation, k=DENSITY_K):
    """对单视频做帧级→段级 merge。

    参数：
      base_segs     : list[{start,end,state,confidence,note}]（按 start 排）
      intervals     : merge_override_points 的输出（已含 feedback + badcase，badcase 在后）
      has_violation : 0/1（来自 videos.csv）
    返回：
      {segments, changed, high_risk, neg_ok, aborted, neg_offenders}
    """
    segs = [dict(s, src="base") for s in base_segs]
    # 钳制：消除 margin 越界造成的伪翻转（feedback 仅确认 base 时不产生变更）
    intervals = [clamp_interval_to_same_state_base(iv, base_segs) for iv in intervals]
    for iv in intervals:
        segs = _apply_interval(segs, iv, k)
    segs.sort(key=lambda s: s["start"])
    segs = _merge_identical_adjacent(segs)

    changed = []
    high_risk = []
    neg_offenders = []
    for seg in segs:
        if seg.get("src") == "base":
            continue
        # 变更段
        old = seg.get("_old", {"state": None, "confidence": None, "note": None})
        changed.append({
            "start": _fmt_num(seg["start"]), "end": _fmt_num(seg["end"]),
            "state": seg["state"], "confidence": seg["confidence"], "note": seg["note"],
            "src": seg["src"],
            "old_state": old["state"], "old_confidence": old["confidence"],
            "src_rows": [{"video": r["video"], "t": _fmt_num(r["t"]), "gt": r["gt"],
                          "verdict": r.get("verdict", ""), "reason": r.get("reason", ""),
                          "note": r.get("note", "")} for r in seg.get("_rows", [])],
        })
        # 高危：green↔red(或 green↔unknown) 翻转，且视频 has_violation=True
        old_en = _state_enables(old["state"])
        new_en = _state_enables(seg["state"])
        if has_violation == 1 and old_en != new_en:
            high_risk.append({
                "start": _fmt_num(seg["start"]), "end": _fmt_num(seg["end"]),
                "old_state": old["state"], "new_state": seg["state"],
                "src": seg["src"], "note": seg["note"],
            })
        # 负例断言：has_violation=False 却出现 green → 违规
        if has_violation == 0 and seg["state"] == "green":
            neg_offenders.append({
                "start": _fmt_num(seg["start"]), "end": _fmt_num(seg["end"]),
                "src": seg["src"], "src_rows": changed[-1]["src_rows"],
            })

    neg_ok = (len(neg_offenders) == 0)
    aborted = (not neg_ok)  # 负例翻转 = 硬 abort，零产出
    return {"segments": segs, "changed": changed, "high_risk": high_risk,
            "neg_ok": neg_ok, "aborted": aborted, "neg_offenders": neg_offenders}


# ---------- 多边形 IoU（crosswalk dry-run，纯标准库） ----------

def _poly_area(poly):
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _signed_area(poly):
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _clip_polygon(subject, clip):
    """Sutherland–Hodgman：用凸 clip 多边形裁剪 subject（凸多边形适用）。"""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
    def intersect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = a; x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    out = list(subject)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        inp = out
        out = []
        if not inp:
            break
        s = inp[-1]
        for p in inp:
            if inside(p, a, b):
                if not inside(s, a, b):
                    out.append(intersect(s, p, a, b))
                out.append(p)
            elif inside(s, a, b):
                out.append(intersect(s, p, a, b))
            s = p
    return out


def poly_iou(p1, p2):
    if not p1 or not p2:
        return 0.0
    # Sutherland–Hodgman 要求 clip 多边形为 CCW（正向面积）；
    # 若为 CW（负向面积）则 inside 判定反转→交集算空。裁剪前归一化。
    clip = list(p2)
    if _signed_area(clip) < 0:
        clip = clip[::-1]
    inter = _clip_polygon(p1, clip)
    ia = _poly_area(inter)
    if ia == 0:
        return 0.0
    a1, a2 = _poly_area(p1), _poly_area(p2)
    union = a1 + a2 - ia
    return ia / union if union > 0 else 0.0


def _parse_poly(s):
    """'718,697;435,251;...' → [[718,697],[435,251],...]"""
    pts = []
    for pair in s.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x, y = pair.split(",")
        pts.append([float(x), float(y)])
    return pts


# ---------- 文件 IO（只读输入） ----------

def load_light_states(path):
    """读取段级 canonical。

    注意：历史 canonical 中存在 note 字段含未加引号逗号的畸形行
    （如 `...,occluded,人工画廊标注:...(60帧other),系统判unknown正确`），
    标准 csv.DictReader 会错误劈列、静默丢内容。故用 maxsplit=5 解析，
    保证 note 全文（含内部逗号）被完整保留。
    """
    segs = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    if not lines:
        return segs
    # 跳过表头（表头本身无内部逗号，split 即可）
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",", 5)  # 前 5 个逗号分隔前 5 列，余下全归 note
        if len(parts) < 6:
            parts += [""] * (6 - len(parts))
        video, start_s, end_s, state, confidence, note = parts
        # 迭代剥去 CSV 引号：兼容①历史畸形未引号逗号行 ②已正确引号行 ③重跑产生的多重引号
        while len(note) >= 2 and note.startswith('"') and note.endswith('"'):
            note = note[1:-1]
        segs[video].append({
            "start": float(start_s), "end": float(end_s),
            "state": state, "confidence": confidence, "note": note,
        })
    for v in segs:
        segs[v].sort(key=lambda s: s["start"])
    return segs


def load_videos(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["video"]] = int(row["has_violation"])
    return out


def load_light_feedback(path):
    """返回 {video: [point, ...]}，point={t,gt,src,video,reason,verdict,note,row}"""
    by_video = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gt = (row.get("gt") or "").strip()
            if not gt:
                continue  # gt 空 → 无修正信号
            by_video[row["video"]].append({
                "t": float(row["t_sec"]), "gt": gt, "src": "feedback",
                "video": row["video"], "reason": (row.get("reason") or "").strip(),
                "verdict": (row.get("verdict") or "").strip(),
                "note": (row.get("note") or "").strip(),
                "row": row,
            })
    return by_video


def load_badcases(path):
    """返回 {video: [point, ...]}，仅 modality==light。gt 取 expected。"""
    by_video = defaultdict(list)
    if not os.path.exists(path):
        return by_video
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            modality = (row.get("modality") or "").strip()
            if modality and modality != "light":
                continue
            expected = (row.get("expected") or "").strip()
            if not expected:
                continue
            t = None
            if (row.get("t_sec") or "").strip():
                t = float(row["t_sec"])
            elif (row.get("window") or "").strip():
                # 形如 "[21-68]" 或 "21-68" → 取中点
                w = row["window"].strip().strip("[]")
                parts = w.split("-")
                if len(parts) == 2:
                    try:
                        t = (float(parts[0]) + float(parts[1])) / 2.0
                    except ValueError:
                        t = None
            if t is None:
                continue
            by_video[row["video"]].append({
                "t": t, "gt": expected, "src": "badcase",
                "video": row["video"], "reason": (row.get("note") or "").strip(),
                "verdict": "badcase", "note": (row.get("note") or "").strip(),
                "row": row,
            })
    return by_video


def crosswalk_dryrun(crosswalk_dir, feedback_path):
    """对每视频比对 base poly(json) vs feedback poly(csv)，仅报告不写。"""
    results = []
    base_by_video = {}
    if crosswalk_dir and os.path.isdir(crosswalk_dir):
        for fn in os.listdir(crosswalk_dir):
            if not fn.endswith(".json"):
                continue
            video = fn[:-5]
            try:
                with open(os.path.join(crosswalk_dir, fn), encoding="utf-8") as f:
                    data = json.load(f)
                frames = [{"ts": fr.get("ts"), "poly": fr.get("poly")}
                          for fr in data.get("frames", []) if fr.get("poly")]
                base_by_video[video] = frames
            except Exception:
                continue

    fb_by_video = defaultdict(list)
    if feedback_path and os.path.exists(feedback_path):
        with open(feedback_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                fb_by_video[row["video"]].append(row)

    videos = set(base_by_video) | set(fb_by_video)
    for video in sorted(videos):
        fb_rows = fb_by_video.get(video, [])
        frames = base_by_video.get(video, [])
        if not fb_rows:
            results.append({"video": video, "status": "no-feedback",
                            "n_base_frames": len(frames)})
            continue
        if not frames:
            results.append({"video": video, "status": "no-base",
                            "n_feedback": len(fb_rows)})
            continue
        ious = []
        for r in fb_rows:
            poly = _parse_poly(r.get("poly", ""))
            if not poly:
                continue
            # 取 ts 最近的 base frame
            best = min(frames, key=lambda fr: abs((fr["ts"] or 0) - float(r.get("t_sec", 0))))
            ious.append(poly_iou(poly, best["poly"]))
        if ious:
            mean_iou = sum(ious) / len(ious)
            min_iou = min(ious)
        else:
            mean_iou = min_iou = 0.0
        results.append({
            "video": video, "status": "compared",
            "n_feedback": len(fb_rows), "n_base_frames": len(frames),
            "mean_iou": round(mean_iou, 4), "min_iou": round(min_iou, 4),
            "consistent": mean_iou >= CROSSWALK_IOU_OK,
        })
    return results


# ---------- 输出 ----------

def write_light_csv(segments_by_video, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "start_s", "end_s", "state", "confidence", "note"])
        for video in sorted(segments_by_video):
            for seg in segments_by_video[video]:
                w.writerow([video, _fmt_num(seg["start"]), _fmt_num(seg["end"]),
                            seg["state"], seg["confidence"], seg["note"]])


def build_audit(videos_result, crosswalk_dry):
    audit = {"videos": {}, "crosswalk_dryrun": crosswalk_dry}
    for video, res in videos_result.items():
        audit["videos"][video] = {
            "has_violation": res["has_violation"],
            "segments_total": len(res["segments"]),
            "changed": res["changed"],
            "high_risk": res["high_risk"],
            "neg_ok": res["neg_ok"],
            "aborted": res["aborted"],
            "neg_offenders": res["neg_offenders"],
        }
    return audit


def build_report_md(videos_result, crosswalk_dry, high_risk_global, neg_abort, args):
    L = []
    L.append("# Phase 2 重建报告（build_gt.py）\n")
    n_changed_videos = sum(1 for r in videos_result.values() if r["changed"])
    n_changed_segs = sum(len(r["changed"]) for r in videos_result.values())
    L.append("## 摘要")
    L.append("- 提交模态：light_states.csv（crosswalk 仅 dry-run，见下）")
    L.append(f"- 变更视频数：{n_changed_videos}")
    L.append(f"- 变更段数：{n_changed_segs}")
    L.append(f"- 高危翻转（待人工签字）：{len(high_risk_global)}")
    L.append(f"- 负例断言：{'通过' if not neg_abort else '失败 → 已 abort 零产出'}")
    L.append("")

    if high_risk_global:
        L.append("## ⚠️ 高风险变更（需人工签字）")
        L.append("以下 green↔red 翻转落在 has_violation=True 视频，可能改变评测资格。")
        L.append("默认不写 canonical；需复核后加 `--accept-high-risk` 重新运行。")
        for h in high_risk_global:
            L.append(f"- {h['video']} [{h['start']},{h['end']}] "
                     f"{h['old_state']}→{h['new_state']} (src={h['src']})")
        L.append("")

    L.append("## 逐视频变更明细")
    for video in sorted(videos_result):
        r = videos_result[video]
        if not r["changed"]:
            continue
        L.append(f"### {video}（has_violation={r['has_violation']}）")
        for c in r["changed"]:
            L.append(f"- [{c['start']},{c['end']}] {c['old_state']}→{c['state']} "
                     f"(conf={c['confidence']}, src={c['src']})")
            for sr in c["src_rows"]:
                L.append(f"    ↳ {sr['video']}@{sr['t']} gt={sr['gt']} "
                         f"verdict={sr['verdict']} reason={sr['reason']}")
        L.append("")

    if neg_abort:
        L.append("## ❌ 负例断言失败（abort）")
        for video, r in videos_result.items():
            for off in r["neg_offenders"]:
                L.append(f"- {video} 出现 green 段 [{off['start']},{off['end']}] "
                         f"（has_violation=0 不得含 green）")
        L.append("")

    L.append("## crosswalk dry-run（不提交）")
    for c in crosswalk_dry:
        if c["status"] == "compared":
            flag = "consistent" if c["consistent"] else "DIVERGENT"
            L.append(f"- {c['video']}: n_fb={c['n_feedback']} n_base={c['n_base_frames']} "
                     f"meanIoU={c['mean_iou']} minIoU={c['min_iou']} → {flag}")
        else:
            L.append(f"- {c['video']}: {c['status']}")
    L.append("")
    return "\n".join(L)


# ---------- 主流程 ----------

def main(argv=None):
    ap = argparse.ArgumentParser(description="统一 GT Phase 2 重建")
    ap.add_argument("--gt-dir", default="datasets/gt", help="GT 根目录")
    ap.add_argument("--artifacts-dir", default=None, help="审计/提案输出目录（默认 <gt-dir>/_build_artifacts）")
    ap.add_argument("--write", action="store_true", help="安全时覆盖 light_states.csv（先快照+tag）")
    ap.add_argument("--accept-high-risk", action="store_true", help="高危翻转也强制提交（需人工已签）")
    args = ap.parse_args(argv)

    gt_dir = args.gt_dir
    artifacts_dir = args.artifacts_dir or os.path.join(gt_dir, "_build_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    base_states_path = os.path.join(gt_dir, "light_states.csv")
    videos_path = os.path.join(gt_dir, "videos.csv")
    fb_path = os.path.join(gt_dir, "feedback", "light_feedback.csv")
    bc_path = os.path.join(gt_dir, "badcases", "template.csv")  # 当前空；真实 badcase 落此目录
    bc_dir = os.path.join(gt_dir, "badcases")
    cw_dir = os.path.join(gt_dir, "crosswalk")
    cw_fb_path = os.path.join(gt_dir, "feedback", "crosswalk_feedback.csv")

    base_segs = load_light_states(base_states_path)
    videos = load_videos(videos_path)
    fb = load_light_feedback(fb_path)
    # badcases：目录内所有 csv（除 template 外），目前 template 空
    bc = defaultdict(list)
    if os.path.isdir(bc_dir):
        for fn in os.listdir(bc_dir):
            if fn.endswith(".csv") and fn != "template.csv":
                for v, pts in load_badcases(os.path.join(bc_dir, fn)).items():
                    bc[v].extend(pts)

    videos_result = {}
    high_risk_global = []
    neg_abort = False

    all_videos = set(base_segs) | set(videos)
    for video in sorted(all_videos):
        bseg = base_segs.get(video, [])
        has_v = videos.get(video, 0)
        # feedback 区间（先）+ badcase 区间（后，优先级更高）
        intervals = merge_override_points(fb.get(video, []))
        intervals += merge_override_points(bc.get(video, []))
        res = build_light_video(bseg, intervals, has_v)
        res["has_violation"] = has_v
        res["segments"] = res["segments"]  # keep
        videos_result[video] = res
        high_risk_global.extend([dict(h, video=video) for h in res["high_risk"]])
        if res["aborted"]:
            neg_abort = True

    # crosswalk dry-run
    crosswalk_dry = crosswalk_dryrun(cw_dir, cw_fb_path)

    # 审计 + 报告
    audit = build_audit(videos_result, crosswalk_dry)
    report = build_report_md(videos_result, crosswalk_dry, high_risk_global, neg_abort, args)
    audit_path = os.path.join(artifacts_dir, "build_audit.json")
    report_path = os.path.join(artifacts_dir, "build_report.md")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 提案 light_states.csv（始终生成，供检视）
    proposed = {v: r["segments"] for v, r in videos_result.items()}
    proposed_path = os.path.join(artifacts_dir, "light_states.proposed.csv")
    write_light_csv(proposed, proposed_path)

    # 决策与退出码
    if neg_abort:
        print(f"[ABORT] 负例断言失败：has_violation=0 的视频出现 green 段。零产出 canonical。")
        print(f"  审计：{audit_path}")
        print(f"  报告：{report_path}")
        return 3

    if high_risk_global and not args.accept_high_risk:
        print(f"[BLOCKED] 检出 {len(high_risk_global)} 处高危 green↔red 翻转（has_violation=True 视频）。")
        print("  默认不写 canonical。复核后加 --accept-high-risk 重新运行。")
        for h in high_risk_global:
            print(f"    - {h['video']} [{h['start']},{h['end']}] {h['old_state']}→{h['new_state']} (src={h['src']})")
        print(f"  提案：{proposed_path}")
        print(f"  审计：{audit_path}")
        print(f"  报告：{report_path}")
        return 2

    if args.write:
        # 决策 4：先快照 + git tag，再覆盖
        snapshot_dir = os.path.join(gt_dir, "_snapshot_pre_build")
        os.makedirs(snapshot_dir, exist_ok=True)
        import shutil
        for fn in ["light_states.csv", "videos.csv"]:
            src = os.path.join(gt_dir, fn)
            dst = os.path.join(snapshot_dir, fn)
            if os.path.exists(src) and not os.path.exists(dst):  # 仅首次拷贝，防重跑覆盖 pre-build 基线
                shutil.copy2(src, dst)
        # .gitignore 快照目录
        # 用 git 定位仓库根（比向上找 .gitignore 更稳）
        repo_root = gt_dir
        try:
            out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                 cwd=gt_dir, capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                repo_root = out.stdout.strip()
        except Exception:
            pass
        gi = os.path.join(repo_root, ".gitignore")
        if os.path.exists(gi):
            with open(gi, encoding="utf-8") as f:
                gic = f.read()
            if "_snapshot_pre_build/" not in gic:
                with open(gi, "a", encoding="utf-8") as f:
                    f.write("\n# Phase 2 GT 重建快照（本地，不入库）\n_snapshot_pre_build/\n")
        # git tag（标记 pre-Phase2 状态；已存在则跳过，不静默吞错）
        tag_r = subprocess.run(["git", "-C", repo_root, "tag", "gt-pre-phase2"],
                               capture_output=True, text=True)
        if tag_r.returncode != 0 and "already exists" not in tag_r.stderr:
            print(f"[WARN] git tag gt-pre-phase2 失败：{tag_r.stderr.strip()}")
        # 覆盖 canonical
        write_light_csv(proposed, base_states_path)
        print(f"[COMMITTED] 已覆盖 {base_states_path}（快照→{snapshot_dir}，tag gt-pre-phase2）")
    else:
        print(f"[DRY-RUN] 未覆盖 canonical。提案见 {proposed_path}")
    print(f"  审计：{audit_path}")
    print(f"  报告：{report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
