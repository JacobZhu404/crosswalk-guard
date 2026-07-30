#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ingest_light_gt.py — 工具C: 校验 Jacob 的 canonical 灯态 GT 导出 → 原子写入库。

方案: docs/plans/2026-07-28-cc-plan-canonical-light-gt-annotation-tool.md
输入: datasets/gt/light_canonical_gt_raw.json(标注画廊导出, 可能是拼接多份→取第一份)
输出: datasets/gt/light_canonical_gt.json(校验+规整后的唯一真值, 原子写)

校验(硬): schema; 每帧字段; box_norm 归一化范围/x1<x2/y1<y2; color∈合法集; governing→type=pedestrian;
          no_light↔boxes 空 一致性; 派生 gt_walk(governing 存在且=green)。
交叉核对(软, 仅告警不阻断): 与 events.csv 段状态比对(green 段 governing 应为绿, 反之亦然)。
用法: python3 scripts/ingest_light_gt.py
"""
import json, sys, csv
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "gt" / "light_canonical_gt_raw.json"
OUT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
EVENTS = ROOT / "datasets" / "gt" / "events.csv"
COLORS = {"red", "green", "off", "countdown", "unclear"}
SCHEMA = "light_canonical_gt_v1"


def load_first_object(path):
    """容忍拼接多份 JSON: 取第一份完整对象。"""
    raw = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i, n = 0, len(raw)
    while i < n and raw[i] in " \t\r\n":
        i += 1
    obj, _ = dec.raw_decode(raw, i)
    return obj


def load_events():
    segs = defaultdict(list)
    with open(EVENTS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            segs[r["video"]].append((float(r["start_s"]), float(r["end_s"]), r["light_state"].strip()))
    return segs


def events_state(segs, video, t):
    for s, e, st in segs.get(video, []):
        if s - 1e-6 <= t <= e + 1e-6:
            return st
    return "gap"


def main():
    obj = load_first_object(RAW)
    errs, warns = [], []
    if obj.get("schema") != SCHEMA:
        errs.append(f"schema 非 {SCHEMA}: {obj.get('schema')}")
    frames = obj.get("frames", [])
    segs = load_events()

    ngov = 0
    color_dist = Counter()
    per_video = defaultdict(lambda: {"frames": 0, "gov": 0, "no_light": 0, "green": 0, "red": 0})
    multi_gov = 0
    n_no_light = 0
    n_unknown = 0  # governing 灯态 unclear → 评测 UNKNOWN
    clean_frames = []

    for fr in frames:
        v = fr.get("video"); t = fr.get("t")
        for k in ("video", "source_fi", "t", "no_light", "boxes"):
            if k not in fr:
                errs.append(f"{v}@{t} 缺字段 {k}")
        pv = per_video[v]; pv["frames"] += 1
        boxes = fr.get("boxes", [])
        nl = fr.get("no_light", False)
        if nl:
            n_no_light += 1; pv["no_light"] += 1
            if boxes:
                warns.append(f"{v}@{t} no_light=True 但有 {len(boxes)} 框")
        gov_here = 0
        gov_colors = set()
        for b in boxes:
            bn = b.get("box_norm")
            if not (isinstance(bn, list) and len(bn) == 4):
                errs.append(f"{v}@{t} box_norm 非法 {bn}"); continue
            x1, y1, x2, y2 = bn
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                errs.append(f"{v}@{t} box 越界/反 {bn}")
            if b.get("color") not in COLORS:
                errs.append(f"{v}@{t} color 非法 {b.get('color')}")
            if b.get("governing"):
                ngov += 1; gov_here += 1; pv["gov"] += 1
                color_dist[b.get("color")] += 1
                gov_colors.add(b.get("color"))
                if b.get("type") != "pedestrian":
                    errs.append(f"{v}@{t} governing 非 pedestrian: {b.get('type')}")
        if gov_here > 1:
            multi_gov += 1
        # 派生 gt_walk / UNKNOWN
        if gov_colors:
            if "green" in gov_colors:
                pv["green"] += 1
            elif gov_colors <= {"unclear"}:
                n_unknown += 1
            else:
                pv["red"] += 1
        # 交叉核对 events(软)
        if gov_colors and not nl:
            es = events_state(segs, v, t)
            if es == "green" and "green" not in gov_colors:
                warns.append(f"{v}@{t}s events=green 但 governing={sorted(gov_colors)}")
            if es in ("red", "occluded", "none") and "green" in gov_colors:
                warns.append(f"{v}@{t}s events={es} 但 governing 含 green")
        clean_frames.append(fr)

    print("=== 校验结果 ===")
    print(f"帧数: {len(frames)} | 覆盖视频: {len(per_video)} | governing 框: {ngov} | 多真值帧: {multi_gov}")
    print(f"no_light 帧: {n_no_light} | governing 灯态 unclear(评测UNKNOWN): {n_unknown}")
    print(f"governing 颜色分布: {dict(color_dist)}")
    print("逐视频:")
    for v in sorted(per_video):
        p = per_video[v]
        print(f"  {v}: 帧{p['frames']:>3} gov框{p['gov']:>3} 绿帧{p['green']:>3} 红帧{p['red']:>3} 无灯{p['no_light']:>3}")
    print(f"\n错误: {len(errs)} | 交叉核对告警: {len(warns)}")
    for e in errs[:40]:
        print("  ERR", e)
    for w in warns[:40]:
        print("  WARN", w)
    if len(warns) > 40:
        print(f"  ...(共 {len(warns)} 条告警)")

    if errs:
        print("\n[中止] 有硬错误, 未写入库。请修正后重跑。")
        sys.exit(1)

    out = {"schema": SCHEMA, "source": "jacob_annotation_gallery_2026-07-30",
           "n_frames": len(clean_frames), "n_governing": ngov, "frames": clean_frames}
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(OUT)
    print(f"\n[入库] {OUT}  ({len(clean_frames)} 帧, {ngov} governing 框)")


if __name__ == "__main__":
    main()
