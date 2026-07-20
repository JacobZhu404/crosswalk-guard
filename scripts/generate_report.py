#!/usr/bin/env python3
"""可复现识别结果报告 generate_report.py

每次重大迭代后跑一次, 产出一份自包含 HTML 仪表盘, 含每视频:
  - 是否违章 / 违章车牌 / 时间 / 小作文(COT) / 截图 / 完整识别视频(annotated.mp4)
  - 与原始标注(GT)对比: 自动判定(复用 violation_eval.match_violation_events) + 并列原始描述与结构化 GT

零件复用(不重造):
  - 识别结果来自每视频最近一次 run_*/ 缓存(annotated.mp4 / violations.csv / cot/COT_*.md / evidence/*)。
  - GT 自动判定复用 redlight.evaluation.violation_eval 的 match_violation_events / load_violation_gt /
    load_video_metadata / classify_false_positives / aggregate, 与 eval_violations 的 tp/fp/fn 口径一致。

红线:
  - 只读、只反映, 不改 violation_engine / GT / canonical。
  - 默认读缓存; --fresh 重跑全 11 视频管线(v2 + occ_denom=box, 对齐 0.889 基线)。缓存缺失→提示需 --fresh。

⚠️ 一致性硬规则(交付报告务必遵守):
  - 交付的报告必须来自「一次干净的 --fresh 全 11 视频」, 不可把旧缓存视频与新重跑视频混在一份报告里。
  - 原因: denom 修复(visualizer 红框改跟随 occ_denom=box)提交后, 修复前渲染的 annotated.mp4 仍是旧 mask-denom 红框, 与 box-denom 判定不一致 → 破「视频红框==结论」。
    判定数(tp/fp/fn)不受影响(引擎一直 box), 但视频一致性对权威报告是硬要求。
  - 故每次 denom/判定相关改动后, 一律 `python scripts/generate_report.py --fresh` 一次跑全 11, 再出 HTML; 不要用「部分缓存+部分重跑」凑报告。
  - 调试期可用 `--videos` 分批前台跑(崩因是后台长任务被 SIGKILL, 非代码; run 目录逐视频落盘=崩了能续), 但 FINAL 报告仍须一次全量 --fresh。

用法:
  python scripts/generate_report.py                  # 读缓存 -> 出 HTML
  python scripts/generate_report.py --fresh          # 重跑全 11 视频 -> 出 HTML(交付报告用这个)
  python scripts/generate_report.py --bundle         # 复制资产成可迁移归档(默认不拷, ~1GB)
"""
import os
import sys
import csv
import re
import json
import glob
import argparse
import shutil
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# 纯函数 GT 比对层(无 cv2 依赖, 便于单测)。直接复用 violation_eval, 不另写匹配逻辑。
from redlight.evaluation.violation_eval import (  # noqa: E402
    load_violation_gt, load_video_metadata, match_violation_events,
    classify_false_positives, aggregate,
)

VIDEOS = [f"违章{i:02d}" for i in range(1, 12)]
PRESET = "balanced"
MIN_OVERLAP_S = 0.5


# ----------------------------------------------------------------------------
# 报告层函数(可单测, 不依赖 cv2)
# ----------------------------------------------------------------------------
def normalize_plate(plate):
    """车牌归一化(用于匹配子标记): 去空白/连字符/间隔符, 转大写。

    京 LNE 560 -> 京LNE560; ABC-123 -> ABC123; 空 -> ""。
    中文省份字(京沪粤...)是车牌一部分, 保留。
    """
    if not plate:
        return ""
    return re.sub(r"[\s\-·.]", "", str(plate)).upper()


def find_run_dir(video, output_root):
    """找某视频最近一次 run 缓存目录(glob run_{video}*, 取 mtime 最新)。无则 None。"""
    pattern = os.path.join(output_root, f"run_{video}*")
    candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not candidates:
        return None
    return max(candidates, key=lambda d: os.path.getmtime(d))


def read_violations(run_dir):
    """读 run 缓存的 violations.csv -> event dict list。"""
    path = os.path.join(run_dir, "violations.csv")
    events = []
    if not os.path.exists(path):
        return events
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                events.append({
                    "event_id": (row.get("event_id") or "").strip(),
                    "track_id": (row.get("track_id") or "").strip(),
                    "status": (row.get("status") or "").strip(),
                    "start_ts": float(row["start_ts"]),
                    "end_ts": float(row["end_ts"]),
                    "vehicle_class": (row.get("vehicle_class") or "").strip(),
                    "light_state": (row.get("light_state") or "").strip(),
                    "plate": (row.get("plate") or "").strip(),
                    "evidence_image": (row.get("evidence_image") or "").strip(),
                })
            except (KeyError, ValueError):
                continue
    return events


def load_source_annotation(path):
    """source/annotation.csv -> {video: gt_description}(free-text GT)。"""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("input_video") or "").strip()
            if v:
                d[v] = (row.get("gt_description") or "").strip()
    return d


def load_events_rows(events_csv):
    """events.csv 全行(结构化 GT), 保留原始字段。"""
    rows = []
    if not os.path.exists(events_csv):
        return rows
    with open(events_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def summarize_video(pred_events, gt_violations, has_violation, min_overlap_s=0.5):
    """对单视频做 GT 自动判定(复用 match_violation_events) + 子标记。

    pred_events: violations.csv 的 confirmed 事件 [{start_ts,end_ts,plate}]
    gt_violations: load_violation_gt 的该视频段 [{start_s,end_s,plates}]
    has_violation: videos.csv 的 has_violation(bool)

    返回 match_violation_events 的结果 dict(含 tp/fp/fn + matches) + 附加:
      is_violation_hit, flags(子标记列表), neg/oow/fragment 计数(供聚合拆解)。
    """
    norm_pred = [{"start_ts": e["start_ts"], "end_ts": e["end_ts"],
                  "plate": normalize_plate(e.get("plate", ""))} for e in pred_events]
    norm_gt = [{"start_s": g["start_s"], "end_s": g["end_s"],
                "plates": [normalize_plate(p) for p in g["plates"]]} for g in gt_violations]
    res = match_violation_events(norm_pred, norm_gt, min_overlap_s)
    fp_cls = classify_false_positives(
        res, norm_pred, norm_gt, is_negative=(not has_violation), min_overlap_s=min_overlap_s)
    res["neg_count"] = fp_cls["neg_count"]
    res["oow_count"] = fp_cls["oow_count"]
    res["fragment_count"] = fp_cls["fragment_count"]
    res["is_violation_hit"] = res["tp"] > 0
    res["flags"] = _flags(res, has_violation)
    return res


def _flags(res, has_violation):
    flags = []
    if has_violation:
        flags.append("is_violation✓" if res["tp"] > 0 else "is_violation✗(漏)")
        if res["fn"] > 0:
            flags.append(f"fn={res['fn']}")
        if res["fp"] > 0:
            flags.append(f"fp={res['fp']}")
    else:
        flags.append("负例✓(未发现违章)" if res["fp"] == 0 else "负例误报✗")
    if res["plate_total"] > 0:
        flags.append(f"车牌{res['plate_hits']}/{res['plate_total']}")
    return flags


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt_span(s, e):
    return f"[{s:.1f}-{e:.1f}s]"


# ----------------------------------------------------------------------------
# 重跑管线(--fresh; cv2 在此处懒导入, 避免模块加载拖入重依赖)
# ----------------------------------------------------------------------------
def run_fresh(videos, output_root):
    from redlight.infrastructure.config import load_config
    from redlight.app.cli import run
    from redlight.models.crosswalk_v2 import CrosswalkDetectorV2

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    # 报告需要标注视频 + 截图 + CSV + COT(确定性渲染, 不改判定)
    cfg.output.annotated_video = True
    cfg.output.csv_report = True
    cfg.output.evidence_images = True
    # 每个视频新建 detector: 与 eval_violations._run_pipeline 一致(每视频 fresh 实例)。
    # CrosswalkDetectorV2 的 _accum 时序缓冲按首帧分辨率初始化, 跨视频复用会
    #  (a) 分辨率变化帧 shape 不匹配崩溃 (b) 同分辨率视频间污染, 均偏离 0.889 基线口径。
    for v in videos:
        vd = CrosswalkDetectorV2(cfg)
        out = os.path.join(output_root, f"run_{v}_{PRESET}")
        print(f"[fresh] {v} -> {out}")
        run(cfg, os.path.join(ROOT, "input_video", f"{v}.mp4"), out,
            preset=PRESET, cot=True, crosswalk_detector=vd, occ_denom="box")
    print("[fresh] 全部视频重跑完成")


# ----------------------------------------------------------------------------
# HTML 渲染
# ----------------------------------------------------------------------------
def _video_block(run_dir, rel_run, video):
    mp4 = os.path.join(run_dir, "annotated.mp4")
    if not os.path.exists(mp4):
        return f"<p class='warn'>未找到 annotated.mp4(需在 --fresh 时生成)</p>"
    return (f"<video src='{esc(rel_run)}/annotated.mp4' controls muted playsinline "
            f"style='max-width:640px;background:#000'></video>")


def _evidence_block(event, run_dir, rel_run):
    ev = event.get("evidence_image", "")
    if not ev or not os.path.exists(ev):
        return ""
    rel = os.path.relpath(ev, run_dir)
    return (f"<figure style='display:inline-block;margin:4px'>"
            f"<img src='{esc(rel_run)}/{esc(rel)}' style='height:120px;border:1px solid #ccc'/>"
            f"<figcaption>tid{esc(event.get('track_id',''))} "
            f"{esc(normalize_plate(event.get('plate','')))} "
            f"{_fmt_span(event['start_ts'], event['end_ts'])}</figcaption></figure>")


def _cot_block(run_dir):
    cot_dir = os.path.join(run_dir, "cot")
    md = None
    if os.path.isdir(cot_dir):
        for fn in os.listdir(cot_dir):
            if fn.startswith("COT_") and fn.endswith(".md"):
                md = os.path.join(cot_dir, fn)
                break
    if not md:
        return "<p class='warn'>未找到 COT_*.md(需 --fresh 时 cot=True)</p>"
    with open(md, encoding="utf-8") as f:
        text = f.read()
    return f"<pre class='cot'>{esc(text)}</pre>"


def _gt_side_by_side(video, src_anno, events_rows):
    free = esc(src_anno.get(video, "(无原始描述)"))
    struct = [r for r in events_rows if (r.get("video") or "").strip() == video]
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{esc(r.get(k,''))}</td>" for k in
                         ["start_s", "end_s", "light_state", "is_violation", "violating_plates", "note"])
        + "</tr>" for r in struct)
    return f"""
    <div class='gt-two'>
      <div class='gt-col'>
        <h4>原始 free-text GT (source/annotation.csv)</h4>
        <div class='freetext'>{free}</div>
      </div>
      <div class='gt-col'>
        <h4>结构化 GT (events.csv)</h4>
        <table class='mini'>
          <tr><th>start</th><th>end</th><th>灯态</th><th>is_viol</th><th>车牌</th><th>note</th></tr>
          {rows_html}
        </table>
      </div>
    </div>"""


def render_html(videos, runs, events_csv, videos_csv, source_csv, out_path,
                min_overlap_s, bundle_dir=None):
    gt = load_violation_gt(events_csv)
    meta = load_video_metadata(videos_csv)
    src_anno = load_source_annotation(source_csv)
    events_rows = load_events_rows(events_csv)

    per_video = {}
    for v in videos:
        run_dir = runs[v]
        rel_run = os.path.relpath(run_dir, os.path.dirname(out_path))
        if bundle_dir:
            rel_run = os.path.relpath(os.path.join(bundle_dir, os.path.basename(run_dir)),
                                      os.path.dirname(out_path))
        events = read_violations(run_dir)
        confirmed = [e for e in events if e["status"] == "confirmed"]
        has_v = meta.get(v, False)
        res = summarize_video(confirmed, gt.get(v, []), has_v, min_overlap_s)
        per_video[v] = {
            "run_dir": run_dir, "rel_run": rel_run, "events": events,
            "confirmed": confirmed, "has_v": has_v, "res": res,
        }

    agg = aggregate([per_video[v]["res"] for v in videos])

    # ---- 总表 ----
    rows = ""
    for v in videos:
        pv = per_video[v]
        res = pv["res"]
        vmark = "✓" if res["is_violation_hit"] else ("✗" if pv["has_v"] else "—")
        plates = ", ".join(normalize_plate(e["plate"]) for e in pv["confirmed"] if e["plate"]) or "—"
        times = ", ".join(_fmt_span(e["start_ts"], e["end_ts"]) for e in pv["confirmed"]) or "—"
        flags = " ".join(esc(f) for f in res["flags"])
        rows += f"""
        <tr>
          <td>{esc(v)}</td>
          <td>{'是' if pv['confirmed'] else '否'}</td>
          <td>{esc(plates)}</td>
          <td>{esc(times)}</td>
          <td class='{'ok' if vmark=='✓' else ('bad' if vmark=='✗' else 'na')}'>{vmark}</td>
          <td class='flags'>{flags}</td>
        </tr>"""

    # ---- 逐视频 drill-down ----
    sections = ""
    for v in videos:
        pv = per_video[v]
        res = pv["res"]
        vmark = "✓" if res["is_violation_hit"] else ("✗" if pv["has_v"] else "—")
        # 违章车辆列表
        veh_rows = ""
        for e in pv["confirmed"]:
            veh_rows += f"""<tr>
              <td>{esc(normalize_plate(e['plate']) or '(无牌)')}</td>
              <td>{_fmt_span(e['start_ts'], e['end_ts'])}</td>
              <td>{esc(e['light_state'])}</td>
              <td>{esc(e['vehicle_class'])}</td>
              <td>{_evidence_block(e, pv['run_dir'], pv['rel_run'])}</td>
            </tr>"""
        if not veh_rows:
            veh_rows = "<tr><td colspan=5>未发现违章(confirmed 事件为空)</td></tr>"

        sec = f"""
        <details class='video-sec'>
          <summary><b>{esc(v)}</b> — {'有违章' if pv['confirmed'] else '无违章'}
            &nbsp; GT {vmark} &nbsp; <span class='flags'>{' '.join(esc(f) for f in res['flags'])}</span></summary>
          <div class='sec-body'>
            <h4>识别结果 drill-down</h4>
            <p>是否发现违章: <b>{'是' if pv['confirmed'] else '否'}</b>
               (violations.csv confirmed 数={len(pv['confirmed'])})</p>
            <table class='mini'>
              <tr><th>车牌</th><th>时间窗</th><th>灯态</th><th>车型</th><th>证据截图</th></tr>
              {veh_rows}
            </table>
            <h4>完整识别视频 (annotated.mp4)</h4>
            {_video_block(pv['run_dir'], pv['rel_run'], v)}
            <h4>小作文 (COT)</h4>
            {_cot_block(pv['run_dir'])}
            <h4>GT 对比(两者都要)</h4>
            {_gt_side_by_side(v, src_anno, events_rows)}
            <div class='match'>
              <b>自动判定:</b> tp={res['tp']} fp={res['fp']} fn={res['fn']}
              | 车牌命中 {res['plate_hits']}/{res['plate_total']}
              | 覆盖 {res['mean_coverage']}
              | 真误报 {res['neg_count']+res['oow_count']} / 碎片 {res['fragment_count']}
            </div>
          </div>
        </details>"""
        sections += sec

    html = f"""<!DOCTYPE html>
<html lang='zh-CN'><head><meta charset='utf-8'>
<title>斑马线违章识别报告</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,'PingFang SC',sans-serif;margin:0;padding:20px;background:#fafafa;color:#222}}
  h1{{font-size:20px}} h4{{margin:14px 0 6px;color:#0a4}}
  table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #ddd;padding:5px 8px;text-align:left}}
  th{{background:#eef}} .mini{{font-size:12px;margin:6px 0}}
  .ok{{color:#0a0;font-weight:bold}} .bad{{color:#c00;font-weight:bold}} .na{{color:#999}}
  .flags{{color:#066;font-size:12px}}
  .warn{{color:#c60}}
  .video-sec{{border:1px solid #ccc;border-radius:6px;margin:8px 0;padding:6px 10px;background:#fff}}
  .sec-body{{margin:8px 0}}
  .gt-two{{display:flex;gap:16px;flex-wrap:wrap}} .gt-col{{flex:1;min-width:300px}}
  .freetext{{white-space:pre-wrap;background:#f5f5f5;padding:8px;border-radius:4px;font-size:12px}}
  pre.cot{{white-space:pre-wrap;background:#f0f4ff;padding:8px;border-left:3px solid #06c;font-size:12px;max-height:360px;overflow:auto}}
  .match{{background:#eef;padding:6px;border-radius:4px;margin-top:8px;font-size:13px}}
  figure{{margin:2px}} figcaption{{font-size:10px;color:#666}}
  .summary{{background:#fff;border:1px solid #ccc;border-radius:6px;padding:10px;margin:10px 0}}
</style></head><body>
<h1>斑马线行人绿灯压线 — 可复现识别报告</h1>
<p class='summary'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  | 管线: v2 + occ_denom=box + preset={PRESET}
  | min_overlap={min_overlap_s}s
  | 聚合: tp={agg['tp']} fp={agg['fp']} fn={agg['fn']}
  | 事件级 F1={agg['f1']} 覆盖={agg['mean_coverage']}
  | 车牌 {agg['plate_hits']}/{agg['plate_total']}
  | 真误报 {agg['true_fp_total']} / 碎片 {agg['fragment_total']}
  <br><i>基线(旧GT-F1)=0.889 保留(Phase 2 merge 为 no-op)。本报告 GT 对比口径与 eval_violations 一致。</i></p>

<h3>总表 (11 视频)</h3>
<table>
  <tr><th>视频</th><th>违章 Y/N</th><th>车牌</th><th>时间</th><th>GT</th><th>子标记</th></tr>
  {rows}
</table>

<h3>逐视频 drill-down + GT 对比</h3>
{sections}

<p style='color:#999;font-size:11px'>本报告只读反映管线与 GT, 不改判定逻辑/GT/canonical。
视频与截图引用相对路径; --bundle 可复制资产成可迁移归档。</p>
</body></html>"""
    return html, agg


def main():
    ap = argparse.ArgumentParser(description="斑马线违章识别结果报告(单 HTML + GT 对比)")
    ap.add_argument("--fresh", action="store_true", help="重跑全 11 视频管线(默认读缓存)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "output", "violation_report.html"))
    ap.add_argument("--bundle", action="store_true",
                    help="复制资产(annotated.mp4/evidence/cot)成可迁移归档(默认不拷, ~1GB)")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--events", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--videos-csv", default=os.path.join(ROOT, "datasets", "gt", "videos.csv"))
    ap.add_argument("--source", default=os.path.join(ROOT, "datasets", "gt", "source", "annotation.csv"))
    ap.add_argument("--min-overlap", type=float, default=MIN_OVERLAP_S)
    args = ap.parse_args()

    videos = args.videos or VIDEOS
    output_root = os.path.join(ROOT, "data", "output")

    if args.fresh:
        run_fresh(videos, output_root)

    # 发现缓存
    runs = {}
    missing = []
    for v in videos:
        d = find_run_dir(v, output_root)
        if d is None:
            missing.append(v)
        else:
            runs[v] = d
    if missing:
        print(f"缓存缺失: {missing} 的 run_*/ 不存在。请加 --fresh 重跑管线。", file=sys.stderr)
        sys.exit(2)

    bundle_dir = None
    if args.bundle:
        bundle_dir = os.path.join(output_root, "report_bundle")
        os.makedirs(bundle_dir, exist_ok=True)
        for v in videos:
            src = runs[v]
            dst = os.path.join(bundle_dir, os.path.basename(src))
            if os.path.abspath(dst) != os.path.abspath(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

    html, agg = render_html(videos, runs, args.events, args.videos_csv,
                            args.source, args.out, args.min_overlap, bundle_dir)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[报告] 已写出 {os.path.abspath(args.out)}")
    print(f"[聚合] tp={agg['tp']} fp={agg['fp']} fn={agg['fn']} "
          f"F1={agg['f1']} 覆盖={agg['mean_coverage']} 车牌={agg['plate_hits']}/{agg['plate_total']}")


if __name__ == "__main__":
    main()
