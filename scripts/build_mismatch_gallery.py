#!/usr/bin/env python3
"""生成 mismatch 画廊 HTML: 供人工目视确认预测错误帧。

每张卡片展示:
- 原帧(带先验ROI蓝框 + 预测/GT文字标注)
- 时间戳、帧号、预测、GT、conf
- 按视频分组, 可折叠
"""
import os
import sys
import csv
import json
import base64
from io import BytesIO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np

from redlight.evaluation.frame_dataset import FrameDataset


def load_priors(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = (float(v[0]), float(v[1]), int(v[2]) if len(v) > 2 else 160)
    return out


def draw_mismatch_thumb(frame, prior, pred, gt, t_sec, frame_idx, pred_conf, thumb_w=400):
    """生成带标注的缩略图, 返回 base64 data URL。"""
    h, w = frame.shape[:2]
    scale = thumb_w / w
    thumb_h = int(h * scale)
    thumb = cv2.resize(frame, (thumb_w, thumb_h))

    # 先验 ROI
    if prior:
        px, py, roi_px = prior
        cx_i, cy_i = int(px * thumb_w), int(py * thumb_h)
        r = int(roi_px * scale / 2)
        cv2.rectangle(thumb, (cx_i - r, cy_i - r), (cx_i + r, cy_i + r), (255, 0, 0), 2)
        cv2.circle(thumb, (cx_i, cy_i), 2, (255, 0, 0), -1)

    # 文字标注
    text = f"t={t_sec:.1f}s  pred={pred}  gt={gt}  conf={pred_conf}"
    color = (0, 255, 0) if pred == gt else (0, 0, 255)
    cv2.putText(thumb, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 底部条: GT 颜色
    bar = np.zeros((22, thumb_w, 3), dtype=np.uint8)
    gt_color = {"green": (0, 255, 0), "red": (0, 0, 255), "flashing": (0, 255, 255), "unknown": (128, 128, 128)}.get(gt, (128, 128, 128))
    bar[:] = gt_color
    pred_color = {"green": (0, 255, 0), "red": (0, 0, 255), "flashing": (0, 255, 255), "unknown": (128, 128, 128)}.get(pred, (128, 128, 128))
    # 左侧小条表示预测
    bar[:, :thumb_w // 2] = pred_color
    cv2.putText(bar, f"P:{pred}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(bar, f"G:{gt}", (thumb_w // 2 + 5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    thumb = np.vstack([thumb, bar])

    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def boundary_split(mismatches, gt_segs, tol=2.0):
    """拆分 mismatch: 落在 GT 段边界 ±tol 内(过渡) vs 段内部(真实问题)。"""
    bounds = []
    for i in range(len(gt_segs) - 1):
        bounds.append((gt_segs[i][1] + gt_segs[i + 1][0]) / 2.0)
    interior, boundary = [], []
    for m in mismatches:
        t = m["t_sec"]
        near = any(abs(t - b) <= tol for b in bounds)
        (boundary if near else interior).append(m)
    return interior, boundary


def main():
    out_dir = os.path.join(ROOT, "data", "output", "temporal_fusion_eval", "gallery")
    os.makedirs(out_dir, exist_ok=True)

    frames_dir = os.path.join(ROOT, "datasets", "frames")
    dataset = FrameDataset(frames_dir)
    priors = load_priors(os.path.join(ROOT, "configs", "light_priors.json"))

    # 读取 mismatch
    mm_csv = os.path.join(ROOT, "data", "output", "temporal_fusion_eval", "mismatch_all.csv")
    mismatches = []
    with open(mm_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mismatches.append({
                "video": row["video"],
                "t_sec": float(row["t_sec"]),
                "frame_idx": int(row["frame_idx"]),
                "pred": row["pred"],
                "gt": row["gt"],
                "gt_conf": row["gt_conf"],
                "pred_conf": row.get("pred_conf", "0"),
            })

    # 读取 GT 段(用于边界拆分)
    gt_csv = os.path.join(ROOT, "datasets", "gt", "light_states.csv")
    gt_segs = {}
    with open(gt_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row["video"]
            gt_segs.setdefault(v, []).append(
                (float(row["start_s"]), float(row["end_s"]), row["state"], row["confidence"])
            )

    # 每视频预加载帧字典(只加载 mismatch 需要的)
    video_frames = {}
    for v in set(m["video"] for m in mismatches):
        d = {}
        for idx, ts, frame in dataset.iter_video(v):
            d[idx] = (ts, frame)
        video_frames[v] = d

    # 按视频分组
    by_video = {}
    for m in mismatches:
        by_video.setdefault(m["video"], []).append(m)

    # 生成 HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Mismatch Gallery - Temporal Fusion Eval</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }
h1 { font-size: 20px; margin-bottom: 8px; }
h2 { font-size: 16px; margin: 24px 0 8px 0; cursor: pointer; color: #4fc3f7; }
h2:hover { color: #81d4fa; }
.summary { color: #aaa; font-size: 13px; margin-bottom: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
.card { background: #2a2a2a; border-radius: 6px; overflow: hidden; border: 1px solid #333; }
.card img { width: 100%; display: block; }
.card .info { padding: 8px 10px; font-size: 12px; color: #bbb; }
.card .tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; margin-right: 4px; }
.tag-pred-green { background: #1b5e20; color: #a5d6a7; }
.tag-pred-red { background: #b71c1c; color: #ef9a9a; }
.tag-pred-flashing { background: #f57f17; color: #fff59d; }
.tag-gt-green { background: #2e7d32; color: #c8e6c9; }
.tag-gt-red { background: #c62828; color: #ffcdd2; }
.tag-confirmed { background: #1565c0; color: #bbdefb; }
.tag-tentative { background: #6a1b9a; color: #e1bee7; }
.tag-interior { border: 1px solid #ef5350; }
.tag-boundary { border: 1px solid #ffa726; }
.filter-bar { margin: 12px 0; }
.filter-bar button { background: #333; color: #ccc; border: 1px solid #555; padding: 4px 12px; margin-right: 6px; border-radius: 4px; cursor: pointer; }
.filter-bar button.active { background: #4fc3f7; color: #111; border-color: #4fc3f7; }
.collapsed .grid { display: none; }
</style>
</head>
<body>
<h1>Temporal Fusion Mismatch Gallery</h1>
<p class="summary">每张卡片 = 预测≠GT 的帧。蓝框 = 先验ROI。底部左半色 = 预测, 右半色 = GT。</p>
<div class="filter-bar">
  <button onclick="filterAll()" id="btn-all" class="active">全部</button>
  <button onclick="filterConfirmed()" id="btn-confirmed">仅 confirmed</button>
  <button onclick="filterInterior()" id="btn-interior">段内真实问题</button>
</div>
""")

    total_cards = 0
    for v in sorted(by_video.keys()):
        v_mms = by_video[v]
        interior, boundary = boundary_split(v_mms, gt_segs.get(v, []), tol=2.0)
        interior_ids = {id(m) for m in interior}

        html_parts.append(f'<h2 onclick="toggle(this)">{v} ({len(v_mms)} mismatch, {len(interior)} 段内 / {len(boundary)} 边界)</h2>')
        html_parts.append('<div class="grid">')

        prior = priors.get(v)
        frames_dict = video_frames[v]

        for m in v_mms:
            ts, frame = frames_dict.get(m["frame_idx"], (0, None))
            if frame is None:
                continue
            is_interior = id(m) in interior_ids
            tag_cls = "tag-interior" if is_interior else "tag-boundary"
            conf_tag = "tag-confirmed" if m["gt_conf"] == "confirmed" else "tag-tentative"
            pred_tag = f"tag-pred-{m['pred']}"
            gt_tag = f"tag-gt-{m['gt']}"

            img_b64 = draw_mismatch_thumb(frame, prior, m["pred"], m["gt"], m["t_sec"], m["frame_idx"], m["pred_conf"])

            html_parts.append(f"""<div class="card" data-conf="{m['gt_conf']}" data-interior="{'true' if is_interior else 'false'}">
  <img src="{img_b64}" loading="lazy">
  <div class="info">
    <span class="tag {pred_tag} {tag_cls}">P:{m['pred']}</span>
    <span class="tag {gt_tag}">G:{m['gt']}</span>
    <span class="tag {conf_tag}">{m['gt_conf']}</span>
    <span class="tag">t={m['t_sec']:.1f}s idx={m['frame_idx']}</span>
    <span class="tag">conf={m['pred_conf']}</span>
  </div>
</div>""")
            total_cards += 1

        html_parts.append('</div>')

    html_parts.append("""
<script>
function toggle(el) { el.parentElement.classList.toggle('collapsed'); }
function filterCards(fn) {
  document.querySelectorAll('.card').forEach(c => c.style.display = fn(c) ? '' : 'none');
  document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
}
function filterAll() { filterCards(() => true); document.getElementById('btn-all').classList.add('active'); }
function filterConfirmed() { filterCards(c => c.dataset.conf === 'confirmed'); document.getElementById('btn-confirmed').classList.add('active'); }
function filterInterior() { filterCards(c => c.dataset.interior === 'true'); document.getElementById('btn-interior').classList.add('active'); }
</script>
</body>
</html>
""")

    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"画廊已生成: {out_path}")
    print(f"共 {total_cards} 张 mismatch 卡片")
    print("打开方式: 用浏览器打开上述 HTML 文件")


if __name__ == "__main__":
    main()
