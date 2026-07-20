"""生成 light-state 重训数据集人工抽检画廊 v3 (HTML, 自包含 base64, 免服务)。

v3 重设计(解决 Jacob 反馈的"上下图不对齐"):
  每张卡只输出一张预合成对比图 —— 左半=全帧缩略图(prior ROI 蓝框 + 抠图区域红框),
  右半=实际抠图放大。服务端 cv2 预渲染为单张 JPEG, 浏览器只显示一张 <img>,
  零 CSS 对齐风险。

用途/抽样/工作流 同 v2。
"""
import os
import sys
import csv
import json
import base64
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np
from redlight.data_pipeline.ped_signal_dataset import load_labeled_crops
from redlight.infrastructure.image_utils import robust_imread
from redlight.evaluation import gt_lookup
from redlight.evaluation.frame_dataset import FrameDataset

LABELS = ["walk", "stand", "off"]
CONF_COLOR = {"walk": "#16a34a", "stand": "#d97706", "off": "#64748b"}
DARK_GREEN_VIDEOS = {"违章06", "违章07"}
SHORT_GREEN_VIDEO = "违章04"
BOUNDARY_SEC = 2.0

# ---- 预合成对比图的尺寸常量 ----
FRAME_W = 420          # 左侧全帧缩略图宽度(px)
CROP_ZOOM = 4          # 右侧 crop 放大倍数(相对于原始 crop 尺寸)
PAD = 8                # 左右间距 + 外边距
TXT_H = 26             # 顶部文字行高


def _load_priors(path):
    """同 mine_classifier_retrain._load_priors: (px, py, roi_px) 归一化+像素。"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {v: (a[0], a[1], (a[2] if len(a) > 2 else 160)) for v, a in raw.items()
            if isinstance(a, (list, tuple)) and len(a) >= 2}


def arr_to_b64(arr, quality=72):
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _boundary_ts_by_video(light_states_csv):
    segs = gt_lookup.load_light_state_csv(light_states_csv)
    out = {}
    for v, seglist in segs.items():
        bts = []
        for i, (a, b, st, conf) in enumerate(seglist):
            prev = seglist[i - 1][2] if i > 0 else None
            nxt = seglist[i + 1][2] if i + 1 < len(seglist) else None
            involves_green = (st == "green") or (prev == "green") or (nxt == "green")
            if involves_green:
                bts.append(a)
                bts.append(b)
        out[v] = bts
    return out


def _is_boundary(video, ts, boundary_map):
    for b in boundary_map.get(video, []):
        if abs(ts - b) <= BOUNDARY_SEC:
            return True
    return False


def _composite_card(frame, crop_img, video, prior, x1, y1, x2, y2, label, src, ts):
    """
    预合成一张对比 JPEG: [文字行 | 左=全帧(带框) | 右=crop(放大)] → 单张图片。

    返回 numpy array (BGR), 或读图失败返回 None。
    """
    if frame is None and crop_img is None:
        return None

    # ---- 左侧: 全帧缩略图(带框叠加) ----
    if frame is not None:
        h, w = frame.shape[:2]
        sc = FRAME_W / max(w, 1)
        S = lambda v: int(round(v * sc))
        frame_thumb = cv2.resize(frame, (FRAME_W, int(h * sc)))

        # Prior ROI 蓝框 + 十字
        if prior is not None:
            px, py, roi_px = prior
            pcx, pcy = int(px * w), int(py * h)
            half = roi_px // 2
            bx1, by1 = max(0, pcx - half), max(0, pcy - half)
            bx2, by2 = min(w, pcx + half), min(h, pcy + half)
            cv2.rectangle(frame_thumb, (S(bx1), S(by1)), (S(bx2), S(by2)),
                          (255, 140, 0), 2)       # 蓝=先验 ROI
            cv2.drawMarker(frame_thumb, (S(pcx), S(pcy)), (255, 140, 0),
                           cv2.MARKER_CROSS, 14, 2)

        # Crop 区域 红框
        cv2.rectangle(frame_thumb, (S(x1), S(y1)), (S(x2), S(y2)),
                      (220, 38, 38), 2)           # 红=抠图框

        # 标注 source 类型在图上(小字)
        src_tag = src.replace("_", " ")[:12]
        cv2.putText(frame_thumb, src_tag, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    else:
        frame_thumb = np.ones((200, FRAME_W, 3), np.uint8) * 30
        cv2.putText(frame_thumb, "no frame", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

    # ---- 右侧: crop 放大 ----
    if crop_img is not None:
        ch, cw = crop_img.shape[:2]
        crop_big = cv2.resize(crop_img, (cw * CROP_ZOOM, ch * CROP_ZOOM))
    else:
        crop_big = np.ones((200, 160, 3), np.uint8) * 30
        cv2.putText(crop_big, "no crop", (5, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # ---- 合成画布: [文字 | 左帧 | 右crop] ----
    total_h = max(frame_thumb.shape[0], crop_big.shape[0]) + PAD * 2 + TXT_H
    total_w = frame_thumb.shape[1] + crop_big.shape[1] + PAD * 3
    canvas = np.ones((total_h, total_w, 3), np.uint8) * 245  # 浅灰底

    # 放入左右两图
    y_off = TXT_H + PAD
    canvas[y_off:y_off + frame_thumb.shape[0],
           PAD:PAD + frame_thumb.shape[1]] = frame_thumb
    cx_off = PAD * 2 + frame_thumb.shape[1]
    canvas[y_off:y_off + crop_big.shape[0],
           cx_off:cx_off + crop_big.shape[1]] = crop_big

    # 顶部信息条
    info = f"[{video}] t={ts}s  label={label}  src={src}"
    info2 = f"red_box=({x1},{y1},{x2},{y2})"
    if prior:
        px, py, _ = prior
        fh, fw = frame.shape[:2] if frame is not None else (1, 1)
        info2 += f"  prior_center=({int(px*fw)},{int(py*fh)})"

    cv2.putText(canvas, info, (PAD, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    cv2.putText(canvas, info2, (PAD, TXT_H - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (100, 100, 100), 1)

    return canvas


def main():
    ap = argparse.ArgumentParser(description="生成 light-state 重训数据集抽检画廊 v3")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "classifier_retrain", "labels.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "classifier_retrain", "classifier_retrain_gallery.html"))
    ap.add_argument("--light-states", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--strategy", default="sample", choices=["sample", "all"])
    ap.add_argument("--sample-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    rows = load_labeled_crops(args.labels)
    dataset_root = os.path.dirname(os.path.abspath(args.labels))
    boundary_map = _boundary_ts_by_video(args.light_states)
    priors = _load_priors(os.path.join(ROOT, "configs", "light_priors.json"))
    fd = FrameDataset(os.path.join(ROOT, "datasets", "frames"))

    # 分层抽样
    if args.strategy == "all":
        sel = list(rows)
    else:
        impostor, dark_green, short_green, boundary, rest = [], [], [], [], []
        for r in rows:
            v, ts, src, lab = r["video"], float(r["frame_ts"]), r["source"], r["label"]
            if src in ("impostor", "impostor_outside"):
                impostor.append(r)
            elif v in DARK_GREEN_VIDEOS and lab == "walk":
                dark_green.append(r)
            elif v == SHORT_GREEN_VIDEO and lab == "walk":
                short_green.append(r)
            elif _is_boundary(v, ts, boundary_map):
                boundary.append(r)
            else:
                rest.append(r)
        sampled_rest = random.sample(rest, max(0, int(len(rest) * args.sample_frac))) if rest else []
        sel = impostor + dark_green + short_green + boundary + sampled_rest

    sel.sort(key=lambda r: (r["video"], float(r["frame_ts"])))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cards_by_video = {}
    skipped = 0

    for r in sel:
        # 读 crop 图
        crop_img = robust_imread(r["crop_path"])
        if crop_img is None:
            skipped += 1
            continue

        rel_cp = os.path.relpath(r["crop_path"], dataset_root)
        cur = r["label"]
        color = CONF_COLOR.get(cur, "#475569")
        tag = "边界" if _is_boundary(r["video"], float(r["frame_ts"]), boundary_map) else ""

        # 读全帧
        frame = None
        fi_val = r.get("fi", "")
        if str(fi_val).isdigit():
            frame = fd.get_frame(r["video"], int(fi_val))

        # 坐标
        x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1","y1","x2","y2"))

        # 预合成单张对比图
        composite = _composite_card(
            frame, crop_img, r["video"],
            priors.get(r["video"]),
            x1, y1, x2, y2, cur, r["source"], r["frame_ts"]
        )

        if composite is None:
            skipped += 1
            continue

        comp_b64 = arr_to_b64(composite, quality=70)
        if comp_b64 is None:
            skipped += 1
            continue

        # ---- 卡片 HTML（只有一张 img） ----
        card = f"""
        <div class="card" data-cp="{rel_cp}" data-video="{r['video']}" data-t="{r['frame_ts']}" data-cur="{cur}">
          <img src="data:image/jpeg;base64,{comp_b64}" loading="lazy"/>
          <div class="meta">{r['video']} t={r['frame_ts']}s <span class="cur" style="background:{color}">{cur}</span>
            <span class="src">{r['source']}</span>{('<span class="bd">'+tag+'</span>') if tag else ''}</div>
          <div class="fb">
            <select class="verdict">
              <option value="">--改标--</option>
              <option value="walk"{' selected' if cur=='walk' else ''}>walk(绿灯/过街)</option>
              <option value="stand"{' selected' if cur=='stand' else ''}>stand(红灯/站立)</option>
              <option value="off"{' selected' if cur=='off' else ''}>off(非信号/impostor)</option>
              <option value="delete">删除(难判/废)</option>
            </select>
            <button class="save">保存</button>
            <span class="status"></span>
          </div>
        </div>"""
        cards_by_video.setdefault(r["video"], []).append(card)

    # ---- 组装 HTML ----
    cards_html = ""
    for v in sorted(cards_by_video):
        cards_html += f'<h2>{v} <span class="cnt">({len(cards_by_video[v])}张)</span></h2><div class="grid">{"".join(cards_by_video[v])}</div>'

    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>light-state 重训数据集抽检</title>
<style>
body{font-family:-apple-system,sans-serif;background:#f1f5f9;margin:0;padding:0 16px 40px;}
#tb{position:sticky;top:0;background:#0f172a;color:#fff;padding:8px 14px;display:flex;gap:12px;align-items:center;z-index:10;margin:0 -16px 14px;font-size:13px;flex-wrap:wrap;}
#export{margin-left:auto;background:#fff;color:#0f172a;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px;}
h1{color:#0f172a;margin:6px 0 2px;font-size:18px;}
.intro{color:#475569;font-size:12px;margin:0 0 12px;line-height:1.55;}
h2{color:#0f172a;font-size:15px;margin:16px 0 5px;} .cnt{color:#64748b;font-weight:400;font-size:12px;}
.grid{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;}
/* 卡片: 单张预合成图, 无 CSS 对齐问题 */
.card{border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;width:auto;max-width:760px;background:#fff;}
.card.saved{box-shadow:0 0 0 2px #22c55e inset;}
.card img{display:block;width:100%;height:auto;background:#000;}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#334155;margin:4px 0 12px;background:#fff;padding:6px 10px;border:1px solid #e2e8f0;border-radius:6px;}
.legend b{font-weight:700;}
.legend .sw{display:inline-block;width:12px;height:12px;vertical-align:-2px;margin-right:3px;border-radius:2px;}
.sw-prior{background:#ff8c00;border:1px solid #cc7000;} .sw-crop{background:#dc2626;border:1px solid #b91c1c;}
.meta{font-size:10.5px;padding:3px 6px;color:#334155;display:flex;gap:6px;align-items:center;flex-wrap:wrap;}
.cur{display:inline-block;color:#fff;font-size:9px;padding:1px 5px;border-radius:6px;font-weight:600;}
.src{font-size:9px;color:#94a3b8;}
.bd{display:inline-block;color:#fff;font-size:8px;padding:1px 4px;border-radius:4px;background:#9333ea;}
.fb{padding:5px 6px;display:flex;gap:4px;align-items:center;font-size:10.5px;background:#f8fafc;border-top:1px solid #e2e8f0;}
.fb select{font-size:10.5px;padding:2px 4px;border:1px solid #cbd5e1;border-radius:4px;}
.fb button{background:#0f172a;color:#fff;border:none;border-radius:4px;padding:3px 8px;cursor:pointer;font-size:10.5px;}
.fb .status{font-size:9px;}
</style></head><body>
<div id="tb"><b>light-state 重训数据集抽检</b><span id="cnt">已保存 0</span><span class="hint">impostor+暗绿+04短绿+段边界全核 其余随机20%</span><button id="export">导出反馈JSON</button></div>
<h1>light-state 重训数据集抽检</h1>
<div class="legend">
  <span><span class="sw sw-prior"></span><b>蓝框</b>=先验 ROI（模型固定 gaze 处）</span>
  <span><span class="sw sw-crop"></span><b>红框</b>=实际抠图区（prior_roi≈蓝框；impostor_outside=引擎绿斑，可能远离蓝框）</span>
</div>
<p class="intro">每张卡=<b>一张预合成图</b>: 左=全帧缩略图(带蓝框+红框)，右=crop 放大。一眼可判「红框里的内容 ≈ 右边放大的内容」是否一致。<br/>
<b>prior_roi</b>: 红框≈蓝框附近，crop 内容应与红框区域匹配。<b>impostor_outside</b>: 红框可能远离蓝框——引擎在别处捡到绿斑，你只需判它是真信号还是非信号。<br/>
操作：选正确标签→保存→全标完点导出→跑 apply_classifier_retrain_feedback.py。</p>
""" + cards_html + """
<script>
const saved=new Map();
function upd(){document.getElementById('cnt').textContent='已保存 '+saved.size;}
document.querySelectorAll('.card .save').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const card=btn.closest('.card');
    const v=card.querySelector('.verdict').value;
    const st=card.querySelector('.status');
    if(!v){st.textContent='请先选标签';st.style.color='#dc2626';return;}
    saved.set(card.dataset.cp,{label:v,video:card.dataset.video,t:card.dataset.t,cur:card.dataset.cur});
    st.textContent='✓';st.style.color='#16a34a';card.classList.add('saved');upd();
  });
});
document.getElementById('export').addEventListener('click',()=>{
  const items=[];for(const [k,v] of saved) items.push({crop_path:k,...v});
  const blob=new Blob([JSON.stringify(items,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='classifier_retrain_feedback.json';a.click();
});
</script></body></html>"""

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] 画廊 -> {args.out}")
    print(f"  展示 {len(sel)} 张 (跳过 {skipped} 张); 全量 {len(rows)} 张")
    print(f"  分层: impostor+暗绿+04短绿+段边界全核 + 其余随机{args.sample_frac:.0%}")
    print(f"  每张卡=预合成对比图(左全帧+右crop), 无CSS对齐风险")


if __name__ == "__main__":
    main()
