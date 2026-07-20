"""生成 light-state 重训数据集人工抽检画廊 v4 (聚类去重 + 批量标注 + 固定尺寸)。

v4 相对 v3 的三大改进 (Jacob 反馈"效率太低: 绝大多数 off / 大量雷同 / 图忽大忽小"):
  1. 时间+空间连续聚类去重: 同 (video, source, label) 内, 时间连续(≤GAP)且抠图
     中心接近(≤DIST)的帧 = 同一串采样, 折叠成一张代表卡 x N, 标代表 = 背书整簇。
     (dHash 对本数据无效: ROI 内行人/车/闪灯致相邻帧像素差异中位数达 15-27。)
     walk/stand(prior_roi 固定位置连续帧) 压成 ~21 簇; off(impostor 位置分散) 约 1250 簇。
     代表卡下方给出簇内成员缩略图条, 供抽查簇内一致性。
  2. 批量标注: 顶部"接受全部 walk/stand(已自洽)" / "接受全部 off(先全标 off 再挑非off)"
     一键背书; 改下拉即自动保存(减少点击); 支持按标签筛选只看 off。
  3. 固定尺寸合成图: 左帧 letterbox 360x202 + 右 crop letterbox 202x202,
     每张卡合成图统一 586x244, 不再忽大忽小。

导出 classifier_retrain_feedback.json: 展开簇成员 -> 每 crop_path 一条 {crop_path,label}。
下游 apply_classifier_retrain_feedback.py 按 crop_path 匹配 labels.csv 改 label + verified=1。
crop_path 使用 labels.csv 原始相对串(如 违章01/xxx.jpg), 与 apply 匹配一致。
"""
import os
import sys
import csv
import json
import html
import base64
import argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np
from redlight.infrastructure.image_utils import robust_imread
from redlight.evaluation import gt_lookup
from redlight.evaluation.frame_dataset import FrameDataset

LABELS = ["walk", "stand", "off"]
CONF_COLOR = {"walk": "#16a34a", "stand": "#d97706", "off": "#64748b"}
DARK_GREEN_VIDEOS = {"违章06", "违章07"}
SHORT_GREEN_VIDEO = "违章04"
BOUNDARY_SEC = 2.0

# ---- 固定尺寸常量(解决"图忽大忽小") ----
FRAME_W, FRAME_H = 360, 202     # 左侧全帧 letterbox 目标框
CROP_SIDE = 202                 # 右侧 crop letterbox 目标框(正方形)
PAD = 8
TXT_H = 24
THUMB = 46                      # 簇成员缩略图边长
MAX_THUMBS = 7                  # 每簇最多展示的成员缩略图数

# 时间+空间连续聚类参数(不用 dHash: 对本数据无效)
GAP_S = 1.0                     # 相邻帧时间差 ≤ 此值(秒)视为连续
DIST_PX = 50                    # 抠图中心位移 ≤ 此值(px)视为同一目标


# ------------------------------------------------------------------ priors
def _load_priors(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {v: (a[0], a[1], (a[2] if len(a) > 2 else 160)) for v, a in raw.items()
            if isinstance(a, (list, tuple)) and len(a) >= 2}


# ------------------------------------------------------------------ 边界帧
def _boundary_ts_by_video(light_states_csv):
    segs = gt_lookup.load_light_state_csv(light_states_csv)
    out = {}
    for v, seglist in segs.items():
        bts = []
        for i, (a, b, st, conf) in enumerate(seglist):
            prev = seglist[i - 1][2] if i > 0 else None
            nxt = seglist[i + 1][2] if i + 1 < len(seglist) else None
            if (st == "green") or (prev == "green") or (nxt == "green"):
                bts.append(a); bts.append(b)
        out[v] = bts
    return out


def _is_boundary(video, ts, boundary_map):
    for b in boundary_map.get(video, []):
        if abs(ts - b) <= BOUNDARY_SEC:
            return True
    return False


# ------------------------------------------------------------------ 图像工具
def arr_to_b64(arr, quality=68):
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _letterbox(img, W, H, bg=20):
    """等比缩放居中 + 填充, 返回 (canvas, scale, ox, oy)。"""
    h, w = img.shape[:2]
    sc = min(W / max(w, 1), H / max(h, 1))
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((H, W, 3), bg, np.uint8)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, sc, ox, oy


# ------------------------------------------------------------------ 合成对比图
def _composite_card(frame, crop_img, prior, x1, y1, x2, y2, label, src, video, ts):
    """固定尺寸对比图: [文字 | 左帧(带框) letterbox 360x202 | 右 crop letterbox 202x202]。"""
    # 左帧
    if frame is not None:
        fthumb, fsc, fox, foy = _letterbox(frame, FRAME_W, FRAME_H)
        FX = lambda v: int(round(v * fsc)) + fox
        FY = lambda v: int(round(v * fsc)) + foy
        h, w = frame.shape[:2]
        if prior is not None:
            px, py, roi_px = prior
            pcx, pcy = int(px * w), int(py * h)
            half = roi_px // 2
            cv2.rectangle(fthumb, (FX(pcx - half), FY(pcy - half)),
                          (FX(pcx + half), FY(pcy + half)), (255, 140, 0), 2)
            cv2.drawMarker(fthumb, (FX(pcx), FY(pcy)), (255, 140, 0),
                           cv2.MARKER_CROSS, 12, 2)
        rx1, ry1, rx2, ry2 = FX(x1), FY(y1), FX(x2), FY(y2)
        cv2.rectangle(fthumb, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
        cl = 11
        for cx_, cy_ in [(rx1, ry1), (rx2, ry1), (rx1, ry2), (rx2, ry2)]:
            dx = cl if cx_ == rx1 else -cl
            dy = cl if cy_ == ry1 else -cl
            cv2.line(fthumb, (cx_, cy_), (cx_ + dx, cy_), (0, 0, 255), 3)
            cv2.line(fthumb, (cx_, cy_), (cx_, cy_ + dy), (0, 0, 255), 3)
    else:
        fthumb = np.full((FRAME_H, FRAME_W, 3), 30, np.uint8)
        cv2.putText(fthumb, "no frame", (10, FRAME_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

    # 右 crop
    if crop_img is not None:
        crop_lb, _, _, _ = _letterbox(crop_img, CROP_SIDE, CROP_SIDE)
    else:
        crop_lb = np.full((CROP_SIDE, CROP_SIDE, 3), 30, np.uint8)
        cv2.putText(crop_lb, "no crop", (10, CROP_SIDE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    W = PAD * 3 + FRAME_W + CROP_SIDE
    H = TXT_H + PAD * 2 + max(FRAME_H, CROP_SIDE)
    canvas = np.full((H, W, 3), 245, np.uint8)
    y0 = TXT_H + PAD
    canvas[y0:y0 + FRAME_H, PAD:PAD + FRAME_W] = fthumb
    cx = PAD * 2 + FRAME_W
    canvas[y0:y0 + CROP_SIDE, cx:cx + CROP_SIDE] = crop_lb
    info = f"{video}  t={ts}s  [{label}]  ({src})"
    cv2.putText(canvas, info, (PAD, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1)
    return canvas


def _thumb_b64(crop_img):
    lb, _, _, _ = _letterbox(crop_img, THUMB, THUMB)
    return arr_to_b64(lb, quality=48)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="生成 light-state 重训数据集抽检画廊 v4")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "classifier_retrain", "labels.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "classifier_retrain", "classifier_retrain_gallery.html"))
    ap.add_argument("--light-states", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--gap", type=float, default=GAP_S)
    ap.add_argument("--dist", type=float, default=DIST_PX)
    args = ap.parse_args()

    dataset_root = os.path.dirname(os.path.abspath(args.labels))
    boundary_map = _boundary_ts_by_video(args.light_states)
    priors = _load_priors(os.path.join(ROOT, "configs", "light_priors.json"))
    fd = FrameDataset(os.path.join(ROOT, "datasets", "frames"))

    # raw 读 labels.csv, 保留原始 crop_path 串(与 apply 匹配)
    rows = []
    with open(args.labels, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # ---- 阶段1: 时间+空间连续聚类(纯 bbox+ts, 不读图) ----
    print(f"[1/2] 时间+空间连续聚类 (gap={args.gap}s dist={args.dist}px) ...")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["video"], r["source"], r["label"])].append(r)

    def _center(r):
        return ((int(float(r["x1"])) + int(float(r["x2"]))) / 2,
                (int(float(r["y1"])) + int(float(r["y2"]))) / 2)

    clusters = []       # 每簇: {"rep","members","video","source","label"}
    for (video, src, label), lst in groups.items():
        lst = sorted(lst, key=lambda r: float(r["frame_ts"]))
        cur, last = None, None
        for r in lst:
            ts = float(r["frame_ts"])
            cx, cy = _center(r)
            if (last is not None and (ts - last[0]) <= args.gap
                    and abs(cx - last[1]) <= args.dist and abs(cy - last[2]) <= args.dist):
                cur["members"].append(r)
            else:
                cur = {"members": [r], "video": video, "source": src, "label": label}
                clusters.append(cur)
            last = (ts, cx, cy)
    # 代表取段中点(最具代表性)
    for c in clusters:
        c["rep"] = c["members"][len(c["members"]) // 2]

    n_img = sum(len(c["members"]) for c in clusters)
    print(f"    {n_img} 张 -> {len(clusters)} 簇 (压缩比 {n_img / max(len(clusters),1):.1f}x)")

    # ---- 阶段2: 每簇合成代表卡 ----
    print(f"[2/2] 合成 {len(clusters)} 张代表卡 ...")
    skipped = 0
    # 视频内排序: off(需审) 优先, 再按时间
    def _clu_sort(c):
        r = c["rep"]
        return (c["video"], 0 if c["label"] == "off" else 1, c["source"], float(r["frame_ts"]))
    clusters.sort(key=_clu_sort)

    cards_by_video = defaultdict(list)
    for c in clusters:
        r = c["rep"]
        crop_img = robust_imread(os.path.join(dataset_root, r["crop_path"]))
        if crop_img is None:
            continue
        frame = None
        fi_val = r.get("fi", "")
        if str(fi_val).isdigit():
            frame = fd.get_frame(r["video"], int(fi_val))
        x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
        comp = _composite_card(frame, crop_img, priors.get(r["video"]),
                               x1, y1, x2, y2, r["label"], r["source"],
                               r["video"], r["frame_ts"])
        comp_b64 = arr_to_b64(comp, quality=68)
        if comp_b64 is None:
            continue

        # 成员缩略图条(抽查簇内一致性)
        thumbs_html = ""
        extra = c["members"][1:1 + MAX_THUMBS]
        for m in extra:
            timg = robust_imread(os.path.join(dataset_root, m["crop_path"]))
            if timg is None:
                continue
            tb = _thumb_b64(timg)
            if tb:
                thumbs_html += f'<img src="data:image/jpeg;base64,{tb}"/>'
        more = len(c["members"]) - 1 - len(extra)
        if more > 0:
            thumbs_html += f'<span class="more">+{more}</span>'

        members_paths = [m["crop_path"] for m in c["members"]]
        members_json = html.escape(json.dumps(members_paths, ensure_ascii=False), quote=True)
        cur = r["label"]
        color = CONF_COLOR.get(cur, "#475569")
        n = len(c["members"])
        tag = "边界" if _is_boundary(r["video"], float(r["frame_ts"]), boundary_map) else ""

        card = f"""
        <div class="card" data-label="{cur}" data-src="{r['source']}" data-members="{members_json}">
          <img class="main" src="data:image/jpeg;base64,{comp_b64}" loading="lazy"/>
          <div class="thumbs">{thumbs_html}</div>
          <div class="meta">{r['video']} t={r['frame_ts']}s
            <span class="cur" style="background:{color}">{cur}</span>
            <span class="cnt2">×{n}</span>
            <span class="src">{r['source']}</span>{('<span class="bd">'+tag+'</span>') if tag else ''}</div>
          <div class="fb">
            <select class="verdict">
              <option value="walk"{' selected' if cur=='walk' else ''}>walk 绿灯/过街</option>
              <option value="stand"{' selected' if cur=='stand' else ''}>stand 红灯/站立</option>
              <option value="off"{' selected' if cur=='off' else ''}>off 非信号</option>
              <option value="delete">删除 废图</option>
            </select>
            <span class="status"></span>
          </div>
        </div>"""
        cards_by_video[r["video"]].append(card)

    # ---- 组装 HTML ----
    cards_html = ""
    for v in sorted(cards_by_video):
        cards_html += (f'<h2 data-vid="{v}">{v} '
                       f'<span class="cnt">({len(cards_by_video[v])} 簇)</span></h2>'
                       f'<div class="grid">{"".join(cards_by_video[v])}</div>')

    total_clusters = sum(len(v) for v in cards_by_video.values())
    html_doc = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>light-state 重训数据集抽检 v4</title>
<style>
body{font-family:-apple-system,sans-serif;background:#f1f5f9;margin:0;padding:0 16px 60px;}
#tb{position:sticky;top:0;background:#0f172a;color:#fff;padding:8px 14px;display:flex;gap:8px;
   align-items:center;z-index:20;margin:0 -16px 12px;font-size:12.5px;flex-wrap:wrap;}
#tb b{font-size:13px;}
#tb button{background:#1e293b;color:#fff;border:1px solid #334155;border-radius:5px;
   padding:4px 9px;cursor:pointer;font-size:12px;}
#tb button:hover{background:#334155;}
#tb button.primary{background:#2563eb;border-color:#2563eb;}
#tb button.warn{background:#b45309;border-color:#b45309;}
#tb .sep{width:1px;height:18px;background:#334155;margin:0 2px;}
#tb .flt{background:#0b1220;}
#tb .flt.on{background:#16a34a;border-color:#16a34a;}
#cnt{margin-left:auto;font-weight:600;}
#export{background:#fff!important;color:#0f172a!important;border:none!important;}
h1{color:#0f172a;margin:6px 0 2px;font-size:17px;}
.intro{color:#475569;font-size:12px;margin:0 0 10px;line-height:1.55;}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:#334155;margin:4px 0 12px;
   background:#fff;padding:6px 10px;border:1px solid #e2e8f0;border-radius:6px;}
.legend b{font-weight:700;}
.legend .sw{display:inline-block;width:12px;height:12px;vertical-align:-2px;margin-right:3px;border-radius:2px;}
.sw-prior{background:#ff8c00;} .sw-crop{background:#dc2626;}
h2{color:#0f172a;font-size:15px;margin:16px 0 5px;} .cnt{color:#64748b;font-weight:400;font-size:12px;}
.grid{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;}
.card{border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;width:602px;background:#fff;}
.card.done{box-shadow:0 0 0 2px #22c55e inset;}
.card.hide{display:none;}
.card img.main{display:block;width:100%;height:auto;background:#000;}
.thumbs{display:flex;gap:2px;padding:3px 4px;background:#0f172a;flex-wrap:wrap;min-height:0;}
.thumbs img{width:46px;height:46px;object-fit:cover;border-radius:2px;}
.thumbs .more{color:#94a3b8;font-size:11px;align-self:center;padding:0 4px;}
.meta{font-size:11px;padding:3px 6px;color:#334155;display:flex;gap:6px;align-items:center;flex-wrap:wrap;}
.cur{display:inline-block;color:#fff;font-size:9px;padding:1px 6px;border-radius:6px;font-weight:600;}
.cnt2{font-size:11px;color:#0f172a;font-weight:700;background:#e2e8f0;padding:1px 6px;border-radius:6px;}
.src{font-size:9px;color:#94a3b8;} .bd{color:#fff;font-size:8px;padding:1px 4px;border-radius:4px;background:#9333ea;}
.fb{padding:5px 6px;display:flex;gap:6px;align-items:center;font-size:11px;background:#f8fafc;border-top:1px solid #e2e8f0;}
.fb select{font-size:12px;padding:3px 6px;border:1px solid #cbd5e1;border-radius:4px;}
.fb .status{font-size:11px;}
</style></head><body>
<div id="tb">
  <b>重训抽检 v4</b>
  <button class="primary" id="acc-signal">✓ 接受全部 walk/stand</button>
  <button class="warn" id="acc-off">✓ 接受全部 off</button>
  <span class="sep"></span>
  <span>筛选:</span>
  <button class="flt on" data-f="all">全部</button>
  <button class="flt" data-f="off">只看 off</button>
  <button class="flt" data-f="walk">walk</button>
  <button class="flt" data-f="stand">stand</button>
  <button class="flt" data-f="todo">未处理</button>
  <span id="cnt">已处理 0/__TOTAL__ 簇</span>
  <button id="export">导出JSON</button>
</div>
<h1>light-state 重训数据集抽检 v4（聚类去重）</h1>
<div class="legend">
  <span><span class="sw sw-prior"></span><b>蓝框</b>=信号灯区域（算法盯的位置）</span>
  <span><span class="sw sw-crop"></span><b>红框</b>=实际抠出的位置（右边放大就是红框里的东西）</span>
  <span>每卡下方小图=本簇其他雷同帧（×N 表示整簇张数）</span>
</div>
<p class="intro">
<b>你只需判右边放大框里是什么</b>，选标签即可（改下拉自动保存，标代表=整簇 N 张一起背书）：<br/>
• <b>walk</b>=绿灯/过街 &nbsp;• <b>stand</b>=红灯/站立 &nbsp;• <b>off</b>=非信号（背心/植物/车身/反光/误捡绿斑） &nbsp;• <b>删除</b>=废图<br/>
<b>红框贴着蓝框</b>→看右边颜色直接定；<b>红框远离蓝框</b>→引擎在别处捡到绿，重点判是真信号还是 off。<br/>
高效流程：① 点「接受全部 walk/stand」背书自洽正样本 → ② 点「只看 off」→ ③ 扫一遍把其实是信号的整簇改掉 → ④ 点「接受全部 off」兜底 → ⑤ 导出JSON → 跑 apply。</p>
""" + cards_html + """
<script>
const TOTAL = __TOTAL__;
const done = new Set();           // 已处理的 card 元素
const results = new Map();        // card -> {label, members:[...]}

function refresh(){
  document.getElementById('cnt').textContent = '已处理 ' + done.size + '/' + TOTAL + ' 簇';
}
function markDone(card, label){
  const members = JSON.parse(card.dataset.members);
  results.set(card, {label, members});
  done.add(card);
  card.classList.add('done');
  const st = card.querySelector('.status');
  st.textContent = '✓'; st.style.color = '#16a34a';
}
// 改下拉即自动保存
document.querySelectorAll('.card .verdict').forEach(sel=>{
  sel.addEventListener('change', ()=>{
    markDone(sel.closest('.card'), sel.value);
    refresh();
  });
});
// 批量: 接受全部 walk/stand (用各卡默认标签)
document.getElementById('acc-signal').addEventListener('click', ()=>{
  document.querySelectorAll('.card').forEach(card=>{
    const d = card.dataset.label;
    if(d==='walk'||d==='stand'){ markDone(card, d); }
  });
  refresh();
});
// 批量: 接受全部 off (默认标签是 off 且尚未被手动改的)
document.getElementById('acc-off').addEventListener('click', ()=>{
  document.querySelectorAll('.card').forEach(card=>{
    if(card.dataset.label==='off' && !done.has(card)){
      card.querySelector('.verdict').value='off';
      markDone(card, 'off');
    }
  });
  refresh();
});
// 筛选
function applyFilter(f){
  document.querySelectorAll('.card').forEach(card=>{
    let show=true;
    if(f==='off')   show = card.dataset.label==='off';
    else if(f==='walk')  show = card.dataset.label==='walk';
    else if(f==='stand') show = card.dataset.label==='stand';
    else if(f==='todo')  show = !done.has(card);
    card.classList.toggle('hide', !show);
  });
}
document.querySelectorAll('#tb .flt').forEach(b=>{
  b.addEventListener('click', ()=>{
    document.querySelectorAll('#tb .flt').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    applyFilter(b.dataset.f);
  });
});
// 导出: 展开簇成员 -> 每 crop_path 一条
document.getElementById('export').addEventListener('click', ()=>{
  const items=[];
  for(const [card, v] of results){
    for(const cp of v.members) items.push({crop_path: cp, label: v.label});
  }
  if(items.length===0){ alert('还没处理任何簇'); return; }
  const blob=new Blob([JSON.stringify(items,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='classifier_retrain_feedback.json';
  a.click();
});
refresh();
</script></body></html>"""
    html_doc = html_doc.replace("__TOTAL__", str(total_clusters))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_doc)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[OK] 画廊 -> {args.out} ({size_mb:.1f} MB)")
    print(f"  全量 {len(rows)} 张(跳过 {skipped}) -> {total_clusters} 簇")
    print(f"  固定卡尺寸 {PAD*3+FRAME_W+CROP_SIDE}x{TXT_H+PAD*2+max(FRAME_H,CROP_SIDE)}; "
          f"聚类 gap={args.gap}s dist={args.dist}px")
    print(f"  导出展开簇成员 -> 每 crop_path 一条, 下游 apply 按 crop_path 匹配")


if __name__ == "__main__":
    main()
