#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_l3_labeling.py — 为 L3(学习式 ped-vs-vehicle 判别) 制备标注工具包。

从现有候选(零新视频)渲染:
  - 每个候选的紧框裁图(裁图即判别头输入视角)
  - 带 GT 参考框(红) + 候选框(绿)的上下文帧(便于 Jacob 判断该框落在 ped 杆还是车信杆)
  - 每候选附 L1 几何分 / L2 时序分(供参考, 不替 Jacob 决策)
生成自包含 HTML 画廊: 单文件、离线可开、单选 ped/vehicle/other、localStorage 自动保存、导出 labels.json。

Jacob 标完 labels.json 后回传, wb 即训 L3 判别头并入 M2b。
用法:
  PYTHONPATH=src ./.venv/bin/python scripts/build_l3_labeling.py \
      [--cands data/output/candidates_temporal.json] [--out data/output/l3_labels]
"""
import json, sys, argparse, base64, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
from redlight.models.ped_light_selector import (
    derive_ped_priors, _l1_geom_score, compute_temporal_scores, iou, _center_dist,
)

CROP_PAD = 0.20          # 裁图外扩比例(给判别头一点上下文)
CONTEXT_MAXW = 480       # 上下文帧最大宽(控 HTML 体积)


def _b64_jpg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("ascii") if ok else ""


def _draw_box(img, box_norm, W, H, color, label):
    x1, y1, x2, y2 = [int(v) for v in (box_norm[0] * W, box_norm[1] * H,
                                       box_norm[2] * W, box_norm[3] * H)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (max(0, x1), max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _cluster_key(b):
    """同视频内按(量化中心 + 宽高比桶)聚类 -> 同一固定设施(灯杆)归一个簇, 只标一次。"""
    cx = round((b[0] + b[2]) / 2 / 0.02)
    cy = round((b[1] + b[3]) / 2 / 0.02)
    w = max(1e-6, b[2] - b[0]); h = max(1e-6, b[3] - b[1])
    asp = round((h / w) / 0.5)
    return (cx, cy, asp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=str(ROOT / "data" / "output" / "candidates_temporal.json"))
    ap.add_argument("--frames-dir", default=str(ROOT / "datasets" / "frames"))
    ap.add_argument("--out", default=str(ROOT / "data" / "output" / "l3_labels"))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "context").mkdir(parents=True, exist_ok=True)

    with open(args.cands, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    frames_dir = Path(args.frames_dir)

    # 全局先验(仅用于给 Jacob 显示 L1 参考, 不替他决策)
    global_prior = derive_ped_priors([r["gt_wh"] for r in records])
    # 每视频 L2(跨帧复现分)
    by_video = {}
    for r in records:
        by_video.setdefault(r["video"], []).append(r)
    video_l2 = {V: compute_temporal_scores(fr) for V, fr in by_video.items()}

    # 逐视频聚类: 同一设施(灯杆)归一簇; 与 GT ped 重合的自动标 ped(训练可用 GT 造标签);
    # 其余 YOLO/持久 HSV 交 Jacob 判(车灯/其他); HSV 瞬时自动标 other(负例)
    manual = []      # Jacob 需标的簇(代表) —— 仅不重合红框者
    auto_ped = []    # 与 GT ped 重合 -> 自动 ped(不占 Jacob 时间)
    auto_other = []  # 自动负例(HSV 瞬时亮斑)
    for V in sorted(by_video):
        fr = by_video[V]
        clusters = {}
        for r in fr:
            fi = r["fi"]
            l2_list = video_l2[V][fr.index(r)]
            gt = r["gt_box_norm"]
            for i, c in enumerate(r["candidates"]):
                b = c.get("box_norm")
                if not b:
                    continue
                key = _cluster_key(b)
                cl = clusters.setdefault(key, {"rep": (fi, i, c, b, gt),
                                               "members": [], "l2": l2_list.get(i, 0.0)})
                cl["members"].append((fi, i))
                cl["l2"] = max(cl["l2"], l2_list.get(i, 0.0))
        for key, cl in clusters.items():
            fi, i, c, b, gt = cl["rep"]
            source = c.get("source", "?")
            l2 = cl["l2"]
            # 与 GT ped 强重合 -> 自动标 ped(训练阶段用 GT 造标签合法; 生产推理仍无 GT)
            if iou(b, gt) > 0.3 or _center_dist(b, gt) < 0.04:
                auto_ped.append({"video": V, "rep_fi": fi, "rep_idx": i,
                                 "box_norm": [round(v, 4) for v in b], "source": source,
                                 "l2": round(l2, 3), "n_members": len(cl["members"]),
                                 "members": cl["members"], "label": "ped"})
            elif source == "yolo" or (source == "hsv" and l2 > 0):
                manual.append({"video": V, "rep_fi": fi, "rep_idx": i,
                               "box_norm": [round(v, 4) for v in b], "source": source,
                               "l1": round(_l1_geom_score(b, global_prior), 3),
                               "l2": round(l2, 3),
                               "n_members": len(cl["members"]),
                               "members": cl["members"], "label": ""})
            else:
                # HSV 瞬时亮斑 -> 自动 other(训练负例), 不占用 Jacob 时间
                auto_other.append({"video": V, "fi": fi, "cand_idx": i,
                                   "box_norm": [round(v, 4) for v in b], "source": source,
                                   "label": "other"})

    # 渲染 manual 簇代表的裁图 + 上下文帧, 生成画廊
    manifest = []
    cards = []
    cid = 0
    for cl in manual:
        V, fi, i = cl["video"], cl["rep_fi"], cl["rep_idx"]
        fpath = frames_dir / V / f"frame_{fi:06d}.jpg"
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        H, W = img.shape[:2]
        b = cl["box_norm"]
        # 紧框裁图(外扩)
        bw, bh = max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x1 = max(0, int((cx - bw * (1 + CROP_PAD) / 2) * W))
        y1 = max(0, int((cy - bh * (1 + CROP_PAD) / 2) * H))
        x2 = min(W, int((cx + bw * (1 + CROP_PAD) / 2) * W))
        y2 = min(H, int((cy + bh * (1 + CROP_PAD) / 2) * H))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_b64 = _b64_jpg(crop)
        # 上下文: 红=GT ped, 绿=本候选
        sc = CONTEXT_MAXW / W
        ctx2 = cv2.resize(img, (CONTEXT_MAXW, int(H * sc)))
        cW, cH = CONTEXT_MAXW, int(H * sc)
        gt = next(r["gt_box_norm"] for r in by_video[V] if r["fi"] == fi)
        _draw_box(ctx2, gt, cW, cH, (0, 0, 255), "GT ped")
        _draw_box(ctx2, b, cW, cH, (0, 200, 0), "cand")
        ctx2_b64 = _b64_jpg(ctx2)

        crop_path = out / "crops" / f"{V}_{fi}_{cid}.jpg"
        cv2.imwrite(str(crop_path), crop)
        rel_frame = os.path.relpath(fpath, out).replace(os.sep, "/")  # lightbox 完整原图
        gt_str = ",".join(f"{v:.4f}" for v in gt)
        box_str = ",".join(f"{v:.4f}" for v in b)

        entry = dict(cl)
        entry["id"] = cid
        entry["crop_path"] = str(crop_path.relative_to(ROOT))
        entry["frame_path"] = str(fpath.relative_to(ROOT))
        manifest.append(entry)
        cards.append(
            f'<div class="card" data-id="{cid}" data-frame="{rel_frame}" '
            f'data-gt="{gt_str}" data-box="{box_str}" tabindex="0">'
            f'<div class="meta"><b>{V}</b> fi={fi} · src={cl["source"]} · '
            f'L1={cl["l1"]} · L2={cl["l2"]} · 同设施×{cl["n_members"]}</div>'
            f'<div class="fulllink" title="点开看完整原始大图(大图内可直接标注/翻页)">'
            f'<img class="ctx" src="data:image/jpeg;base64,{ctx2_b64}" onclick="openLightbox({cid})"/></div>'
            f'<img class="crop" src="data:image/jpeg;base64,{crop_b64}"/>'
            f'<div class="opts">'
            f'<label><input type="radio" name="c{cid}" value="ped">行人灯</label>'
            f'<label><input type="radio" name="c{cid}" value="vehicle">车灯</label>'
            f'<label><input type="radio" name="c{cid}" value="other">其他</label>'
            f'</div></div>'
        )
        cid += 1

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"n_manual": len(manifest), "n_auto_ped": len(auto_ped),
                   "n_auto_other": len(auto_other),
                   "manual": manifest, "auto_ped": auto_ped, "auto_other": auto_other},
                  f, ensure_ascii=False, indent=2)

    html = _HTML_TEMPLATE.replace("__CARDS__", "\n".join(cards)).replace("__N__", str(len(cards)))
    (out / "label_gallery.html").write_text(html, encoding="utf-8")

    print(f"[out] {out}/")
    print(f"  Jacob 需标簇(不重合红框的设施): {len(manifest)}  ← 只标这些, 每簇=同视频一根灯杆")
    print(f"  自动 ped(与红框强重合, 训练用): {len(auto_ped)}")
    print(f"  自动 other(HSV 瞬时, 训练用): {len(auto_other)}")
    print(f"  label_gallery.html: 离线标注页(单选+导出 labels.json)")


_HTML_TEMPLATE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>L3 标注 · ped-vs-vehicle</title>
<style>
body{font-family:system-ui,sans-serif;background:#f4f5f7;margin:0;padding:16px 16px 90px;}
h1{font-size:18px;} .hint{color:#555;font-size:13px;margin:4px 0 14px;line-height:1.6;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);cursor:pointer;outline:none;}
.card.active{border:3px solid #2563eb;box-shadow:0 0 0 2px #93c5fd;}
.meta{font-size:12px;color:#333;margin-bottom:6px;}
.ctx{width:100%;display:block;border:1px solid #eee;}
.fulllink{display:block;position:relative;cursor:zoom-in;}
.fulllink::after{content:"\\1F50D 原图";position:absolute;right:4px;bottom:4px;background:rgba(0,0,0,.6);color:#fff;font-size:11px;padding:1px 5px;border-radius:4px;}
.crop{width:120px;height:120px;object-fit:cover;display:block;margin:6px auto;border:1px solid #ccc;}
.opts{display:flex;gap:10px;justify-content:center;font-size:13px;margin-top:4px;}
.opts label{cursor:pointer;}
.toolbar{position:fixed;left:0;right:0;bottom:0;background:#111827ee;color:#fff;display:flex;gap:8px;align-items:center;padding:8px 14px;font-size:14px;z-index:10;flex-wrap:wrap;}
.toolbar button{background:#2563eb;color:#fff;border:none;border-radius:6px;padding:7px 12px;cursor:pointer;font-size:13px;}
.toolbar button.batch{background:#374151;}
.toolbar .count{margin-left:auto;font-variant-numeric:tabular-nums;}
.toolbar .keys{color:#9ca3af;font-size:12px;}
/* lightbox */
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:100;align-items:center;justify-content:center;flex-direction:column;}
#lbImgWrap{position:relative;max-width:94vw;max-height:82vh;}
#lbImg{width:100%;height:100%;object-fit:contain;display:block;}
#lbCanvas{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain;pointer-events:none;}
#lbCrop{position:absolute;right:10px;bottom:10px;width:150px;height:150px;object-fit:cover;border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.6);background:#000;}
#lbClose{position:absolute;top:10px;right:14px;background:#374151;color:#fff;border:none;border-radius:6px;padding:8px 14px;font-size:15px;cursor:pointer;z-index:101;}
#lbHint{position:absolute;bottom:10px;left:14px;color:#ddd;font-size:13px;background:rgba(0,0,0,.5);padding:6px 10px;border-radius:6px;}
</style></head><body>
<h1>L3 标注 · 候选是车灯 / 其他?</h1>
<div class="hint"><b>下面只显示需要你判的候选（绿框不重合红框的）。绿框重合红框的已自动标"行人灯"，不用管。</b>
红框=GT 行人灯(位置参考)。绿框=被检测到的候选灯，落别的杆上、多半是车灯。<br>
你只判每张绿框里那盏灯：<b>车灯</b>=红/黄/绿圆灯或车尾灯；<b>其他</b>=纯反光/杂亮斑/无清晰灯形。<br>
点大图(🔍原图)在<b>当前页弹出灯箱(lightbox)</b>：大图内可直接按 <b>1</b>行人灯 <b>2</b>车灯 <b>3</b>其他 标注并跳下一张，<b>← →</b> 翻页，<b>Esc</b> 退出。</div>
<div class="grid">__CARDS__</div>
<div class="toolbar">
  <span class="keys">选中卡片后 1/2/3 标注并跳下一张</span>
  <button onclick="quick('ped')">1 行人灯</button>
  <button onclick="quick('vehicle')">2 车灯</button>
  <button onclick="quick('other')">3 其他</button>
  <button class="batch" onclick="batch('vehicle')">全部标车灯</button>
  <button class="batch" onclick="batch('other')">全部标其他</button>
  <button class="batch" onclick="clearAll()">清空</button>
  <button class="batch" onclick="exportLabels()">导出 labels.json</button>
  <span class="count" id="cnt">已标 0 / __N__</span>
</div>
<div id="lb">
  <button id="lbClose" onclick="closeLightbox()">✕ 退出 (Esc)</button>
  <div id="lbImgWrap">
    <img id="lbImg" src=""/>
    <canvas id="lbCanvas"></canvas>
    <img id="lbCrop" src=""/>
  </div>
  <div id="lbHint">1 行人灯 · 2 车灯 · 3 其他 · ← → 翻页 · Esc 退出</div>
</div>
<script>
const KEY='l3_labels_v1';
const TOTAL=__N__;
const LABEL_CN={ped:'行人灯',vehicle:'车灯',other:'其他'};
let lightboxOpen=false, lbId=null;
function key(id){return 'c'+id;}
function cards(){return [...document.querySelectorAll('.card')];}
function setActive(id){document.querySelectorAll('.card.active').forEach(c=>c.classList.remove('active'));const el=document.querySelector(`.card[data-id="${id}"]`);if(el){el.classList.add('active');el.scrollIntoView({block:'nearest'});}}
function getLabel(id){const r=document.querySelector(`input[name="${key(id)}"]:checked`);return r?r.value:null;}
function setLabel(id,v){const r=document.querySelector(`input[name="${key(id)}"][value="${v}"]`);if(r){r.checked=true;save();}}
function save(){const m={};document.querySelectorAll('input[type=radio]:checked').forEach(r=>{m[r.name]=r.value;});localStorage.setItem(KEY,JSON.stringify(m));updateCount();}
function restore(){try{const m=JSON.parse(localStorage.getItem(KEY)||'{}');for(const k in m){const el=document.querySelector(`input[name="${k}"][value="${m[k]}"]`);if(el)el.checked=true;}}catch(e){}updateCount();}
function updateCount(){let n=0;cards().forEach(c=>{if(getLabel(c.dataset.id))n++;});document.getElementById('cnt').textContent=`已标 ${n} / ${TOTAL}`;}
function firstUnlabeled(){for(const c of cards()){if(!getLabel(c.dataset.id))return c.dataset.id;}return cards()[0]?cards()[0].dataset.id:null;}
function nextUnlabeled(fromIdx){const cs=cards();for(let i=fromIdx+1;i<cs.length;i++){if(!getLabel(cs[i].dataset.id))return cs[i].dataset.id;}for(let i=0;i<=fromIdx;i++){if(!getLabel(cs[i].dataset.id))return cs[i].dataset.id;}return null;}
function quick(v){const cur=document.querySelector('.card.active');let id=cur?cur.dataset.id:firstUnlabeled();if(!id)return;const idx=cards().findIndex(c=>c.dataset.id===id);setLabel(id,v);const nxt=nextUnlabeled(idx);if(nxt)setActive(nxt);}
function batch(v){if(!confirm('将全部 '+TOTAL+' 张标为「'+LABEL_CN[v]+'」？'))return;cards().forEach(c=>setLabel(c.dataset.id,v));}
function clearAll(){if(!confirm('清空所有标注？'))return;document.querySelectorAll('input[type=radio]:checked').forEach(r=>r.checked=false);save();}
function exportLabels(){const m={};document.querySelectorAll('input[type=radio]:checked').forEach(r=>{m[r.name]=r.value;});const out=Object.keys(m).map(k=>({id:+k.slice(1),label:m[k]}));const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='labels.json';a.click();}
// ---- lightbox ----
function openLightbox(id){const card=document.querySelector(`.card[data-id="${id}"]`);if(!card)return;lbId=String(id);lightboxOpen=true;document.getElementById('lb').style.display='flex';document.getElementById('lbImg').src=card.dataset.frame;const crop=card.querySelector('.crop');if(crop)document.getElementById('lbCrop').src=crop.src;setActive(String(id));}
function closeLightbox(){lightboxOpen=false;document.getElementById('lb').style.display='none';}
document.getElementById('lbImg').addEventListener('load',()=>{const card=document.querySelector(`.card[data-id="${lbId}"]`);if(!card)return;const img=document.getElementById('lbImg');const nW=img.naturalWidth,nH=img.naturalHeight;document.getElementById('lbImgWrap').style.aspectRatio=nW+'/'+nH;const cv=document.getElementById('lbCanvas');cv.width=nW;cv.height=nH;const ctx=cv.getContext('2d');ctx.clearRect(0,0,nW,nH);const lw=Math.max(2,nW/500);const gt=card.dataset.gt.split(',').map(Number);ctx.lineWidth=lw;ctx.strokeStyle='#ff2222';ctx.strokeRect(gt[0]*nW,gt[1]*nH,(gt[2]-gt[0])*nW,(gt[3]-gt[1])*nH);ctx.fillStyle='#ff2222';ctx.font=Math.round(nW/45)+'px sans-serif';ctx.fillText('GT ped',gt[0]*nW,Math.max(16,gt[1]*nH-6));const bx=card.dataset.box.split(',').map(Number);ctx.strokeStyle='#22dd22';ctx.strokeRect(bx[0]*nW,bx[1]*nH,(bx[2]-bx[0])*nW,(bx[3]-bx[1])*nH);});
function lbLabel(v){if(!lbId)return;setLabel(lbId,v);const idx=cards().findIndex(c=>c.dataset.id===lbId);const nxt=nextUnlabeled(idx);if(nxt)openLightbox(nxt);}
function lbNav(d){const cs=cards();const cur=cs.findIndex(c=>c.dataset.id===lbId);if(cur<0)return;const i=Math.max(0,Math.min(cs.length-1,cur+d));openLightbox(cs[i].dataset.id);}
document.getElementById('lb').addEventListener('click',e=>{if(e.target===document.getElementById('lb'))closeLightbox();});
// ---- events ----
document.querySelectorAll('.card').forEach(c=>c.addEventListener('click',e=>{if(e.target.tagName==='INPUT')return;setActive(c.dataset.id);}));
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
  if(lightboxOpen){
    if(e.key==='1'){lbLabel('ped');e.preventDefault();}
    else if(e.key==='2'){lbLabel('vehicle');e.preventDefault();}
    else if(e.key==='3'){lbLabel('other');e.preventDefault();}
    else if(e.key==='ArrowRight'||e.key==='j'){lbNav(1);e.preventDefault();}
    else if(e.key==='ArrowLeft'||e.key==='k'){lbNav(-1);e.preventDefault();}
    else if(e.key==='Escape'){closeLightbox();e.preventDefault();}
    return;
  }
  if(e.key==='1'){quick('ped');e.preventDefault();}
  else if(e.key==='2'){quick('vehicle');e.preventDefault();}
  else if(e.key==='3'){quick('other');e.preventDefault();}
  else if(e.key==='ArrowRight'||e.key==='j'){navRel(1);e.preventDefault();}
  else if(e.key==='ArrowLeft'||e.key==='k'){navRel(-1);e.preventDefault();}});
function navRel(d){const cs=cards();const cur=document.querySelector('.card.active');let i=cur?cs.findIndex(c=>c===cur):-1;i=Math.max(0,Math.min(cs.length-1,i+d));if(cs[i])setActive(cs[i].dataset.id);}
restore();
if(!document.querySelector('.card.active')){const f=firstUnlabeled();if(f)setActive(f);}
</script></body></html>"""


if __name__ == "__main__":
    main()
