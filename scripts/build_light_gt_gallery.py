#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_light_gt_gallery.py — 工具A: 抽帧 + 预填, 供 canonical 灯态 GT 标注画廊。

方案: docs/plans/2026-07-28-cc-plan-canonical-light-gt-annotation-tool.md
- 每 N 秒(默认2s)从 input_video/*.mp4 抽一帧, 存 data/output/light_gt_gallery/frames/。
- 每帧预填两类框(供 Jacob 确认/改, 真值以他确认为准):
  · 系统检测(source=pred): YOLO(cls9)∪HSV 候选, 带 color 猜测(HSV 主色) + type 猜测(L3 argmax)。
  · 旧标注(source=prior): light_location_gt.json 的 true_box_norm 贴到最近采样帧, type=pedestrian/governing=true。
- 产出 frames_prefill.json (FRAMES 数组) 供 annotate.html 加载。

**只读**: 只读视频抽帧 + 只读现有模型出预填建议; 不接线/不动生产权重/prior。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/build_light_gt_gallery.py                 # 全11视频
  PYTHONPATH=src ./.venv/bin/python scripts/build_light_gt_gallery.py --videos 违章05 --limit 6   # smoke
输出: data/output/light_gt_gallery/{frames/*.jpg, frames_prefill.json}
"""
import json, sys, argparse, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
import types
import torch
from PIL import Image

from redlight.models.traffic_light import TrafficLightDetector
from redlight.models.signal_candidates import build_candidates
from redlight.models.l3_ped_vehicle import L3PedVehicleNet, crop_candidate, score_crop, CLASSES

VIDEOS_DIR = ROOT / "input_video"
GTLOC = ROOT / "datasets" / "light_location_gt.json"
L3W = ROOT / "models" / "l3_ped_full.pt"
OUT_DIR = ROOT / "data" / "output" / "light_gt_gallery"
FRAMES_DIR = OUT_DIR / "frames"
DISPLAY_W = 1100  # 画廊显示宽度(box_norm 归一化, 缩放无碍)
# L3 类 → GT schema type
_L3_TO_TYPE = {"ped": "pedestrian", "vehicle": "vehicle", "other": "distractor"}


def _cfg():
    tl = types.SimpleNamespace(method="color", smoothing_window=8, sat_min=130,
                               value_floor=60, min_area_px=30, max_area_ratio=0.008,
                               max_aspect_ratio=3.5, color_s_min=22)
    return types.SimpleNamespace(traffic_light=tl)


def color_guess(roi):
    """HSV 主色猜测: green / red / off(无足够有色像素)。"""
    if roi is None or roi.size == 0:
        return "off"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_g = cv2.inRange(hsv, np.array([35, 130, 60]), np.array([85, 255, 255]))
    mask_r = cv2.inRange(hsv, np.array([0, 130, 60]), np.array([10, 255, 255])) | \
             cv2.inRange(hsv, np.array([170, 130, 60]), np.array([180, 255, 255]))
    g = int(mask_g.sum()) // 255
    r = int(mask_r.sum()) // 255
    if g == 0 and r == 0:
        return "off"
    return "green" if g >= r else "red"


def load_prior_boxes():
    """按 video 收集 light_location_gt 旧框: {video: [(fi, box_norm, color), ...]}。"""
    d = json.load(open(GTLOC, encoding="utf-8"))
    by_video = {}
    for a in d.get("annotations", []):
        if a.get("no_light") or not a.get("true_box_norm"):
            continue
        by_video.setdefault(a["video"], []).append(
            (a["fi"], a["true_box_norm"], a.get("color", "unclear")))
    return by_video


def crop_bgr(frame, box_norm):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box_norm
    px = (max(0, int(x1 * W)), max(0, int(y1 * H)), min(W, int(x2 * W)), min(H, int(y2 * H)))
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return frame[px[1]:px[3], px[0]:px[2]]


_HTML_HEAD = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>canonical 灯态 GT 标注(预填确认式)</title><style>
 body{font-family:-apple-system,Segoe UI,sans-serif;margin:12px;background:#f4f4f4;color:#222}
 h1{font-size:18px} .hint{color:#333;background:#fff;border-left:4px solid #06c;padding:8px 12px;margin:6px 0;font-size:13px;line-height:1.6}
 .frame{background:#fff;border:1px solid #ccc;border-radius:8px;margin:12px 0;padding:8px;display:flex;gap:10px}
 .frame.done{opacity:.5;border-color:#0a0}
 .left{flex:0 0 auto} .right{flex:1 1 auto;font-size:13px;max-height:640px;overflow:auto}
 .ttl{font-weight:600;margin-bottom:4px} .wrap{position:relative;display:inline-block;line-height:0}
 .wrap canvas{position:absolute;left:0;top:0;cursor:crosshair}
 table{border-collapse:collapse;width:100%;font-size:12px} td,th{border:1px solid #ddd;padding:2px 4px}
 tr.sel{background:#fde} tr.gov{background:#dfd}
 button{font-size:12px;padding:2px 8px;margin:2px} .big{background:#0a0;color:#fff;font-size:14px;padding:6px 14px}
 select{font-size:12px} .nav{position:sticky;top:0;background:#fff;padding:6px;border-bottom:1px solid #ccc;font-size:12px;z-index:9}
 #out{width:100%;height:120px;font-family:monospace;font-size:11px}
 .bpred{color:#e67e00} .bprior{color:#06f} .bgov{color:#0a0;font-weight:700}
</style></head><body>
<h1>canonical 灯态 GT 标注 — 预填确认式</h1>
<div class=hint>
<b>橙框</b>=系统检测(YOLO∪HSV) <b>蓝框</b>=你之前的标注 <b>绿框</b>=governing(管这条斑马线的行人灯)。<br>
每帧:勾出 <b>governing</b>(管斑马线的行人灯,<b>可多盏</b>,勾 gov☑)并核对其颜色;错框点 <b>删</b>;漏了就在图上 <b>拖</b>一个新框;整帧无灯勾 <b>无灯</b>。核对完点 <b>[✓确认本帧]</b>。<br>只有 governing 那些框影响评测,其余候选框不用管。<br>
进度自动存本地(localStorage),可关页续标。全标完点底部 <b>[生成GT JSON]→[复制]</b> 贴回给 cc。
</div>
<div class=nav id=nav></div><div id=frames></div>
<hr><button class=big onclick=exportJSON()>生成GT JSON</button>
<button onclick="document.getElementById('out').select();document.execCommand('copy')">复制</button>
<button onclick="if(confirm('清空所有本地标注,回到预填?')){localStorage.removeItem(LSKEY);location.reload()}">重置</button>
<span id=prog></span><textarea id=out placeholder="点[生成GT JSON]后在此,整段复制"></textarea>
<script>
const COLORS=["red","green","off","countdown","unclear"];
const TYPES=["pedestrian","vehicle","distractor"];
"""

_HTML_JS = """
const LSKEY="light_gt_annot_v1";
let state=JSON.parse(localStorage.getItem(LSKEY)||"null");
if(!state){ state={}; FRAMES.forEach(f=>{ state[f.id]={boxes:f.boxes.map(b=>Object.assign({},b)),no_light:false,confirmed:false}; }); }
else { FRAMES.forEach(f=>{ if(!state[f.id]) state[f.id]={boxes:f.boxes.map(b=>Object.assign({},b)),no_light:false,confirmed:false}; }); }
let sel={};  // frameId -> selected box index
function save(){ localStorage.setItem(LSKEY,JSON.stringify(state)); }

function mk(f){
  const d=document.createElement('div'); d.className='frame'+(state[f.id].confirmed?' done':''); d.id='fr_'+f.id;
  d.innerHTML=`<div class=left><div class=ttl>${f.video} fi=${f.fi} t=${f.t}s</div>
     <div class=wrap><img src="${f.img}" width=${f.dw} height=${f.dh}>
      <canvas id="cv_${f.id}" width=${f.dw} height=${f.dh}></canvas></div>
     <div><label><input type=checkbox id="nl_${f.id}" ${state[f.id].no_light?'checked':''}> 整帧无灯</label>
      <button onclick="confirmFrame('${f.id}')">✓确认本帧</button></div></div>
   <div class=right id="rt_${f.id}"></div>`;
  document.getElementById('frames').appendChild(d);
  const cv=d.querySelector('canvas');
  let drag=false,sx,sy,cur=null;
  cv.addEventListener('mousedown',e=>{const r=cv.getBoundingClientRect();sx=(e.clientX-r.left)*cv.width/r.width;sy=(e.clientY-r.top)*cv.height/r.height;drag=true;cur=null;});
  cv.addEventListener('mousemove',e=>{if(!drag)return;const r=cv.getBoundingClientRect();const x=(e.clientX-r.left)*cv.width/r.width,y=(e.clientY-r.top)*cv.height/r.height;cur=[Math.min(sx,x),Math.min(sy,y),Math.max(sx,x),Math.max(sy,y)];redraw(f);drawTmp(f,cur);});
  cv.addEventListener('mouseup',()=>{drag=false; if(cur&&(cur[2]-cur[0])>3&&(cur[3]-cur[1])>3){
     state[f.id].boxes.push({box_norm:[+(cur[0]/cv.width).toFixed(4),+(cur[1]/cv.height).toFixed(4),+(cur[2]/cv.width).toFixed(4),+(cur[3]/cv.height).toFixed(4)],color:"green",type:"pedestrian",governing:false,source:"manual"});
     sel[f.id]=state[f.id].boxes.length-1; save(); renderRows(f);} cur=null;redraw(f);});
  renderRows(f); redraw(f);
}
function colorOf(b){return b.source==='prior'?'#06f':(b.source==='manual'?'#c0f':'#e67e00');}
function redraw(f){const cv=document.getElementById('cv_'+f.id);const c=cv.getContext('2d');c.clearRect(0,0,cv.width,cv.height);
  state[f.id].boxes.forEach((b,i)=>{const p=b.box_norm;const x=p[0]*cv.width,y=p[1]*cv.height,w=(p[2]-p[0])*cv.width,h=(p[3]-p[1])*cv.height;
    c.lineWidth=(i===sel[f.id])?4:2; c.strokeStyle=b.governing?'#0a0':colorOf(b); if(i===sel[f.id])c.strokeStyle='#e0e';
    c.strokeRect(x,y,w,h);
    c.fillStyle=c.strokeStyle;c.font='11px sans-serif';c.fillText((b.governing?'★':'')+b.type[0]+'/'+b.color,x,Math.max(9,y-2));});}
function drawTmp(f,bx){const cv=document.getElementById('cv_'+f.id);const c=cv.getContext('2d');c.strokeStyle='#e0e';c.lineWidth=2;c.strokeRect(bx[0],bx[1],bx[2]-bx[0],bx[3]-bx[1]);}
function renderRows(f){const box=state[f.id].boxes;
  let h='<table><tr><th>#</th><th>类型</th><th>颜色</th><th>gov</th><th>源</th><th></th></tr>';
  box.forEach((b,i)=>{ h+=`<tr class="${i===sel[f.id]?'sel':''} ${b.governing?'gov':''}" onclick="pick('${f.id}',${i})">
    <td>${i}</td>
    <td><select onchange="setF('${f.id}',${i},'type',this.value)">${TYPES.map(t=>`<option ${b.type===t?'selected':''}>${t}</option>`).join('')}</select></td>
    <td><select onchange="setF('${f.id}',${i},'color',this.value)">${COLORS.map(cc=>`<option ${b.color===cc?'selected':''}>${cc}</option>`).join('')}</select></td>
    <td><input type=checkbox ${b.governing?'checked':''} onclick="event.stopPropagation();setGov('${f.id}',${i})"></td>
    <td class="b${b.source}">${b.source}</td>
    <td><button onclick="event.stopPropagation();delBox('${f.id}',${i})">删</button></td></tr>`; });
  h+='</table><button onclick="clearGov(\\''+f.id+'\\')">清 governing</button>';
  document.getElementById('rt_'+f.id).innerHTML=h;}
function pick(id,i){sel[id]=i;const f=FRAMES.find(x=>x.id===id);redraw(f);renderRows(f);}
function setF(id,i,k,v){state[id].boxes[i][k]=v;save();const f=FRAMES.find(x=>x.id===id);redraw(f);}
function setGov(id,i){state[id].boxes[i].governing=!state[id].boxes[i].governing;save();const f=FRAMES.find(x=>x.id===id);redraw(f);renderRows(f);}  // 可多盏; checkbox 可切换(修取消不还原)
function clearGov(id){state[id].boxes.forEach(b=>b.governing=false);save();const f=FRAMES.find(x=>x.id===id);redraw(f);renderRows(f);}
function delBox(id,i){state[id].boxes.splice(i,1);if(sel[id]>=i)sel[id]=Math.max(0,sel[id]-1);save();const f=FRAMES.find(x=>x.id===id);redraw(f);renderRows(f);}
function confirmFrame(id){state[id].confirmed=true;state[id].no_light=document.getElementById('nl_'+id).checked;save();document.getElementById('fr_'+id).classList.add('done');prog();}
function prog(){const done=FRAMES.filter(f=>state[f.id].confirmed).length;
  document.getElementById('prog').textContent=` 已确认 ${done}/${FRAMES.length} 帧`;
  const vids=[...new Set(FRAMES.map(f=>f.video))];
  document.getElementById('nav').innerHTML='进度: '+vids.map(v=>{const fs=FRAMES.filter(f=>f.video===v);const dn=fs.filter(f=>state[f.id].confirmed).length;return `<span style="color:${dn===fs.length?'#0a0':'#b00'}">${v.slice(2)} ${dn}/${fs.length}</span>`;}).join(' | ');}
function exportJSON(){const out=FRAMES.map(f=>{const s=state[f.id];return {video:f.video,source_fi:f.fi,t:f.t,no_light:s.no_light,confirmed:s.confirmed,
    boxes:s.no_light?[]:s.boxes.map(b=>({box_norm:b.box_norm,color:b.color,type:b.type,governing:!!b.governing}))};});
  document.getElementById('out').value=JSON.stringify({schema:"light_canonical_gt_v1",frames:out},null,1);}
FRAMES.forEach(mk); prog();
</script></body></html>
"""


def write_html(FRAMES):
    html = _HTML_HEAD + "const FRAMES=" + json.dumps(FRAMES, ensure_ascii=False) + ";\n" + _HTML_JS
    (OUT_DIR / "annotate.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0, help="每 N 秒抽一帧")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--videos", nargs="*", default=None, help="只处理指定视频(smoke)")
    ap.add_argument("--limit", type=int, default=0, help="每视频最多抽 N 帧(0=全部)")
    ap.add_argument("--topk", type=int, default=12, help="每帧预填最多 N 个候选框(裁剪 HSV 噪声)")
    args = ap.parse_args()

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    det = TrafficLightDetector(_cfg(), verbose=False)
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    l3 = L3PedVehicleNet()
    l3.load_state_dict(torch.load(L3W, map_location="cpu", weights_only=True))
    l3.eval()
    priors = load_prior_boxes()

    vids = sorted(VIDEOS_DIR.glob("*.mp4"))
    if args.videos:
        vids = [v for v in vids if v.stem in args.videos]

    FRAMES = []
    for vp in vids:
        video = vp.stem
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.70
        step = max(1, int(round(fps * args.seconds)))
        vprior = priors.get(video, [])
        src_fi, sampled = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if src_fi % step == 0:
                H, W = frame.shape[:2]
                t = round(src_fi / fps, 2)
                # 显示缩放
                dh = int(DISPLAY_W * H / W)
                disp = cv2.resize(frame, (DISPLAY_W, dh))
                img_rel = f"frames/{video}_{src_fi}.jpg"
                cv2.imwrite(str(OUT_DIR / img_rel), disp, [cv2.IMWRITE_JPEG_QUALITY, 85])
                # 系统检测候选
                res = model(frame, conf=args.conf, classes=[9], imgsz=args.imgsz, verbose=False)[0]
                yolo_px = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
                hsv_px = [s["box"] for s in det._candidates(frame)]
                cands = build_candidates(yolo_px, hsv_px, W, H)
                # 裁剪: 原始候选是误绿"消防栓"(HSV对每个绿点发火, 每帧上百框, 没法标)。
                # 只留可信的少量: YOLO(交通灯形状检测)强优先, 其余按面积, 取 top-N。
                def _rank(c):
                    bx = c["box"]
                    area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                    return (1 if c.get("source") == "yolo" else 0, area)
                cands = sorted(cands, key=_rank, reverse=True)[:args.topk]
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pred_boxes = []
                for c in cands:
                    bx = c["box"]
                    bn = [bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H]
                    roi = crop_bgr(frame, bn)
                    ten = crop_candidate(frame_pil, tuple(bn))
                    sc = score_crop(l3, ten)
                    ttype = _L3_TO_TYPE[max(sc, key=sc.get)]
                    pred_boxes.append({
                        "box_norm": [round(v, 4) for v in bn],
                        "color": color_guess(roi), "type": ttype,
                        "governing": False,  # 预填不猜 governing(每帧至多一个, 由 Jacob 指定); 只旧标注框预设 true
                        "source": "pred",
                    })
                # 旧标注框(贴最近采样帧: |fi - src_fi| < step/2)
                prior_boxes = []
                for (pfi, pbox, pcol) in vprior:
                    if abs(pfi - src_fi) < step / 2:
                        prior_boxes.append({
                            "box_norm": [round(v, 4) for v in pbox],
                            "color": pcol, "type": "pedestrian",
                            "governing": True, "source": "prior",
                        })
                FRAMES.append({
                    "id": f"{video}_{src_fi}", "video": video, "fi": src_fi, "t": t,
                    "img": img_rel, "dw": DISPLAY_W, "dh": dh,
                    "boxes": prior_boxes + pred_boxes,  # 旧标注在前(优先确认)
                })
                sampled += 1
                if args.limit and sampled >= args.limit:
                    break
            src_fi += 1
        cap.release()
        print(f"  {video}: fps={fps:.2f} step={step} 采样帧={sampled} 预填框(累计FRAMES={len(FRAMES)})")

    out_json = OUT_DIR / "frames_prefill.json"
    out_json.write_text(json.dumps(FRAMES, ensure_ascii=False, indent=1), encoding="utf-8")
    write_html(FRAMES)
    npred = sum(len([b for b in f["boxes"] if b["source"] == "pred"]) for f in FRAMES)
    nprior = sum(len([b for b in f["boxes"] if b["source"] == "prior"]) for f in FRAMES)
    print(f"[out] {out_json}  帧={len(FRAMES)} 预填框: pred={npred} prior={nprior}")
    print(f"[out] {OUT_DIR / 'annotate.html'}  ← 浏览器打开标注")


if __name__ == "__main__":
    main()
