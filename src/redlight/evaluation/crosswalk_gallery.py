"""斑马线多边形标注画廊 (BaseGalleryBuilder 实现, Part B1)。

用途: 人工标注每个关键帧真斑马线的**四点多边形**(斜视角下斑马线是斜带,
非水平全宽带)作为 GT, 供 eval_crosswalk_mask.py 的 2D mask-IoU 评测。

复用画廊基类, 交互为**画布点 4 个角点成多边形**。
标注策略(重要): 斑马线是静止实体, 被车/人**遮挡的部分也要脑补补全**整块形状,
不要只画露出来的碎块(否则违章车脚下那块反而没被标为斑马线 -> 占道判据自相矛盾)。

video_items: {video: [{"t_sec","frame_idx","frame"(np),"pred_band"|None,"note"}]}
gt_map: {video: [{"ts","poly","note"}, ...]}  (骨架, poly 可能为 null)
"""
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .gallery_builder import BaseGalleryBuilder


class CrosswalkMaskGalleryBuilder(BaseGalleryBuilder):
    """斑马线四点多边形标注画廊。"""

    @property
    def title(self) -> str:
        return "斑马线多边形标注画廊 (B1: 标真斑马线多边形)"

    @property
    def toolbar_label(self) -> str:
        return "斑马线多边形标注"

    def intro_html(self) -> str:
        return (
            "每张=一个关键帧。<b>青色带</b>=检测器 v11 当前预测(全宽水平带, 仅参考)。"
            "<br/><b>标注方式(任意多边形):</b> 点小图放大 → 在大图上<b>点击加顶点</b>(斜/透视用 ≥3 点贴合, 想加几个加几个)→ "
            "<b>拖动顶点</b>可挪位置, <b>右键顶点</b>删点, <b>撤销</b>退上一点 → 点<b>✓ 确认保存</b>(一步落盘+关灯箱)。"
            "<br/>重开会<b>回显</b>上次标注(可继续调整)。"
            "<br/><b>⚠️ 遮挡策略:</b> 斑马线被车/人挡住的部分<b>也要脑补补全</b>整块形状(斑马线是静止实体, "
            "违章车正压在它上面)——<b>不要只画露出来的碎块</b>。遮挡严重时估计顶点, 判定选“有遮挡-已估计”。"
            "<br/>判定: <b>已标多边形</b> / <b>此帧无斑马线</b>(留空) / <b>有遮挡-已估计</b> / <b>看不清</b>。ESC 关灯箱不保存。"
        )

    def verdict_options(self) -> List[Tuple[str, str]]:
        return [
            ("labeled", "已标多边形"),
            ("occluded_est", "有遮挡-已估计"),
            ("no_crosswalk", "此帧无斑马线"),
            ("uncertain", "看不清"),
        ]

    def reason_options(self) -> List[Tuple[str, str]]:
        return []

    def is_mismatch(self, item: dict, gt: Any) -> bool:
        return True  # 标注模式: 全部关键帧都展示

    def _sample_representatives(self, mismatches: List[dict]) -> List[dict]:
        return list(mismatches)  # 关键帧本就少, 不下采样

    def annotate_crop(self, frame, item, gt):
        return _draw_ref(frame, item.get("pred_band"), item.get("note", ""),
                         float(item.get("t_sec", 0)), thumb_w=340)

    def annotate_full(self, frame, item, gt):
        return _draw_ref(frame, item.get("pred_band"), item.get("note", ""),
                         float(item.get("t_sec", 0)), thumb_w=None)

    def item_meta_html(self, item: dict, gt: Any) -> str:
        pb = item.get("pred_band")
        pb_str = f"{pb[0]}-{pb[1]}" if pb else "无带"
        return (f'预测带 <b>{pb_str}</b> · t={float(item.get("t_sec", 0)):.1f}s'
                f'<br/><span style="color:#64748b;font-size:10px;">{item.get("note", "")}</span>')

    def feedback_key(self, video: str, item: dict) -> Tuple[str, str]:
        return (video, f"{float(item.get('t_sec', 0)):.1f}")

    def extra_data_attrs(self, video: str, item: dict, gt: Any) -> Dict[str, str]:
        H, W = 0, 0
        fr = item.get("frame")
        if fr is not None:
            H, W = fr.shape[:2]
        return {"imgw": str(W), "imgh": str(H)}

    def _extra_feedback_inputs_html(self, video, item, gt, fb) -> str:
        poly = fb.get("poly", "")
        return (
            '<div style="display:flex;gap:4px;align-items:center;">'
            '<span style="font-size:10px;color:#475569;">四点</span>'
            f'<input class="poly" value="{poly}" style="width:180px;font-size:10px;" '
            'placeholder="x,y;x,y;x,y;x,y"/>'
            '<button type="button" class="clear-poly" style="font-size:10px;padding:2px 6px;">重画</button>'
            '</div>'
        )

    def _extra_js(self) -> str:
        return _CROSSWALK_POLY_JS


def _draw_ref(frame, pred_band, note, ts, thumb_w=None):
    out = frame.copy()
    H, W = out.shape[:2]
    if pred_band:
        y0, y1 = int(pred_band[0]), int(pred_band[1])
        ov = out.copy()
        cv2.rectangle(ov, (0, y0), (W - 1, y1), (255, 255, 0), -1)
        out = cv2.addWeighted(ov, 0.20, out, 0.80, 0)
    for y in range(0, H, 100):
        cv2.line(out, (0, y), (24, y), (0, 255, 0), 1)
        cv2.putText(out, str(y), (2, min(H - 2, y + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    for x in range(0, W, 100):
        cv2.line(out, (x, 0), (x, 24), (0, 255, 0), 1)
        cv2.putText(out, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.rectangle(out, (0, 0), (W - 1, 26), (0, 0, 0), -1)
    pb = f"预测带 {pred_band[0]}-{pred_band[1]}" if pred_band else "预测:无带"
    cv2.putText(out, f"t={ts:.1f}s {pb} 青=v11预测(参考) 遮挡也补全",
                (28, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if thumb_w and W > thumb_w:
        out = cv2.resize(out, (thumb_w, int(H * thumb_w / W)))
    return out


# ---------- 四点多边形点击 JS ----------

_CROSSWALK_POLY_JS = r"""
// ===== 斑马线四点多边形标注(带确认条 + 重开回显) =====
(function(){
  const lb = document.getElementById('lb');
  if(!lb) return;
  const lbImg = lb.querySelector('img');
  let srcCard = null;
  let pts = [];                       // 已点的自然像素坐标 [[nx,ny],...]
  const cv = document.createElement('canvas');
  cv.id='anno-canvas';
  cv.style.cssText='position:fixed;z-index:210;cursor:crosshair;display:none;';
  document.body.appendChild(cv);
  const ctx = cv.getContext('2d');

  // 悬浮控制条(灯箱内)
  const bar = document.createElement('div');
  bar.id='anno-bar';
  bar.style.cssText='position:fixed;z-index:230;top:14px;left:50%;transform:translateX(-50%);'
    +'background:#0f172a;color:#fff;padding:8px 14px;border-radius:8px;display:none;gap:10px;'
    +'align-items:center;font-size:13px;box-shadow:0 4px 18px rgba(0,0,0,.55);';
  bar.innerHTML='<span id="anno-msg">点击加顶点</span>'
    +'<button id="anno-undo" style="background:#475569;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;">撤销</button>'
    +'<button id="anno-redraw" style="background:#475569;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;">重画</button>'
    +'<button id="anno-confirm" style="background:#22c55e;color:#fff;border:none;border-radius:4px;padding:4px 12px;cursor:pointer;font-weight:700;">✓ 确认保存</button>'
    +'<span style="opacity:.6;">点击加点·拖顶点挪·右键删·ESC不存</span>';
  document.body.appendChild(bar);
  const msg = ()=>document.getElementById('anno-msg');
  let dragIdx = -1, downPos = null;              // 拖动的顶点下标 / 空白按下位置

  function parsePoly(v){
    if(!v) return [];
    return v.trim().split(';').map(s=>s.split(',').map(Number)).filter(p=>p.length===2 && p.every(n=>!isNaN(n)));
  }
  function loadExisting(){
    pts = [];
    if(srcCard){ const v=srcCard.querySelector('.poly'); if(v) pts = parsePoly(v.value); }
  }
  document.querySelectorAll('img.zoom').forEach(im=>{
    im.addEventListener('click', ()=>{ srcCard = im.closest('.crop-card'); loadExisting(); });
  });

  function natToDisp(nx,ny){ const nw=lbImg.naturalWidth||1, nh=lbImg.naturalHeight||1; return [nx*cv.width/nw, ny*cv.height/nh]; }
  function dispToNat(dx,dy){ const nw=lbImg.naturalWidth||1, nh=lbImg.naturalHeight||1;
    return [Math.round(Math.max(0,Math.min(nw, dx*nw/(cv.width||1)))), Math.round(Math.max(0,Math.min(nh, dy*nh/(cv.height||1))))]; }
  function hitTest(dx,dy){                        // 命中的顶点下标(显示坐标, <12px), 无=-1
    for(let i=0;i<pts.length;i++){ const [px,py]=natToDisp(pts[i][0],pts[i][1]); if(Math.hypot(dx-px,dy-py)<12) return i; }
    return -1;
  }
  function updBar(){ const m=msg(); if(!m) return;
    m.textContent = pts.length>=3 ? ('已 '+pts.length+' 点 ✓ 可确认(可继续加/拖/删)') : ('已 '+pts.length+' 点 (≥3 点才可确认)');
  }
  function place(){
    const r=lbImg.getBoundingClientRect();
    cv.style.left=r.left+'px'; cv.style.top=r.top+'px';
    cv.width=r.width; cv.height=r.height; cv.style.width=r.width+'px'; cv.style.height=r.height+'px';
    const show = lb.classList.contains('show');
    cv.style.display = show ? 'block' : 'none';
    bar.style.display = show ? 'flex' : 'none';
    redraw(); updBar();
  }
  function redraw(){
    ctx.clearRect(0,0,cv.width,cv.height);
    if(!pts.length) return;
    ctx.fillStyle='rgba(239,68,68,0.22)'; ctx.strokeStyle='#ef4444'; ctx.lineWidth=2;
    ctx.beginPath();
    pts.forEach((p,i)=>{ const [dx,dy]=natToDisp(p[0],p[1]); if(i===0)ctx.moveTo(dx,dy); else ctx.lineTo(dx,dy); });
    if(pts.length>=3) ctx.closePath();
    if(pts.length>=3) ctx.fill();
    ctx.stroke();
    pts.forEach((p,i)=>{ const [dx,dy]=natToDisp(p[0],p[1]);
      ctx.fillStyle = i===dragIdx ? '#fbbf24' : '#fff'; ctx.beginPath(); ctx.arc(dx,dy,6,0,7); ctx.fill();
      ctx.strokeStyle='#ef4444'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(dx,dy,6,0,7); ctx.stroke();
      ctx.fillStyle='#ef4444'; ctx.font='13px sans-serif'; ctx.fillText(String(i+1),dx+8,dy-8);
    });
  }
  function writeInput(){
    if(!srcCard) return;
    const pi=srcCard.querySelector('.poly');
    if(pi) pi.value = pts.length>=3 ? pts.map(p=>p[0]+','+p[1]).join(';') : '';
    const vsel=srcCard.querySelector('.verdict'); if(vsel && !vsel.value && pts.length>=3) vsel.value='labeled';
  }

  cv.addEventListener('mousedown', e=>{
    e.stopPropagation();
    const r=cv.getBoundingClientRect(), dx=e.clientX-r.left, dy=e.clientY-r.top;
    const hit=hitTest(dx,dy);
    if(hit>=0){ dragIdx=hit; downPos=null; redraw(); }   // 命中顶点 -> 拖动
    else { downPos=[dx,dy]; }                            // 空白 -> 可能加点(松手判定)
  });
  window.addEventListener('mousemove', e=>{
    if(dragIdx<0) return;
    const r=cv.getBoundingClientRect();
    pts[dragIdx]=dispToNat(e.clientX-r.left, e.clientY-r.top); redraw();
  });
  window.addEventListener('mouseup', e=>{
    if(dragIdx>=0){ dragIdx=-1; writeInput(); redraw(); return; }
    if(downPos && cv.style.display!=='none'){
      const r=cv.getBoundingClientRect(), dx=e.clientX-r.left, dy=e.clientY-r.top;
      if(Math.hypot(dx-downPos[0],dy-downPos[1])<6 && dx>=0 && dy>=0 && dx<=cv.width && dy<=cv.height){
        pts.push(dispToNat(dx,dy)); writeInput(); redraw(); updBar();
      }
    }
    downPos=null;
  });
  cv.addEventListener('contextmenu', e=>{
    e.preventDefault(); e.stopPropagation();
    const r=cv.getBoundingClientRect(); const hit=hitTest(e.clientX-r.left, e.clientY-r.top);
    if(hit>=0){ pts.splice(hit,1); writeInput(); redraw(); updBar(); }
  });
  cv.addEventListener('click', e=>e.stopPropagation());   // 吞掉 click 防关灯箱

  document.getElementById('anno-undo').addEventListener('click', e=>{ e.stopPropagation(); pts.pop(); writeInput(); redraw(); updBar(); });
  document.getElementById('anno-redraw').addEventListener('click', e=>{ e.stopPropagation(); pts=[]; writeInput(); redraw(); updBar(); });
  document.getElementById('anno-confirm').addEventListener('click', e=>{
    e.stopPropagation();
    if(pts.length<3){ const m=msg(); if(m) m.textContent='至少需要 3 个顶点'; return; }
    writeInput();
    const btn=srcCard && srcCard.querySelector('.save');
    if(btn) btn.click();
    lb.classList.remove('show');
  });

  const mo=new MutationObserver(()=>{ if(lb.classList.contains('show')){ setTimeout(place,30);} else { cv.style.display='none'; bar.style.display='none'; pts=[]; } });
  mo.observe(lb,{attributes:true,attributeFilter:['class']});
  lbImg.addEventListener('load', ()=>{ if(lb.classList.contains('show')) setTimeout(place,10); });
  window.addEventListener('resize', ()=>{ if(lb.classList.contains('show')) place(); });

  document.querySelectorAll('.clear-poly').forEach(btn=>{
    btn.addEventListener('click', e=>{ e.stopPropagation();
      const card=btn.closest('.crop-card'); const p=card.querySelector('.poly'); if(p)p.value='';
      if(srcCard===card){ pts=[]; redraw(); updBar(); }
    });
  });
})();
"""
