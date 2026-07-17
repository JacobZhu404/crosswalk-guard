"""跟踪标注画廊 (BaseGalleryBuilder 实现, Part B2)。

用途: 人工标注每个锚帧的**违章车框 box**[x0,y0,x1,y1] 作为 GT,
供 eval_tracking.py 的 ID 碎片化 + 静止判定准确率评测(锚定用)。

复用画廊基类, 交互为**画布拖拽矩形框**(镜像 crosswalk 的竖向带)。

video_items 期望: {video: [{"t_sec","frame_idx","frame"(np),"note"}]}
gt_map: {video: 原始 tracking 骨架 dict}(仅透传, 不强用)
"""
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .gallery_builder import BaseGalleryBuilder


class TrackingGalleryBuilder(BaseGalleryBuilder):
    """违章车 box 标注画廊。"""

    @property
    def title(self) -> str:
        return "跟踪标注画廊 (B2: 框违章车)"

    @property
    def toolbar_label(self) -> str:
        return "违章车 box 标注"

    def intro_html(self) -> str:
        return (
            "每张=一个锚帧。<b>标注方式:</b> 点小图放大 → 在大图上<b>拖拽矩形框</b>框住违章车 → "
            "松手自动填入 box=x0,y0,x1,y1(原始像素) → 关灯箱后点<b>保存</b>。"
            "<br/>判定: <b>已框车</b> / <b>此帧无违章车</b>(留空 box) / <b>看不清</b>。拖拽会自动把判定设为“已框车”。ESC 关灯箱。"
        )

    def verdict_options(self) -> List[Tuple[str, str]]:
        return [("labeled", "已框车"), ("no_car", "此帧无违章车"), ("uncertain", "看不清")]

    def reason_options(self) -> List[Tuple[str, str]]:
        return []

    def is_mismatch(self, item: dict, gt: Any) -> bool:
        return True

    def _sample_representatives(self, mismatches: List[dict]) -> List[dict]:
        return list(mismatches)

    def annotate_crop(self, frame: np.ndarray, item: dict, gt: Any) -> np.ndarray:
        return _draw_ruler(frame, item.get("note", ""), float(item.get("t_sec", 0)), thumb_w=320)

    def annotate_full(self, frame: np.ndarray, item: dict, gt: Any) -> np.ndarray:
        return _draw_ruler(frame, item.get("note", ""), float(item.get("t_sec", 0)), thumb_w=None)

    def item_meta_html(self, item: dict, gt: Any) -> str:
        return (f't={float(item.get("t_sec", 0)):.1f}s'
                f'<br/><span style="color:#64748b;font-size:10px;">{item.get("note", "")}</span>')

    def feedback_key(self, video: str, item: dict) -> Tuple[str, str]:
        return (video, f"{float(item.get('t_sec', 0)):.1f}")

    def extra_data_attrs(self, video: str, item: dict, gt: Any) -> Dict[str, str]:
        H, W = 0, 0
        fr = item.get("frame")
        if fr is not None:
            H, W = fr.shape[:2]
        return {"imgw": str(W), "imgh": str(H)}

    def _extra_feedback_inputs_html(self, video: str, item: dict, gt: Any, fb: dict) -> str:
        box = fb.get("box", "")
        return (
            '<div style="display:flex;gap:4px;align-items:center;">'
            '<span style="font-size:10px;color:#475569;">box</span>'
            f'<input class="box" value="{box}" style="width:150px;" placeholder="x0,y0,x1,y1"/>'
            '<button type="button" class="clear-box" style="font-size:10px;padding:2px 6px;">清除</button>'
            '</div>'
        )

    def _extra_js(self) -> str:
        return _TRACKING_DRAW_JS


def _draw_ruler(frame, note, ts, thumb_w=None):
    out = frame.copy()
    H, W = out.shape[:2]
    for y in range(0, H, 100):
        cv2.line(out, (0, y), (24, y), (0, 255, 0), 1)
        cv2.putText(out, str(y), (2, min(H - 2, y + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    for x in range(0, W, 100):
        cv2.line(out, (x, 0), (x, 24), (0, 255, 0), 1)
        cv2.putText(out, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.rectangle(out, (0, 0), (W - 1, 26), (0, 0, 0), -1)
    cv2.putText(out, f"t={ts:.1f}s  {note}  拖拽框住违章车",
                (28, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if thumb_w and W > thumb_w:
        scale = thumb_w / W
        out = cv2.resize(out, (thumb_w, int(H * scale)))
    return out


_TRACKING_DRAW_JS = r"""
// ===== 违章车 box 画布拖拽标注 =====
(function(){
  const lb = document.getElementById('lb');
  if(!lb) return;
  const lbImg = lb.querySelector('img');
  let srcCard = null;
  const cv = document.createElement('canvas');
  cv.id = 'anno-canvas';
  cv.style.cssText = 'position:fixed;z-index:210;cursor:crosshair;display:none;';
  document.body.appendChild(cv);
  const ctx = cv.getContext('2d');
  let dragging=false, xA=0,yA=0,xB=0,yB=0;

  document.querySelectorAll('img.zoom').forEach(im=>{
    im.addEventListener('click', ()=>{ srcCard = im.closest('.crop-card'); });
  });
  function place(){
    const r = lbImg.getBoundingClientRect();
    cv.style.left=r.left+'px'; cv.style.top=r.top+'px';
    cv.width=r.width; cv.height=r.height;
    cv.style.width=r.width+'px'; cv.style.height=r.height+'px';
    cv.style.display = lb.classList.contains('show') ? 'block' : 'none';
    redraw();
  }
  function redraw(){
    ctx.clearRect(0,0,cv.width,cv.height);
    if(xB!==xA || yB!==yA){
      const x0=Math.min(xA,xB), y0=Math.min(yA,yB), w=Math.abs(xB-xA), h=Math.abs(yB-yA);
      ctx.fillStyle='rgba(239,68,68,0.20)'; ctx.fillRect(x0,y0,w,h);
      ctx.strokeStyle='#ef4444'; ctx.lineWidth=2; ctx.strokeRect(x0,y0,w,h);
    }
  }
  function toNat(dx,dy){
    const nw=lbImg.naturalWidth||1, nh=lbImg.naturalHeight||1;
    return [Math.round(Math.max(0,Math.min(nw, dx*nw/(cv.width||1)))),
            Math.round(Math.max(0,Math.min(nh, dy*nh/(cv.height||1))))];
  }
  cv.addEventListener('mousedown', e=>{
    e.stopPropagation(); dragging=true;
    const r=cv.getBoundingClientRect(); xA=e.clientX-r.left; yA=e.clientY-r.top; xB=xA; yB=yA; redraw();
  });
  cv.addEventListener('mousemove', e=>{
    if(!dragging) return;
    const r=cv.getBoundingClientRect(); xB=e.clientX-r.left; yB=e.clientY-r.top; redraw();
  });
  window.addEventListener('mouseup', ()=>{
    if(!dragging) return; dragging=false;
    if((Math.abs(xB-xA)<4 && Math.abs(yB-yA)<4) || !srcCard) return;
    const [nx0,ny0]=toNat(Math.min(xA,xB),Math.min(yA,yB));
    const [nx1,ny1]=toNat(Math.max(xA,xB),Math.max(yA,yB));
    const bi=srcCard.querySelector('.box');
    if(bi) bi.value = nx0+','+ny0+','+nx1+','+ny1;
    const vsel=srcCard.querySelector('.verdict'); if(vsel) vsel.value='labeled';
    const st=srcCard.querySelector('.status');
    if(st){ st.textContent='已框 '+nx0+','+ny0+','+nx1+','+ny1+' (关灯箱后点保存)'; st.style.color='#2563eb'; }
  });
  cv.addEventListener('click', e=>e.stopPropagation());
  const mo=new MutationObserver(()=>{ if(lb.classList.contains('show')){ setTimeout(place,30);} else { cv.style.display='none'; xA=xB=yA=yB=0; } });
  mo.observe(lb,{attributes:true,attributeFilter:['class']});
  lbImg.addEventListener('load', ()=>{ if(lb.classList.contains('show')) setTimeout(place,10); });
  window.addEventListener('resize', ()=>{ if(lb.classList.contains('show')) place(); });
  document.querySelectorAll('.clear-box').forEach(btn=>{
    btn.addEventListener('click', e=>{ e.stopPropagation(); const b=btn.closest('.crop-card').querySelector('.box'); if(b) b.value=''; });
  });
})();
"""
