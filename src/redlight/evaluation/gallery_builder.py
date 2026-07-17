"""通用人工复核画廊生成器基类 (架构 A-D5)。

抽象层: 提供 HTML 外壳、CSS/JS、反馈机制、图像 I/O。
子类只需实现: 标题、选项、标注逻辑、时间线(可选)。

用法 (子类):
    builder = LightGalleryBuilder(eval_dir, frames_dir, feedback_path)
    builder.build(video_items, gt_map)

其中 video_items = {video_name: [item_dict, ...]}, gt_map = {video_name: gt}。
item_dict 结构由子类定义; 基类通过抽象方法委托给子类处理。
"""

import csv
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..infrastructure.image_utils import save_jpg


class BaseGalleryBuilder(ABC):
    """通用复核画廊基类。

    设计原则 (A-D5: 保持轻量, 勿造标注平台):
    - 只生成静态 HTML + 本地 JS(localStorage 回退), 无服务端依赖。
    - 图像标注、判定选项、原因选项由子类定制。
    - 公共部分 (布局/CSS/JS/反馈导出) 由基类统一。
    """

    # ---------- 子类必须实现的属性/方法 ----------

    @property
    @abstractmethod
    def title(self) -> str:
        """页面 <title> 与 <h1> 标题。"""
        ...

    @property
    @abstractmethod
    def toolbar_label(self) -> str:
        """顶部 toolbar 左侧标签。"""
        ...

    @abstractmethod
    def intro_html(self) -> str:
        """返回介绍段落 HTML (已包在 <p class="intro"> 内或自行包)。"""
        ...

    @abstractmethod
    def verdict_options(self) -> List[Tuple[str, str]]:
        """返回 [(value, label), ...] 判定选项。"""
        ...

    @abstractmethod
    def reason_options(self) -> List[Tuple[str, str]]:
        """返回 [(value, label), ...] 原因选项。"""
        ...

    @abstractmethod
    def is_mismatch(self, item: dict, gt: Any) -> bool:
        """判断一条 item 是否与 GT 构成 mismatch(需复核)。"""
        ...

    @abstractmethod
    def annotate_crop(self, frame: np.ndarray, item: dict, gt: Any) -> np.ndarray:
        """生成缩略图/特写图 (点小图弹大图前的预览)。"""
        ...

    @abstractmethod
    def annotate_full(self, frame: np.ndarray, item: dict, gt: Any) -> np.ndarray:
        """生成原始整帧标注图 (lightbox 弹出的大图)。"""
        ...

    @abstractmethod
    def item_meta_html(self, item: dict, gt: Any) -> str:
        """返回 crop-card 中 meta 行 HTML。

        示例: '预测 <b>red</b> / GT <b>green</b> <span>置信度 0.85</span>'
        """
        ...

    @abstractmethod
    def feedback_key(self, video: str, item: dict) -> Tuple[str, str]:
        """返回 feedback dict 的 lookup key: (video, identifier)。

        identifier 可以是 t_sec、frame_idx 等, 由子类决定。
        """
        ...

    # ---------- 子类可选覆写的方法 ----------

    def timeline_html(self, video: str, items: List[dict], gt: Any) -> str:
        """返回该视频的时间线 HTML; 默认空字符串。"""
        return ""

    def video_header_html(self, video: str, items: List[dict], gt: Any) -> str:
        """返回视频卡片头部 HTML (在 timeline 与 crop grid 之间)。

        默认只输出基本信息行; 子类可覆写添加额外信息。
        """
        return ""

    def extra_data_attrs(self, video: str, item: dict, gt: Any) -> Dict[str, str]:
        """返回额外 data-* 属性, 注入 crop-card 的 div 中。"""
        return {}

    def _extra_feedback_inputs_html(
        self, video: str, item: dict, gt: Any, fb: dict
    ) -> str:
        """返回 crop-card 反馈区中额外 input/select HTML; 默认空。

        子类如需 corrected_plate 等额外字段, 覆写此方法。
        """
        return ""

    def _extra_js(self) -> str:
        """返回追加到页面末尾的额外 JS(在基类 _JS 之后注入); 默认空。

        子类如需自定义交互(如画布拖拽标注几何坐标), 覆写此方法。
        基类保证在所有基础 JS 初始化完成后再执行本段, 可安全访问 .crop-card / #lb 等。
        """
        return ""

    # ---------- 公共基础设施 ----------

    def __init__(
        self,
        eval_dir: str,
        frames_dir: Optional[str] = None,
        feedback_path: Optional[str] = None,
        max_crops: int = 10,
    ):
        self.eval_dir = eval_dir
        self.frames_dir = frames_dir
        self.feedback_path = feedback_path
        self.max_crops = max_crops
        self._feedback = self._load_feedback()

    # ---- 反馈读写 ----

    def _load_feedback(self) -> Dict[Tuple[str, str], dict]:
        d: Dict[Tuple[str, str], dict] = {}
        if not self.feedback_path or not os.path.exists(self.feedback_path):
            return d
        with open(self.feedback_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                video = row.get("video", "")
                # t_sec 归一为一位小数, 与 feedback_key (f"{t:.1f}") 对齐, 避免 10.84≠10.8 匹配不上
                ident = row.get("t_sec", row.get("frame_idx", ""))
                try:
                    ident = f"{float(ident):.1f}"
                except (TypeError, ValueError):
                    pass
                # 保留 CSV 所有字段, 支持子类扩展(如 corrected_plate)
                d[(video, ident)] = dict(row)
        return d

    def _lookup_feedback(self, key: Tuple[str, str]) -> dict:
        return self._feedback.get(key, {"verdict": "", "reason": "", "note": ""})

    # ---- 图像 I/O ----

    def _resolve_frame(self, video: str, frame_idx: int) -> Optional[np.ndarray]:
        """从 frames_dir 加载单帧。子类可覆写以支持其他帧源(如 VideoSampler)。"""
        if not self.frames_dir:
            return None
        from ..infrastructure.image_utils import robust_imread

        fp = os.path.join(self.frames_dir, video, f"frame_{frame_idx:06d}.jpg")
        if os.path.exists(fp):
            return robust_imread(fp)
        return None

    def _save_image(self, img: np.ndarray, out_path: str) -> bool:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return save_jpg(img, out_path)

    # ---- 采样 ----

    def _sample_representatives(self, mismatches: List[dict]) -> List[dict]:
        """从 mismatch 列表中均匀采样, 最多 max_crops 条。"""
        if not mismatches:
            return []
        step = max(1, len(mismatches) // self.max_crops)
        return mismatches[::step][: self.max_crops]

    # ---- HTML 生成 ----

    def _build_option_html(
        self, options: List[Tuple[str, str]], selected: str, placeholder: str
    ) -> str:
        parts = [f'<option value="">{placeholder}</option>']
        for val, label in options:
            sel = ' selected' if val == selected else ""
            parts.append(f'<option value="{val}"{sel}>{label}</option>')
        return "".join(parts)

    def _build_card(
        self,
        video: str,
        item: dict,
        gt: Any,
        crop_rel: str,
        full_rel: str,
    ) -> str:
        fb = self._lookup_feedback(self.feedback_key(video, item))
        vv = fb.get("verdict", "")
        rr = fb.get("reason", "")
        nn = fb.get("note", "")
        done_badge = '<span class="badge-done">已标注</span>' if vv else ""
        saved_cls = ' saved' if vv else ''
        annotated = '1' if vv else '0'

        extra_attrs = ""
        for k, v in self.extra_data_attrs(video, item, gt).items():
            extra_attrs += f' data-{k}="{v}"'

        verdict_html = self._build_option_html(
            self.verdict_options(), vv, "--判定--"
        )
        reason_html = self._build_option_html(
            self.reason_options(), rr, "--原因--"
        )
        extra_inputs = self._extra_feedback_inputs_html(video, item, gt, fb)

        return f"""
            <div class="crop-card{saved_cls}" data-video="{video}" data-t="{item.get('t_sec', '')}"
                 data-idx="{item.get('frame_idx', '')}" data-annotated="{annotated}"{extra_attrs}>
              <div class="img-wrap">
                {done_badge}
                <span class="check" title="点击选中(Shift区间/Ctrl多选)">✅</span>
                <img class="zoom" src="{crop_rel}" data-full="{full_rel}"
                     style="width:200px;display:block;cursor:zoom-in;"/>
              </div>
              <div class="meta">{self.item_meta_html(item, gt)}{'<span class="done">已标</span>' if vv else ''}</div>
              <div class="fb">
                <select class="verdict">{verdict_html}</select>
                <select class="reason">{reason_html}</select>
                {extra_inputs}
                <input class="note" value="{nn}" placeholder="备注(可选)"/>
                <button class="save">保存</button>
                <span class="status"></span>
              </div>
            </div>"""

    def _build_video_section(
        self, video: str, items: List[dict], gt: Any
    ) -> str:
        mismatches = [it for it in items if self.is_mismatch(it, gt)]
        reps = self._sample_representatives(mismatches)

        crop_html_parts: List[str] = []
        for it in reps:
            # 优先使用 item 内嵌 frame(实时检测模式), 其次从 frames_dir 加载
            frame = it.get("frame")
            if frame is None:
                frame_idx = int(it.get("frame_idx", 0))
                frame = self._resolve_frame(video, frame_idx)
            if frame is None:
                continue
            tsec = f"{float(it.get('t_sec', 0)):.1f}"

            crop_dir = os.path.join(self.eval_dir, "crops", video)
            orig_dir = os.path.join(self.eval_dir, "orig", video)
            crop_path = os.path.join(crop_dir, f"t{tsec}.jpg")
            full_path = os.path.join(orig_dir, f"t{tsec}.jpg")

            crop_img = self.annotate_crop(frame, it, gt)
            full_img = self.annotate_full(frame, it, gt)

            if not self._save_image(crop_img, crop_path):
                continue
            self._save_image(full_img, full_path)

            crop_rel = os.path.relpath(crop_path, self.eval_dir).replace("\\", "/")
            full_rel = os.path.relpath(full_path, self.eval_dir).replace("\\", "/")
            crop_html_parts.append(
                self._build_card(video, it, gt, crop_rel, full_rel)
            )

        timeline = self.timeline_html(video, items, gt)
        header = self.video_header_html(video, items, gt)
        crop_grid = (
            "".join(crop_html_parts)
            if crop_html_parts
            else '<span style="color:#94a3b8;">无 (段内全对)</span>'
        )

        return f"""
        <div class="video-card">
          <h2>{video}</h2>
          {timeline}
          {header}
          <div class="crop-grid-label">代表性误差帧 (预测≠GT):</div>
          <div class="crop-grid">{crop_grid}</div>
        </div>"""

    # ---- 主入口 ----

    def build(
        self,
        video_items: Dict[str, List[dict]],
        gt_map: Dict[str, Any],
    ) -> str:
        """生成完整画廊 HTML 并写盘。

        Args:
            video_items: {video_name: [item_dict, ...]}
            gt_map: {video_name: gt_value}  (gt_value 结构由子类定义)

        Returns:
            生成的 HTML 文件路径
        """
        os.makedirs(self.eval_dir, exist_ok=True)

        sections: List[str] = []
        for video in sorted(video_items.keys()):
            gt = gt_map.get(video)
            items = video_items[video]
            if not items:
                continue
            sections.append(self._build_video_section(video, items, gt))

        cards_html = "".join(sections)
        html = self._render_page(cards_html)

        out_path = os.path.join(self.eval_dir, "gallery.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        return out_path

    # ---- HTML 模板 ----

    _CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f1f5f9;margin:0;padding:0 20px 40px;}
#toolbar{position:sticky;top:0;background:#0f172a;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:10;margin:0 -20px 18px;border-radius:0 0 8px 8px;font-size:13px;}
#toolbar b{font-size:15px;}
#toolbar .hint{opacity:.72;font-weight:400;}
#export{margin-left:auto;background:#fff;color:#0f172a;border:none;border-radius:4px;padding:5px 12px;cursor:pointer;font-size:12px;}
h1{color:#0f172a;margin:8px 0;}
.intro{color:#475569;font-size:13px;margin:0 0 14px;}
.video-card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.06);}
.video-card h2{margin:0 0 8px;font-size:18px;color:#0f172a;}
.crop-grid-label{font-size:12px;margin:10px 0 4px;color:#475569;}
.crop-grid{display:flex;flex-wrap:wrap;gap:8px;}
.crop-card{border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;width:212px;background:#fff;}
.crop-card.saved{box-shadow:0 0 0 2px #22c55e inset;}
.meta{font-size:11px;padding:4px 6px;color:#334155;}
.done{color:#16a34a;font-size:10px;margin-left:4px;}
.badge-done{position:absolute;top:0;right:0;background:#22c55e;color:#fff;font-size:11px;font-weight:700;padding:2px 8px;border-bottom-left-radius:8px;z-index:6;box-shadow:0 1px 3px rgba(0,0,0,.2);}
.fbtn{background:#334155;color:#cbd5e1;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px;margin-left:4px;}
.fbtn.active{background:#3b82f6;color:#fff;}
#filter-count{font-size:11px;color:#94a3b8;margin-left:8px;}
.conf{display:inline-block;color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:4px;font-weight:700;}
.fb{padding:6px;display:flex;flex-direction:column;gap:4px;font-size:11px;background:#f8fafc;}
.fb select,.fb input{font-size:11px;padding:3px;border:1px solid #cbd5e1;border-radius:4px;}
.fb .save{background:#0f172a;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;align-self:flex-start;}
.fb .status{font-size:10px;}
.zoom{cursor:zoom-in;}
.img-wrap{position:relative;}
.check{position:absolute;top:4px;left:4px;width:22px;height:22px;border-radius:4px;background:rgba(255,255,255,.9);display:flex;align-items:center;justify-content:center;font-size:15px;cursor:pointer;z-index:5;user-select:none;border:2px solid #94a3b8;color:transparent;line-height:1;}
.check:hover{border-color:#3b82f6;}
.crop-card.selected .check{background:#3b82f6;color:#fff;border-color:#3b82f6;}
.crop-card{cursor:pointer;}
.crop-card.selected{box-shadow:0 0 0 3px #3b82f6 inset;}
.crop-card.selected .meta{background:#eff6ff;}
#batch-bar{position:sticky;top:54px;background:#1e293b;color:#fff;padding:8px 16px;display:flex;gap:8px;align-items:center;z-index:9;margin:0 0 12px;font-size:12px;flex-wrap:wrap;border-radius:6px;}
#batch-bar b{font-size:13px;}
#batch-bar select,#batch-bar input{font-size:12px;padding:3px;border:1px solid #475569;border-radius:4px;background:#0f172a;color:#fff;}
#batch-bar button{background:#3b82f6;color:#fff;border:none;border-radius:4px;padding:5px 12px;cursor:pointer;font-size:12px;}
#batch-bar #batch-clear{background:#475569;}
#batch-bar #batch-status{font-size:11px;color:#94a3b8;}
.lightbox{position:fixed;inset:0;background:rgba(15,23,42,.9);display:none;align-items:center;justify-content:center;z-index:200;}
.lightbox.show{display:flex;}
.lightbox img{max-width:94vw;max-height:94vh;border:2px solid #fff;border-radius:6px;box-shadow:0 8px 40px rgba(0,0,0,.6);}
.lightbox .hint{position:absolute;bottom:18px;color:#cbd5e1;font-size:12px;}
"""

    _JS = """
function showProtocolWarn(){
  if(location.protocol!=='file:') return;
  const w=document.createElement('div');
  w.id='protocol-warn';
  w.style.cssText='position:sticky;top:0;left:0;right:0;z-index:9999;background:#dc2626;color:#fff;padding:10px 16px;font-size:13px;font-weight:600;text-align:center;line-height:1.5;';
  w.innerHTML='⚠️ 你正用 <b>file://</b> 直接打开本页，标注不会写入服务器 CSV（仅临时存浏览器，刷新即丢）。请改用 <a href="http://localhost:8765/" target="_blank" style="color:#fff;text-decoration:underline;">http://localhost:8765/</a> 打开。';
  document.body.insertBefore(w, document.body.firstChild);
}
showProtocolWarn();
function markSaved(){
  fetch('/feedback').then(r=>r.json()).then(j=>{
    if(!j || !j.ok) return;
    const byKey={};
    (j.rows||[]).forEach(r=>{ byKey[r.video+'|'+r.t_sec+'|'+r.frame_idx]=r; });
    document.querySelectorAll('.crop-card').forEach(card=>{
      const key=card.dataset.video+'|'+card.dataset.t+'|'+card.dataset.idx;
      if(byKey[key]){
        card.dataset.annotated='1'; card.classList.add('saved');
        const iw=card.querySelector('.img-wrap');
        if(iw && !card.querySelector('.badge-done')) iw.insertAdjacentHTML('afterbegin','<span class="badge-done">已标注</span>');
        const m=card.querySelector('.meta'); if(m && !card.querySelector('.done')) m.insertAdjacentHTML('beforeend',' <span class="done">已标</span>');
      }
    });
    if(typeof applyFilter==='function') applyFilter('all');
    if(typeof updateCount==='function') updateCount();
  }).catch(()=>{});
}
markSaved();
async function postFeedback(p){
  try{
    const r = await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    if(r.ok) return 'ok';
  }catch(e){}
  const key = 'fb_'+p.video+'_'+(p.t!==undefined?p.t:p.idx);
  localStorage.setItem(key, JSON.stringify(p));
  return 'local';
}
function updateCount(){ document.getElementById('saved').textContent = document.querySelectorAll('.crop-card.saved').length; }
document.querySelectorAll('.crop-card .save').forEach(btn=>{
  btn.addEventListener('click', async ()=>{
    const card = btn.closest('.crop-card');
    const p = { video:card.dataset.video, t:card.dataset.t, idx:card.dataset.idx };
    // 自动收集 .fb 区域内所有 input/select(按 class name 映射为字段名)
    const fb = card.querySelector('.fb');
    if(fb){
      fb.querySelectorAll('input,select').forEach(el=>{
        if(el.classList.contains('save') || el.classList.contains('status')) return;
        const key = el.className || el.tagName.toLowerCase();
        p[key] = el.value;
      });
    }
    // 把子类注入的 data-* 属性也带上,便于导出时区分类型
    for(const attr of card.attributes){
      if(attr.name.startsWith('data-') && !['data-video','data-t','data-idx'].includes(attr.name)){
        p[attr.name.slice(5)] = attr.value;
      }
    }
    const st = card.querySelector('.status');
    if(!p.verdict){ st.textContent='请先选判定'; st.style.color='#dc2626'; return; }
    st.textContent='保存中...'; st.style.color='#64748b';
    const res = await postFeedback(p);
    if(res==='ok'){ st.textContent='已保存'; st.style.color='#16a34a'; card.classList.add('saved'); card.dataset.annotated='1';
      const iw = card.querySelector('.img-wrap');
      if(iw && !card.querySelector('.badge-done')) iw.insertAdjacentHTML('afterbegin','<span class="badge-done">已标注</span>');
      if(!card.querySelector('.done')) card.querySelector('.meta').insertAdjacentHTML('beforeend',' <span class="done">已标</span>'); }
    else { st.textContent='⚠️ 未存服务器(仅浏览器临时,刷新丢)'; st.style.color='#dc2626'; card.classList.add('saved'); }
    updateCount();
  });
});
document.getElementById('export').addEventListener('click', ()=>{
  const items=[]; for(let i=0;i<localStorage.length;i++){const k=localStorage.key(i); if(k&&k.startsWith('fb_')) items.push(JSON.parse(localStorage.getItem(k)));}
  const blob=new Blob([JSON.stringify(items,null,2)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='feedback_local.json'; a.click();
});
updateCount();
const lb=document.getElementById('lb'), lbImg=lb.querySelector('img');
document.querySelectorAll('img.zoom').forEach(im=>{
  im.addEventListener('click',()=>{ lbImg.src=im.dataset.full; lb.classList.add('show'); });
});
lb.addEventListener('click',()=>lb.classList.remove('show'));
// ESC 关闭大图灯箱, 返回画廊
document.addEventListener('keydown',e=>{
  if(e.key==='Escape' && lb.classList.contains('show')) lb.classList.remove('show');
});

// ---- 批量标注: Shift区间选 / Ctrl单选 / 应用到选中 ----
let lastSelIdx = -1;
const cards = () => Array.from(document.querySelectorAll('.crop-card'));
function selIdx(card){ return cards().indexOf(card); }
function updSelCount(){ document.getElementById('sel').textContent = document.querySelectorAll('.crop-card.selected').length; }
function setSel(card, on){ card.classList.toggle('selected', on); }
// 统一选中行为:
//   普通点未选中=单选(取消其他); 普通点已选中=取消该张(toggle, 直觉式取消单个)
//   Shift+点=区间选; Ctrl/Cmd+点=多选增减(不影响其他已选)
function selectCard(card, e){
  const idx = selIdx(card);
  const wasSel = card.classList.contains('selected');
  if(e.shiftKey && lastSelIdx >= 0){
    const [a,b] = [Math.min(lastSelIdx,idx), Math.max(lastSelIdx,idx)];
    cards().forEach((c,i)=> setSel(c, i>=a && i<=b));
  } else if(e.ctrlKey || e.metaKey){
    setSel(card, !wasSel);
    lastSelIdx = idx;
  } else {
    // 普通点: 已选中则取消该张, 未选中则单选它(取消其他)
    if(wasSel){
      setSel(card, false);
    } else {
      cards().forEach(c=> setSel(c, false));
      setSel(card, true);
      lastSelIdx = idx;
    }
  }
  updSelCount();
}
// 点卡片(非控件区)选中; 点 .check 勾选按钮也选中
document.querySelectorAll('.crop-card').forEach(card=>{
  card.addEventListener('click', e=>{
    // 点图片放大、或点 .fb 内控件(保存/下拉/输入)时不触发选中
    if(e.target.closest('.zoom') || e.target.closest('.fb')) return;
    selectCard(card, e);
  });
  // 左上角 ✅ 勾选按钮: 点它切换选中(支持 Shift/Ctrl), 阻止冒泡避免二次触发
  const chk = card.querySelector('.check');
  if(chk){
    chk.addEventListener('click', e=>{
      e.stopPropagation();
      selectCard(card, e);
    });
  }
});
document.getElementById('batch-clear').addEventListener('click', ()=>{
  cards().forEach(c=> setSel(c,false)); lastSelIdx=-1; updSelCount();
});
// 批量应用: 把选定 verdict/reason/note 写入每张选中卡片的 .fb 控件, 然后逐张保存
document.getElementById('batch-apply').addEventListener('click', async ()=>{
  const sel = Array.from(document.querySelectorAll('.crop-card.selected'));
  const st = document.getElementById('batch-status');
  if(!sel.length){ st.textContent='未选中任何帧'; st.style.color='#f87171'; return; }
  const bv = document.getElementById('batch-verdict').value;
  if(!bv){ st.textContent='请先选判定'; st.style.color='#f87171'; return; }
  const br = document.getElementById('batch-reason').value;
  const bn = document.getElementById('batch-note').value;
  st.textContent='批量保存中 0/'+sel.length; st.style.color='#94a3b8';
  let done = 0;
  for(const card of sel){
    const fb = card.querySelector('.fb');
    if(fb){
      if(bv){ const v=fb.querySelector('.verdict'); if(v) v.value=bv; }
      if(br){ const r=fb.querySelector('.reason'); if(r) r.value=br; }
      if(bn){ const n=fb.querySelector('.note'); if(n) n.value=bn; }
    }
    // 复用单张保存逻辑: 触发该卡的 save 按钮点击
    const saveBtn = card.querySelector('.save');
    if(saveBtn){ saveBtn.click(); }
    done++; st.textContent='批量保存中 '+done+'/'+sel.length;
    await new Promise(r=>setTimeout(r,30)); // 给 postFeedback 一点喘息, 避免服务并发丢
  }
  st.textContent='已批量保存 '+done+' 帧 ✓'; st.style.color='#4ade80';
  updSelCount();
});
// ---- 复制选中 / 粘贴到选中 (剪贴板式) ----
let clip = null; // {verdict, reason, note}
function updClip(){
  const el = document.getElementById('clip');
  if(!clip || !clip.verdict){ el.textContent='空'; el.style.color='#94a3b8'; return; }
  el.textContent = `${clip.verdict} / ${clip.reason||'-'} / ${clip.note||'-'}`;
  el.style.color='#4ade80';
  el.parentElement.title = el.textContent;
}
// 复制选中: 取第一个选中帧的 verdict/reason/note 存剪贴板
document.getElementById('copy-sel').addEventListener('click', ()=>{
  const st = document.getElementById('batch-status');
  const sel = Array.from(document.querySelectorAll('.crop-card.selected'));
  if(!sel.length){ st.textContent='先选中一帧再复制'; st.style.color='#f87171'; return; }
  const fb = sel[0].querySelector('.fb');
  const v = fb.querySelector('.verdict').value;
  if(!v){ st.textContent='选中的帧未标判定, 无法复制'; st.style.color='#f87171'; return; }
  clip = {
    verdict: v,
    reason: fb.querySelector('.reason').value,
    note: fb.querySelector('.note').value,
  };
  st.textContent='已复制选中帧的标注 ✓'; st.style.color='#4ade80';
  updClip();
});
// 粘贴到选中: 把剪贴板内容填入每张选中帧并保存
document.getElementById('paste-sel').addEventListener('click', async ()=>{
  const st = document.getElementById('batch-status');
  if(!clip || !clip.verdict){ st.textContent='剪贴板空, 先选中一帧点"复制选中"'; st.style.color='#f87171'; return; }
  const sel = Array.from(document.querySelectorAll('.crop-card.selected'));
  if(!sel.length){ st.textContent='先选中目标帧(可多选)再粘贴'; st.style.color='#f87171'; return; }
  st.textContent='粘贴保存中 0/'+sel.length; st.style.color='#94a3b8';
  let done = 0;
  for(const card of sel){
    const fb = card.querySelector('.fb');
    fb.querySelector('.verdict').value = clip.verdict;
    fb.querySelector('.reason').value = clip.reason || '';
    fb.querySelector('.note').value = clip.note || '';
    const saveBtn = card.querySelector('.save');
    if(saveBtn) saveBtn.click();
    done++; st.textContent='粘贴保存中 '+done+'/'+sel.length;
    await new Promise(r=>setTimeout(r,30));
  }
  st.textContent='已粘贴到 '+done+' 帧 ✓'; st.style.color='#4ade80';
});
// ---- 筛选: 全部 / 已标注 / 未标注 ----
function applyFilter(mode){
  let vis=0, done=0, undone=0;
  cards().forEach(c=>{
    const ann = c.dataset.annotated==='1';
    if(ann) done++; else undone++;
    const show = mode==='all' || (mode==='done'&&ann) || (mode==='undone'&&!ann);
    c.style.display = show ? '' : 'none';
    if(show) vis++;
  });
  document.querySelectorAll('.video-card').forEach(vc=>{
    const anyVisible = Array.from(vc.querySelectorAll('.crop-card')).some(c=>c.style.display!=='none');
    vc.style.display = anyVisible ? '' : 'none';
  });
  const fc = document.getElementById('filter-count');
  if(fc) fc.textContent = `显示 ${vis} (已标注 ${done} / 未标注 ${undone})`;
}
document.querySelectorAll('.fbtn').forEach(b=> b.addEventListener('click', ()=>{
  document.querySelectorAll('.fbtn').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
  applyFilter(b.id.replace('f-',''));
}));
applyFilter('all');
updClip();
updSelCount();
"""

    def _render_page(self, cards_html: str) -> str:
        return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{self.title}</title>
<style>{self._CSS}</style></head><body>
<div id="toolbar">
  <b>{self.toolbar_label}</b>
  <span>已保存 <span id="saved">0</span> 帧</span>
  <span class="hint">单张点"保存"; 批量: 点☐选中(Shift区间/Ctrl多选/再点取消) → 选判定/原因 → "应用到选中"; 或"复制选中"→选目标→"粘贴到选中"</span>
  <button id="export">导出本地标注(JSON)</button>
</div>
<div id="batch-bar">
  <b>批量标注</b>
  <span>已选 <span id="sel">0</span> 帧</span>
  <select id="batch-verdict">{self._build_option_html(self.verdict_options(), '', '--判定--')}</select>
  <select id="batch-reason">{self._build_option_html(self.reason_options(), '', '--原因--')}</select>
  <input id="batch-note" placeholder="备注(可选)"/>
  <button id="batch-apply">应用到选中</button>
  <button id="batch-clear">清空选择</button>
  <button id="copy-sel">复制选中</button>
  <button id="paste-sel">粘贴到选中</button>
  <span style="margin-left:12px;border-left:1px solid #475569;padding-left:12px;">筛选:
    <button id="f-all" class="fbtn active">全部</button>
    <button id="f-done" class="fbtn">已标注</button>
    <button id="f-undone" class="fbtn">未标注</button>
    <span id="filter-count"></span>
  </span>
  <span style="margin-left:12px;border-left:1px solid #475569;padding-left:12px;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="">剪贴板: <span id="clip" style="color:#94a3b8;">空</span></span>
  <span id="batch-status"></span>
</div>
<h1>{self.title}</h1>
<p class="intro">{self.intro_html()}</p>
{cards_html}
<div id="lb" class="lightbox"><img alt="zoom"/><div class="hint">点击任意处关闭</div></div>
<script>{self._JS}
{self._extra_js()}</script>
</body></html>"""
