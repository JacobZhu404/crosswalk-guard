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
                # 优先用 t_sec; 其次 frame_idx; 兜底空字符串
                ident = row.get("t_sec", row.get("frame_idx", ""))
                d[(video, ident)] = {
                    "verdict": row.get("verdict", ""),
                    "reason": row.get("reason", ""),
                    "note": row.get("note", ""),
                }
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
        vv, rr, nn = fb["verdict"], fb["reason"], fb["note"]
        done_badge = ' <span class="done">已标</span>' if vv else ""

        extra_attrs = ""
        for k, v in self.extra_data_attrs(video, item, gt).items():
            extra_attrs += f' data-{k}="{v}"'

        verdict_html = self._build_option_html(
            self.verdict_options(), vv, "--判定--"
        )
        reason_html = self._build_option_html(
            self.reason_options(), rr, "--原因--"
        )

        return f"""
            <div class="crop-card" data-video="{video}" data-t="{item.get('t_sec', '')}"
                 data-idx="{item.get('frame_idx', '')}"{extra_attrs}>
              <img class="zoom" src="{crop_rel}" data-full="{full_rel}"
                   style="width:200px;display:block;cursor:zoom-in;"/>
              <div class="meta">{self.item_meta_html(item, gt)}{done_badge}</div>
              <div class="fb">
                <select class="verdict">{verdict_html}</select>
                <select class="reason">{reason_html}</select>
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
.conf{display:inline-block;color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:4px;font-weight:700;}
.fb{padding:6px;display:flex;flex-direction:column;gap:4px;font-size:11px;background:#f8fafc;}
.fb select,.fb input{font-size:11px;padding:3px;border:1px solid #cbd5e1;border-radius:4px;}
.fb .save{background:#0f172a;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;align-self:flex-start;}
.fb .status{font-size:10px;}
.zoom{cursor:zoom-in;}
.lightbox{position:fixed;inset:0;background:rgba(15,23,42,.9);display:none;align-items:center;justify-content:center;z-index:200;}
.lightbox.show{display:flex;}
.lightbox img{max-width:94vw;max-height:94vh;border:2px solid #fff;border-radius:6px;box-shadow:0 8px 40px rgba(0,0,0,.6);}
.lightbox .hint{position:absolute;bottom:18px;color:#cbd5e1;font-size:12px;}
"""

    _JS = """
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
    const p = { video:card.dataset.video, t:card.dataset.t, idx:card.dataset.idx,
                verdict:card.querySelector('.verdict').value,
                reason:card.querySelector('.reason').value,
                note:card.querySelector('.note').value };
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
    if(res==='ok'){ st.textContent='已保存'; st.style.color='#16a34a'; card.classList.add('saved');
      if(!card.querySelector('.done')) card.querySelector('.meta').insertAdjacentHTML('beforeend',' <span class="done">已标</span>'); }
    else { st.textContent='已存本地(无服务)'; st.style.color='#d97706'; card.classList.add('saved'); }
    const nxt = card.nextElementSibling;
    if(nxt && nxt.classList.contains('crop-card')){
      nxt.querySelector('.verdict').value = p.verdict;
      nxt.querySelector('.reason').value = p.reason;
      nxt.querySelector('.note').value = p.note;
    }
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
"""

    def _render_page(self, cards_html: str) -> str:
        return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{self.title}</title>
<style>{self._CSS}</style></head><body>
<div id="toolbar">
  <b>{self.toolbar_label}</b>
  <span>已保存 <span id="saved">0</span> 帧</span>
  <span class="hint">每张图下方点"保存"即可标注</span>
  <button id="export">导出本地标注(JSON)</button>
</div>
<h1>{self.title}</h1>
<p class="intro">{self.intro_html()}</p>
{cards_html}
<div id="lb" class="lightbox"><img alt="zoom"/><div class="hint">点击任意处关闭</div></div>
<script>{self._JS}</script>
</body></html>"""
