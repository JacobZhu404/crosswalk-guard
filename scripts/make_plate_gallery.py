"""生成车牌识别误差确认画廊 (HTML): 每视频并排 识别/GT 车牌对比 + 代表性误差帧车牌特写。

用途: 用户滚动逐张判定关键帧，填写正确车牌文本，建立GT数据集。
标注结果可直接用于算法迭代和回归测试，形成 标注→评测→迭代 的闭环。

风格与灯态画廊保持一致，便于统一标注体验。

用法:
  python scripts/make_plate_gallery.py
  python scripts/make_plate_gallery.py --videos 违章02
"""
import os
import sys
import csv
import glob
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer


def load_gt(gt_csv):
    g = {}
    try:
        with open(gt_csv, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                video = r["video"]
                plates = []
                for field in ["violating_plates", "other_plates"]:
                    if r.get(field):
                        for p in r[field].split(";"):
                            p = p.strip()
                            if p and p != "?" and p != "无牌":
                                plates.append(p)
                if plates:
                    g[video] = list(set(plates))
    except (PermissionError, FileNotFoundError):
        print(f"警告: 无法读取GT文件 {gt_csv}, 使用命令行指定的GT车牌")
    return g


def load_feedback(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            k = (r.get("video", ""), r.get("frame_idx", ""))
            d[k] = {"verdict": r.get("verdict", ""), "reason": r.get("reason", ""),
                    "note": r.get("note", ""), "corrected_plate": r.get("corrected_plate", "")}
    return d


def _save_jpg(img, out_path):
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return False
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    return True


def _conf_bgr(conf):
    if conf >= 0.85:
        return (0, 180, 0)
    if conf >= 0.6:
        return (0, 215, 230)
    return (0, 0, 220)


def annotate_crop(fr, plate_info, label_txt, conf=0.0):
    xyxy = plate_info.get("xyxy", [0, 0, 0, 0])
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    
    H, W = fr.shape[:2]
    roi = 40
    x1_crop = max(0, x1 - roi)
    y1_crop = max(0, y1 - roi)
    x2_crop = min(W, x2 + roi)
    y2_crop = min(H, y2 + roi)
    
    crop = fr[y1_crop:y2_crop, x1_crop:x2_crop].copy()
    Hc, Wc = crop.shape[:2]
    if Hc < 4 or Wc < 4:
        return fr[:200, :300].copy() if len(fr.shape) == 3 else fr[:200, :300]
    
    cv2.rectangle(crop, (max(0, x1 - x1_crop), max(0, y1 - y1_crop)),
                  (min(Wc - 1, x2 - x1_crop), min(Hc - 1, y2 - y1_crop)),
                  (0, 0, 255), 2)
    
    cv2.rectangle(crop, (0, 0), (Wc - 1, 30), (0, 0, 0), -1)
    cv2.putText(crop, label_txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cb = _conf_bgr(conf)
    bar_w = 70
    cv2.rectangle(crop, (Wc - bar_w - 4, 4), (Wc - 4, 26), cb, -1)
    cv2.putText(crop, f"{conf:.2f}", (Wc - bar_w, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    
    return crop


def annotate_full(fr, plate_info, label_txt, conf=0.0):
    out = fr.copy()
    H, W = out.shape[:2]
    
    for p in plate_info.get("all_plates", [plate_info]):
        xyxy = p.get("xyxy", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = p.get("text", "")
        p_conf = p.get("conf", 0.0)
        cv2.putText(out, f"{text} ({p_conf:.2f})", (x1, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.rectangle(out, (0, 0), (W - 1, 32), (0, 0, 0), -1)
    cv2.putText(out, label_txt + "   红框=车牌检测框", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cb = _conf_bgr(conf)
    cv2.rectangle(out, (W - 130, 4), (W - 8, 28), cb, -1)
    cv2.putText(out, f"置信度 {conf:.2f}", (W - 126, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    return out


def analyze_video(video_name, gt_plates, frames_dir, eval_dir, max_crops=10):
    video_path = os.path.join(ROOT, "input_video", f"{video_name}.mp4")
    if not os.path.exists(video_path):
        return [], []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []
    
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps) if src_fps > 0 else 0
    
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    plate = PlateRecognizer(cfg, verbose=False)
    
    candidates = []
    seen_plates = set()
    frame_idx = 0
    
    while True:
        ret = cap.grab()
        if not ret:
            break
        
        if frame_idx % 8 == 0:
            ret, frame = cap.retrieve()
            if not ret:
                frame_idx += 1
                continue
            
            ts = frame_idx / src_fps
            plates = plate.detect(frame)
            
            for p in plates:
                if not p.get("text"):
                    continue
                
                text = p["text"]
                conf = p.get("conf", 0.0)
                matched = any(text == gt for gt in gt_plates)
                
                priority = 0
                if not matched:
                    priority += 3
                if conf < 0.6:
                    priority += 2
                if text not in seen_plates:
                    priority += 1
                    seen_plates.add(text)
                
                candidates.append({
                    "frame_idx": frame_idx,
                    "t_sec": ts,
                    "detected": text,
                    "conf": conf,
                    "plate_info": p,
                    "all_plates": plates,
                    "frame": frame.copy(),
                    "priority": priority,
                    "matched": matched,
                })
        
        frame_idx += 1
    
    cap.release()
    
    candidates.sort(key=lambda x: (-x["priority"], x["t_sec"]))
    reps = candidates[:max_crops]
    reps.sort(key=lambda x: x["t_sec"])
    
    return reps, total_dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "output", "plate_eval"))
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "events.csv"))
    ap.add_argument("--gt-plates", nargs="*", default=None, help="GT车牌列表(如: 京LNE560 京ADH9206)")
    ap.add_argument("--feedback", default=os.path.join(ROOT, "data", "output", "annotated", "plate_feedback.csv"))
    ap.add_argument("--max-crops", type=int, default=10)
    args = ap.parse_args()
    
    gt = load_gt(args.gt)
    
    if args.gt_plates and args.videos:
        for v in args.videos:
            gt[v] = args.gt_plates
    
    feedback = load_feedback(args.feedback)
    
    videos = args.videos or list(gt.keys())
    
    cards = []
    os.makedirs(args.eval_dir, exist_ok=True)
    
    for v in videos:
        gt_plates = gt.get(v, [])
        
        reps, total_dur = analyze_video(v, gt_plates, args.frames_dir, args.eval_dir, args.max_crops)
        
        crop_html = ""
        crop_dir = os.path.join(args.eval_dir, "crops", v)
        os.makedirs(crop_dir, exist_ok=True)
        
        for m in reps:
            idx = m["frame_idx"]
            tsec = f"{m['t_sec']:.1f}"
            conf = float(m.get("conf", 0.0))
            detected = m["detected"]
            frame = m["frame"]
            plate_info = m["plate_info"]
            plate_info["all_plates"] = m["all_plates"]
            
            lab = f"t={tsec}s 识{detected}/GT{','.join(gt_plates)}"
            
            crop = annotate_crop(frame, plate_info, lab, conf=conf)
            out = os.path.join(crop_dir, f"t{tsec}.jpg")
            if not _save_jpg(crop, out):
                continue
            rel = os.path.relpath(out, args.eval_dir).replace("\\", "/")
            
            full = annotate_full(frame, plate_info, lab, conf=conf)
            full_dir = os.path.join(args.eval_dir, "orig", v)
            os.makedirs(full_dir, exist_ok=True)
            out_full = os.path.join(full_dir, f"t{tsec}.jpg")
            _save_jpg(full, out_full)
            rel_full = os.path.relpath(out_full, args.eval_dir).replace("\\", "/")
            
            fb = feedback.get((v, str(idx)), {})
            vv, rr, nn, cp = fb.get("verdict", ""), fb.get("reason", ""), fb.get("note", ""), fb.get("corrected_plate", "")
            done_badge = ' <span class="done">✓已标</span>' if vv else ""
            conf_color = "#16a34a" if conf >= 0.85 else ("#d97706" if conf >= 0.6 else "#dc2626")
            
            crop_html += f"""
            <div class="crop-card" data-video="{v}" data-t="{tsec}" data-idx="{idx}"
                 data-detected="{detected}" data-gt="{','.join(gt_plates)}">
              <img class="zoom" src="{rel}" data-full="{rel_full}" style="width:200px;display:block;cursor:zoom-in;"/>
              <div class="meta">识别 <b class="p-detected">{detected}</b> / GT <b class="p-gt">{','.join(gt_plates)}</b> <span class="conf" style="background:{conf_color}">置信度 {conf:.2f}</span>{done_badge}</div>
              <div class="fb">
                <select class="verdict">
                  <option value="">--判定--</option>
                  <option value="algo_wrong"{' selected' if vv=='algo_wrong' else ''}>算法错</option>
                  <option value="label_wrong"{' selected' if vv=='label_wrong' else ''}>标注错</option>
                  <option value="too_hard"{' selected' if vv=='too_hard' else ''}>难度太大</option>
                  <option value="other"{' selected' if vv=='other' else ''}>其他</option>
                </select>
                <select class="reason">
                  <option value="">--原因--</option>
                  <option value="occlusion"{' selected' if rr=='occlusion' else ''}>遮挡(被其他车辆/物体挡住)</option>
                  <option value="angle"{' selected' if rr=='angle' else ''}>角度问题(侧拍/俯拍太偏)</option>
                  <option value="light"{' selected' if rr=='light' else ''}>光线问题(反光/过曝/过暗)</option>
                  <option value="blur"{' selected' if rr=='blur' else ''}>模糊(运动模糊/失焦)</option>
                  <option value="partial"{' selected' if rr=='partial' else ''}>部分遮挡(只看到部分车牌)</option>
                  <option value="gt_error"{' selected' if rr=='gt_error' else ''}>GT标注错误</option>
                  <option value="other"{' selected' if rr=='other' else ''}>其他</option>
                </select>
                <input class="corrected_plate" value="{cp}" placeholder="正确车牌(如: 京LNE560)"/>
                <input class="note" value="{nn}" placeholder="备注(可选)"/>
                <button class="save">保存</button>
                <span class="status"></span>
              </div>
            </div>"""
        
        gt_count = len(gt_plates)
        mismatch_count = len([r for r in reps if not r.get("matched", True)])
        
        cards.append(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.06);">
          <h2 style="margin:0 0 8px;font-size:18px;color:#0f172a;">{v}</h2>
          <div style="font-size:12px;margin:2px 0;color:#475569;">GT车牌: {', '.join(gt_plates)} · 时长: {total_dur:.0f}s · 待确认: {mismatch_count}帧</div>
          <div style="font-size:12px;margin:10px 0 4px;color:#475569;">关键帧 (优先: 识别错误 &gt; 低置信度 &gt; 新车牌):</div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;">{crop_html or '<span style="color:#94a3b8;">无 (全对)</span>'}</div>
        </div>""")
    
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>车牌识别误差确认画廊</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f1f5f9;margin:0;padding:0 20px 40px;}
#toolbar{position:sticky;top:0;background:#0f172a;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:10;margin:0 -20px 18px;border-radius:0 0 8px 8px;font-size:13px;}
#toolbar b{font-size:15px;}
#toolbar .hint{opacity:.72;font-weight:400;}
#export{margin-left:auto;background:#fff;color:#0f172a;border:none;border-radius:4px;padding:5px 12px;cursor:pointer;font-size:12px;}
h1{color:#0f172a;margin:8px 0;}
.intro{color:#475569;font-size:13px;margin:0 0 14px;}
.crop-card{border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;width:212px;background:#fff;}
.crop-card.saved{box-shadow:0 0 0 2px #22c55e inset;}
.meta{font-size:11px;padding:4px 6px;color:#334155;}
.p-detected{color:#2563eb;font-weight:700;} .p-gt{color:#16a34a;font-weight:700;}
.done{color:#16a34a;font-size:10px;margin-left:4px;}
.conf{display:inline-block;color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:4px;font-weight:700;}
.fb{padding:6px;display:flex;flex-direction:column;gap:4px;font-size:11px;background:#f8fafc;}
.fb select,.fb input{font-size:11px;padding:3px;border:1px solid #cbd5e1;border-radius:4px;}
.fb .corrected_plate{width:140px;font-weight:700;text-transform:uppercase;}
.fb .save{background:#0f172a;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;align-self:flex-start;}
.fb .status{font-size:10px;}
.zoom{cursor:zoom-in;}
.lightbox{position:fixed;inset:0;background:rgba(15,23,42,.9);display:none;align-items:center;justify-content:center;z-index:200;}
.lightbox.show{display:flex;}
.lightbox img{max-width:94vw;max-height:94vh;border:2px solid #fff;border-radius:6px;box-shadow:0 8px 40px rgba(0,0,0,.6);}
.lightbox .hint{position:absolute;bottom:18px;color:#cbd5e1;font-size:12px;}
</style></head><body>
<div id="toolbar">
  <b>车牌识别误差确认画廊</b>
  <span>已保存 <span id="saved">0</span> 帧</span>
  <span class="hint">每张图下方点"保存"即可标注 · 判定=算法错/标注错/难度太大/其他 · 请填写正确车牌用于算法迭代</span>
  <button id="export">导出本地标注(JSON)</button>
</div>
<h1>车牌识别误差确认画廊</h1>
<p class="intro">左=车牌区域特写(<span style="color:#dc2626;font-weight:700;">红框=检测框</span>: 算法识别的车牌位置)。<b>点小图看原始整帧</b>(红框=所有检测到的车牌)。
请判定: <b>算法错</b>(识别结果与真实不符) / <b>标注错</b>(GT标注有误) / <b>难度太大</b>(遮挡/角度/光线等无法识别) / <b>其他</b>;
原因按判定树选: <b>遮挡</b>(被其他车辆/物体挡住) → <b>角度问题</b>(侧拍/俯拍太偏) → <b>光线问题</b>(反光/过曝/过暗) → <b>模糊</b>(运动模糊/失焦) → <b>部分遮挡</b>(只看到部分车牌) / GT标注错误 / 其他。
<b>置信度</b>=算法对识别结果的置信度: <span style="color:#16a34a;font-weight:700;">绿≥0.85</span>高 / <span style="color:#d97706;font-weight:700;">黄0.6–0.85</span>中 / <span style="color:#dc2626;font-weight:700;">红&lt;0.6</span>低(难帧)。
<br/><b>关键帧采样优先级:</b> 识别错误帧(优先) &gt; 低置信度帧(conf&lt;0.6) &gt; 首次识别到的新车牌。</p>
{CARDS}
<div id="lb" class="lightbox"><img alt="zoom"/><div class="hint">点击任意处关闭</div></div>
<script>
async function postFeedback(p){
  try{
    const r = await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    if(r.ok) return 'ok';
  }catch(e){}
  localStorage.setItem('fb_'+p.video+'_'+p.idx, JSON.stringify(p));
  return 'local';
}
function updateCount(){ document.getElementById('saved').textContent = document.querySelectorAll('.crop-card.saved').length; }
document.querySelectorAll('.crop-card .save').forEach(btn=>{
  btn.addEventListener('click', async ()=>{
    const card = btn.closest('.crop-card');
    const p = { video:card.dataset.video, t:card.dataset.t, idx:card.dataset.idx,
                detected:card.dataset.detected, gt:card.dataset.gt,
                verdict:card.querySelector('.verdict').value,
                reason:card.querySelector('.reason').value,
                corrected_plate:card.querySelector('.corrected_plate').value.toUpperCase(),
                note:card.querySelector('.note').value };
    const st = card.querySelector('.status');
    if(!p.verdict){ st.textContent='请先选判定'; st.style.color='#dc2626'; return; }
    st.textContent='保存中...'; st.style.color='#64748b';
    const res = await postFeedback(p);
    if(res==='ok'){ st.textContent='已保存 ✓'; st.style.color='#16a34a'; card.classList.add('saved');
      if(!card.querySelector('.done')) card.querySelector('.meta').insertAdjacentHTML('beforeend',' <span class="done">✓已标</span>'); }
    else { st.textContent='已存本地(无服务)'; st.style.color='#d97706'; card.classList.add('saved'); }
    const nxt = card.nextElementSibling;
    if(nxt && nxt.classList.contains('crop-card')){
      nxt.querySelector('.verdict').value = p.verdict;
      nxt.querySelector('.reason').value = p.reason;
    }
    updateCount();
  });
});
document.getElementById('export').addEventListener('click', ()=>{
  const items=[]; for(let i=0;i<localStorage.length;i++){const k=localStorage.key(i); if(k&&k.startsWith('fb_')) items.push(JSON.parse(localStorage.getItem(k)));}
  const blob=new Blob([JSON.stringify(items,null,2)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='plate_feedback_local.json'; a.click();
});
updateCount();
const lb=document.getElementById('lb'), lbImg=lb.querySelector('img');
document.querySelectorAll('img.zoom').forEach(im=>{
  im.addEventListener('click',()=>{ lbImg.src=im.dataset.full; lb.classList.add('show'); });
});
lb.addEventListener('click',()=>lb.classList.remove('show'));
</script>
</body></html>"""
    
    html = html.replace("{CARDS}", "".join(cards))
    out_html = os.path.join(args.eval_dir, "gallery.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] 画廊 -> {out_html}")


if __name__ == "__main__":
    main()