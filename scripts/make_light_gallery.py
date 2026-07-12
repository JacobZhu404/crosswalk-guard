"""生成灯态误差确认画廊 (HTML): 每视频并排 预测/GT 时间线色带 + 代表性误差帧信号灯特写。

用途: 用户滚动逐张判定 mismatch 帧是 "算法错 / 标注错 / 都错 / 其他"。

用法:
  python scripts/make_light_gallery.py
  python scripts/make_light_gallery.py --videos 违章04
"""
import os
import sys
import csv
import json
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np
from redlight.infrastructure.config import load_config
from redlight.models.traffic_light import TrafficLightDetector

STATE_COLOR = {"green": "#22c55e", "red": "#ef4444", "unknown": "#9ca3af", "flashing": "#f59e0b"}


def load_priors(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = (float(v[0]), float(v[1]), int(v[2]) if len(v) > 2 else 160)
        elif isinstance(v, dict):
            out[k] = (float(v["cx"]), float(v["cy"]), int(v.get("roi", 160)))
    return out


def load_pred(pred_csv):
    rows = []
    with open(pred_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_gt(gt_csv):
    g = {}
    with open(gt_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g.setdefault(r["video"], []).append(
                (float(r["start_s"]), float(r["end_s"]), r["state"], r.get("confidence", "confirmed")))
    for v in g:
        g[v].sort(key=lambda x: x[0])
    return g


def load_feedback(path):
    """读取已存的用户反馈, 用于画廊回显. key=(video, t_sec_str)"""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            k = (r.get("video", ""), r.get("t_sec", ""))
            d[k] = {"verdict": r.get("verdict", ""), "reason": r.get("reason", ""),
                    "note": r.get("note", "")}
    return d


def compress(records, key="pred"):
    segs = []
    for r in records:
        st = r[key]
        t = float(r["t_sec"])
        if segs and segs[-1][0] == st:
            segs[-1][2] = t
        else:
            segs.append([st, t, t])
    return segs


def timeline_html(segs, total_dur, label):
    parts = []
    for st, a, b in segs:
        dur = max(b - a, 0.01)
        w = dur / total_dur * 100
        c = STATE_COLOR.get(st, "#64748b")
        parts.append(f'<div style="width:{w:.2f}%;background:{c};" title="{st} {a:.1f}-{b:.1f}s"></div>')
    return (f'<div style="font-size:12px;margin:2px 0;color:#475569;">{label}</div>'
            f'<div style="display:flex;height:22px;border-radius:4px;overflow:hidden;'
            f'border:1px solid #e2e8f0;">{"".join(parts)}</div>')


def _detect_center(fr, det):
    """跑检测器取算法认定的灯中心(归一化 cx,cy). 失败返回 None."""
    if det is None:
        return None
    try:
        res = det.detect(fr)
        t = res.get("track") or res.get("anchor")
        if t and "cx" in t:
            return (float(t["cx"]), float(t["cy"]))
    except Exception:
        return None
    return None


def _save_jpg(img, out_path):
    """字节写盘, 规避 OpenCV 中文路径 imwrite 静默失败."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return False
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    return True


def annotate_crop(fr, prior, det, label_txt):
    """信号灯 ROI 特写(小图): 浅蓝框=搜索区边界(明显内缩), 黄圈=读取点."""
    H, W = fr.shape[:2]
    if prior is not None:
        cx, cy, roi = prior
        ax, ay = int(cx * W), int(cy * H)
        x1, y1 = max(0, ax - roi), max(0, ay - roi)
        x2, y2 = min(W, ax + roi), min(H, ay + roi)
    else:
        x1, y1, x2, y2 = 0, 0, W, H
    crop = fr[y1:y2, x1:x2].copy()
    Hc, Wc = crop.shape[:2]
    if Hc < 4 or Wc < 4:
        return crop
    # 浅蓝框 = 搜索区(ROI)边界, 明显内缩可见(不再贴边像图片边框)
    cv2.rectangle(crop, (6, 6), (Wc - 7, Hc - 7), (255, 178, 102), 3)
    # 黄圈 = 算法读取颜色的中心点
    c = _detect_center(fr, det)
    if c is None:
        c = (cx, cy) if prior is not None else (0.5, 0.5)
    ccx = max(0, min(Wc - 1, int(c[0] * W - x1)))
    ccy = max(0, min(Hc - 1, int(c[1] * H - y1)))
    cv2.circle(crop, (ccx, ccy), 12, (0, 255, 255), 2)
    cv2.drawMarker(crop, (ccx, ccy), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
    # 顶部黑条 + 文字
    cv2.rectangle(crop, (0, 0), (Wc - 1, 28), (0, 0, 0), -1)
    cv2.putText(crop, label_txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return crop


def annotate_full(fr, prior, det, label_txt):
    """原始整帧(放大图): 蓝框=搜索区ROI, 黄圈=读取点 — 给标注者看"算法盯哪".

    这才是用户想要的"原始图片 + 识别过程"; 点小图即弹出此图.
    """
    H, W = fr.shape[:2]
    out = fr.copy()
    if prior is not None:
        cx, cy, roi = prior
        ax, ay = int(cx * W), int(cy * H)
        x1, y1 = max(0, ax - roi), max(0, ay - roi)
        x2, y2 = min(W, ax + roi), min(H, ay + roi)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 178, 102), 3)
    c = _detect_center(fr, det)
    if c is None:
        c = (cx, cy) if prior is not None else (0.5, 0.5)
    ccx = max(0, min(W - 1, int(c[0] * W)))
    ccy = max(0, min(H - 1, int(c[1] * H)))
    cv2.circle(out, (ccx, ccy), 18, (0, 255, 255), 3)
    cv2.drawMarker(out, (ccx, ccy), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
    cv2.rectangle(out, (0, 0), (W - 1, 30), (0, 0, 0), -1)
    cv2.putText(out, label_txt + "   蓝框=搜索区ROI   黄圈=读取点", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "output", "light_eval"))
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--priors", default=os.path.join(ROOT, "configs", "light_priors.json"))
    ap.add_argument("--feedback", default=os.path.join(ROOT, "data", "output", "annotated", "light_feedback.csv"))
    ap.add_argument("--max-crops", type=int, default=10)
    args = ap.parse_args()

    priors = load_priors(args.priors)
    gt = load_gt(args.gt)
    feedback = load_feedback(args.feedback)
    pred_files = sorted(glob.glob(os.path.join(args.eval_dir, "pred_*.csv")))
    videos = args.videos or [os.path.basename(p)[5:-4] for p in pred_files]

    cards = []
    for v in videos:
        pred_csv = os.path.join(args.eval_dir, f"pred_{v}.csv")
        if not os.path.exists(pred_csv):
            continue
        preds = load_pred(pred_csv)
        if not preds:
            continue
        total_dur = float(preds[-1]["t_sec"])
        pred_segs = compress(preds, "pred")
        gt_segs = compress([{"pred": s[2], "t_sec": s[0]} for s in gt.get(v, [])]
                            + [{"pred": gt[v][-1][2], "t_sec": total_dur}], "pred") if v in gt else []
        # 代表性 mismatch 帧: 优先 confirmed GT; 无 confirmed(如 04 tentative)则退回用全部 GT
        mm = [p for p in preds if p["pred"] != p["gt"] and p["gt_conf"] == "confirmed"]
        if not mm:
            mm = [p for p in preds if p["pred"] != p["gt"]]
        mm.sort(key=lambda p: float(p["t_sec"]))
        step = max(1, len(mm) // args.max_crops)
        reps = mm[::step][:args.max_crops]

        # 每视频建一个检测器, 用于裁图上画"算法读取中心点"黄圈
        det = None
        pv = priors.get(v)
        if pv is not None:
            try:
                cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
                det = TrafficLightDetector(cfg, verbose=False)
                det.signal_prior = (pv[0], pv[1])
                det.prior_roi_px = pv[2]
            except Exception:
                det = None
        crop_html = ""
        crop_dir = os.path.join(args.eval_dir, "crops", v)
        os.makedirs(crop_dir, exist_ok=True)
        for m in reps:
            idx = int(m["frame_idx"])
            fp = os.path.join(args.frames_dir, v, f"frame_{idx:06d}.jpg")
            if not os.path.exists(fp):
                continue
            tsec = f"{float(m['t_sec']):.1f}"
            lab = f"t={tsec}s 预{m['pred']}/GT{m['gt']}"
            with open(fp, "rb") as f:
                b = f.read()
            fr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if fr is None:
                continue
            # 小图: ROI 特写(蓝框搜索区 + 黄圈读取点)
            crop = annotate_crop(fr, pv, det, lab)
            out = os.path.join(crop_dir, f"t{tsec}.jpg")
            if not _save_jpg(crop, out):
                continue
            rel = os.path.relpath(out, args.eval_dir).replace("\\", "/")
            # 大图: 原始整帧 + 算法关注的蓝框/黄圈(点小图弹此图)
            full = annotate_full(fr, pv, det, lab)
            full_dir = os.path.join(args.eval_dir, "orig", v)
            os.makedirs(full_dir, exist_ok=True)
            out_full = os.path.join(full_dir, f"t{tsec}.jpg")
            _save_jpg(full, out_full)
            rel_full = os.path.relpath(out_full, args.eval_dir).replace("\\", "/")
            # 回显已标内容
            fb = feedback.get((v, tsec), {})
            vv, rr, nn = fb.get("verdict", ""), fb.get("reason", ""), fb.get("note", "")
            done_badge = ' <span class="done">✓已标</span>' if vv else ""
            crop_html += f"""
            <div class="crop-card" data-video="{v}" data-t="{tsec}" data-pred="{m['pred']}" data-gt="{m['gt']}" data-idx="{idx}">
              <img class="zoom" src="{rel}" data-full="{rel_full}" style="width:200px;display:block;cursor:zoom-in;"/>
              <div class="meta">预测 <b class="p-{m['pred']}">{m['pred']}</b> / GT <b class="p-{m['gt']}">{m['gt']}</b> (g={m['g_px']},r={m['r_px']}){done_badge}</div>
              <div class="fb">
                <select class="verdict">
                  <option value="">--判定--</option>
                  <option value="algo_wrong"{' selected' if vv=='algo_wrong' else ''}>算法错</option>
                  <option value="label_wrong"{' selected' if vv=='label_wrong' else ''}>标注错</option>
                  <option value="both_wrong"{' selected' if vv=='both_wrong' else ''}>都错</option>
                  <option value="other"{' selected' if vv=='other' else ''}>其他</option>
                </select>
                <select class="reason">
                  <option value="">--原因--</option>
                  <option value="search_area"{' selected' if rr=='search_area' else ''}>搜索区没罩住真信号(蓝框不对)</option>
                  <option value="reading_point"{' selected' if rr=='reading_point' else ''}>读取点没落在真灯上(黄圈不对)</option>
                  <option value="color"{' selected' if rr=='color' else ''}>颜色读错(位置对但色错)</option>
                  <option value="both"{' selected' if rr=='both' else ''}>搜索区+读取点都错</option>
                  <option value="gt_flipped"{' selected' if rr=='gt_flipped' else ''}>GT段边界标反</option>
                  <option value="other"{' selected' if rr=='other' else ''}>其他</option>
                </select>
                <input class="note" value="{nn}" placeholder="备注(可选)"/>
                <button class="save">保存</button>
                <span class="status"></span>
              </div>
            </div>"""

        tl = timeline_html(pred_segs, total_dur, "预测时间线")
        if gt_segs:
            tl += timeline_html(gt_segs, total_dur, "GT 时间线 (标 confirmed)")
        cards.append(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.06);">
          <h2 style="margin:0 0 8px;font-size:18px;color:#0f172a;">{v}</h2>
          {tl}
          <div style="font-size:12px;margin:10px 0 4px;color:#475569;">代表性误差帧 (预测≠GT):</div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;">{crop_html or '<span style="color:#94a3b8;">无 (段内全对)</span>'}</div>
        </div>""")

    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>灯态误差确认画廊</title>
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
.p-green{color:#16a34a;font-weight:700;} .p-red{color:#dc2626;font-weight:700;} .p-unknown{color:#6b7280;}
.done{color:#16a34a;font-size:10px;margin-left:4px;}
.fb{padding:6px;display:flex;flex-direction:column;gap:4px;font-size:11px;background:#f8fafc;}
.fb select,.fb input{font-size:11px;padding:3px;border:1px solid #cbd5e1;border-radius:4px;}
.fb .save{background:#0f172a;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;align-self:flex-start;}
.fb .status{font-size:10px;}
.zoom{cursor:zoom-in;}
.lightbox{position:fixed;inset:0;background:rgba(15,23,42,.9);display:none;align-items:center;justify-content:center;z-index:200;}
.lightbox.show{display:flex;}
.lightbox img{max-width:94vw;max-height:94vh;border:2px solid #fff;border-radius:6px;box-shadow:0 8px 40px rgba(0,0,0,.6);}
.lightbox .hint{position:absolute;bottom:18px;color:#cbd5e1;font-size:12px;}
</style></head><body>
<div id="toolbar">
  <b>灯态误差确认画廊</b>
  <span>已保存 <span id="saved">0</span> 帧</span>
  <span class="hint">每张图下方点"保存"即可标注 · 判定=算法错/标注错/都错/其他 · 原因=搜索区没罩住真信号/读取点没落真灯/颜色读错/GT段边界标反</span>
  <button id="export">导出本地标注(JSON)</button>
</div>
<h1>灯态识别误差确认画廊</h1>
<p class="intro">左=信号灯 ROI 特写(<span style="color:#3b82f6;font-weight:700;">蓝框=搜索区</span>: 算法只在此框内找灯头; <span style="color:#eab308;font-weight:700;">黄圈=读取点</span>: 算法实际取色的中心点)。<b>点小图看原始整帧</b>(蓝框=搜索区, 黄圈=读取点)。
请判定: <b>算法错</b> / <b>标注错</b> / <b>都错</b> / <b>其他</b>;
原因: <b>搜索区没罩住真信号</b>(蓝框不对) / <b>读取点没落在真灯上</b>(黄圈不对) / <b>颜色读错</b>(位置对但色错) / 二者都错 / GT段边界标反 / 其他。时间线: 绿=绿灯, 红=红灯, 灰=unknown。</p>
{CARDS}
<div id="lb" class="lightbox"><img alt="zoom"/><div class="hint">点击任意处关闭</div></div>
<script>
async function postFeedback(p){
  try{
    const r = await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    if(r.ok) return 'ok';
  }catch(e){}
  localStorage.setItem('fb_'+p.video+'_'+p.t, JSON.stringify(p));
  return 'local';
}
function updateCount(){ document.getElementById('saved').textContent = document.querySelectorAll('.crop-card.saved').length; }
document.querySelectorAll('.crop-card .save').forEach(btn=>{
  btn.addEventListener('click', async ()=>{
    const card = btn.closest('.crop-card');
    const p = { video:card.dataset.video, t:card.dataset.t, idx:card.dataset.idx,
                pred:card.dataset.pred, gt:card.dataset.gt,
                verdict:card.querySelector('.verdict').value,
                reason:card.querySelector('.reason').value,
                note:card.querySelector('.note').value };
    const st = card.querySelector('.status');
    if(!p.verdict){ st.textContent='请先选判定'; st.style.color='#dc2626'; return; }
    st.textContent='保存中...'; st.style.color='#64748b';
    const res = await postFeedback(p);
    if(res==='ok'){ st.textContent='已保存 ✓'; st.style.color='#16a34a'; card.classList.add('saved');
      if(!card.querySelector('.done')) card.querySelector('.meta').insertAdjacentHTML('beforeend',' <span class="done">✓已标</span>'); }
    else { st.textContent='已存本地(无服务)'; st.style.color='#d97706'; card.classList.add('saved'); }
    // 批量便利: 把刚保存的判定/原因预填到"下一张"(错误常重复), 用户改完直接点保存即可
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
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='light_feedback_local.json'; a.click();
});
updateCount();
// 点击小图放大
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
