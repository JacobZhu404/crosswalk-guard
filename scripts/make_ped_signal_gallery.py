"""生成行人信号 crop 人工校验画廊 (HTML, 自包含 base64, 免服务)。

用途: 逐张目视确认 datasets/ped_signal/labels.csv 的弱标签(walk/stand/off),
修正错误标签并标记 verified=1, 供 train_ped_signal.py (--verified-only) 训练。
弱标签来自 prior ROI 直抽 + GT 段灯态, prior 位置标错/灯遮挡时弱标签会错, 需人工把关。

工作流:
  1. python scripts/make_ped_signal_gallery.py            # 生成 gallery.html
  2. 浏览器打开, 逐张选正确标签(walk/stand/off) + 保存(本地存 JSON, 可导出)
  3. 导出 ped_signal_feedback.json
  4. python scripts/apply_ped_signal_feedback.py          # 把反馈合并回 labels.csv (verified=1)

用法:
  python scripts/make_ped_signal_gallery.py
  python scripts/make_ped_signal_gallery.py --videos 违章02 --max-per-video 200
"""
import os
import sys
import csv
import json
import base64
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.data_pipeline.ped_signal_dataset import load_labeled_crops
from redlight.infrastructure.image_utils import robust_imread


def img_to_b64(path):
    img = robust_imread(path)
    if img is None:
        return None
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser(description="生成行人信号 crop 校验画廊")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "ped_signal", "labels.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "output", "ped_signal_gallery", "gallery.html"))
    ap.add_argument("--videos", nargs="*", default=None, help="限定视频(默认全部)")
    ap.add_argument("--max-per-video", type=int, default=None, help="每视频最多展示N张(默认全部)")
    ap.add_argument("--show-off", action="store_true", default=True,
                     help="含 off 背景样本(默认含; --no-show-off 排除以只校验正样本)")
    args = ap.parse_args()

    rows = load_labeled_crops(args.labels)
    if args.videos:
        rows = [r for r in rows if r["video"] in args.videos]
    if not args.show_off:
        rows = [r for r in rows if r["label"] != "off"]
    # 按视频+时间排序, 每视频限N张
    rows.sort(key=lambda r: (r["video"], float(r["frame_ts"])))
    if args.max_per_video:
        from collections import defaultdict
        per = defaultdict(list)
        for r in rows:
            if len(per[r["video"]]) < args.max_per_video:
                per[r["video"]].append(r)
        rows = [r for v in sorted(per) for r in per[v]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 按视频分组生成卡片
    cards_by_video = {}
    skipped = 0
    for r in rows:
        b64 = img_to_b64(r["crop_path"])
        if b64 is None:
            skipped += 1
            continue
        cur_label = r["label"]
        conf_color = {"walk": "#16a34a", "stand": "#d97706", "off": "#64748b"}.get(cur_label, "#475569")
        card = f"""
        <div class="crop-card" data-cp="{r['crop_path']}" data-video="{r['video']}" data-t="{r['frame_ts']}" data-cur="{cur_label}">
          <img src="data:image/jpeg;base64,{b64}"/>
          <div class="meta">{r['video']} t={r['frame_ts']}s <span class="cur" style="background:{conf_color}">{cur_label}</span></div>
          <div class="fb">
            <select class="verdict">
              <option value="">--改标--</option>
              <option value="walk"{' selected' if cur_label=='walk' else ''}>walk(绿灯/过街)</option>
              <option value="stand"{' selected' if cur_label=='stand' else ''}>stand(红灯/站立)</option>
              <option value="off"{' selected' if cur_label=='off' else ''}>off(非信号/背景)</option>
              <option value="delete">删除(难判/废)</option>
            </select>
            <button class="save">保存</button>
            <span class="status"></span>
          </div>
        </div>"""
        cards_by_video.setdefault(r["video"], []).append(card)

    cards_html = ""
    for v in sorted(cards_by_video):
        cnt = len(cards_by_video[v])
        cards_html += f'<h2>{v} <span class="cnt">({cnt}张)</span></h2><div class="grid">{"".join(cards_by_video[v])}</div>'

    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>行人信号 crop 校验画廊</title>
<style>
body{font-family:-apple-system,sans-serif;background:#f1f5f9;margin:0;padding:0 20px 40px;}
#tb{position:sticky;top:0;background:#0f172a;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:10;margin:0 -20px 18px;font-size:13px;}
#tb .hint{opacity:.72;font-weight:400;}
#export{margin-left:auto;background:#fff;color:#0f172a;border:none;border-radius:4px;padding:5px 12px;cursor:pointer;font-size:12px;}
h1{color:#0f172a;margin:8px 0 4px;} .intro{color:#475569;font-size:13px;margin:0 0 14px;}
h2{color:#0f172a;font-size:16px;margin:18px 0 6px;} .cnt{color:#64748b;font-weight:400;font-size:13px;}
.grid{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;}
.crop-card{border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;width:200px;background:#fff;}
.crop-card.saved{box-shadow:0 0 0 2px #22c55e inset;}
.crop-card img{width:200px;height:200px;object-fit:cover;display:block;background:#000;}
.meta{font-size:11px;padding:4px 6px;color:#334155;}
.cur{display:inline-block;color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:4px;font-weight:700;}
.fb{padding:6px;display:flex;gap:4px;align-items:center;font-size:11px;background:#f8fafc;}
.fb select{font-size:11px;padding:3px;border:1px solid #cbd5e1;border-radius:4px;}
.fb button{background:#0f172a;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px;}
.fb .status{font-size:10px;}
</style></head><body>
<div id="tb"><b>行人信号 crop 校验</b><span id="cnt">已保存 0</span><span class="hint">每张选正确标签→保存; 全部标完点"导出反馈JSON"; 再跑 apply_ped_signal_feedback.py 合并回 labels.csv</span><button id="export">导出反馈JSON</button></div>
<h1>行人信号 crop 人工校验</h1>
<p class="intro">每张小图是 prior ROI 抠出的候选信号灯(或 prior 外背景=off)。弱标签来自 prior+GT段, 可能错(prior标错/灯遮挡)。<b>选正确标签→保存</b>: walk=绿灯过街 / stand=红灯站立 / off=非信号背景 / 删除=难判废图。<b>默认选中的是当前弱标签</b>, 对就保存, 错就改。</p>
""" + cards_html + """
<script>
const saved=new Map(); // crop_path -> {label}
function upd(){document.getElementById('cnt').textContent='已保存 '+saved.size;}
document.querySelectorAll('.crop-card .save').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const card=btn.closest('.crop-card');
    const v=card.querySelector('.verdict').value;
    const st=card.querySelector('.status');
    if(!v){st.textContent='请先选标签';st.style.color='#dc2626';return;}
    saved.set(card.dataset.cp,{label:v,video:card.dataset.video,t:card.dataset.t,cur:card.dataset.cur});
    st.textContent='已保存 ✓';st.style.color='#16a34a';card.classList.add('saved');upd();
  });
});
document.getElementById('export').addEventListener('click',()=>{
  const items=[];for(const [k,v] of saved) items.push({crop_path:k,...v});
  const blob=new Blob([JSON.stringify(items,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='ped_signal_feedback.json';a.click();
});
</script></body></html>"""

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] 画廊 -> {args.out}")
    print(f"  crop 数: {len(rows)} (跳过 {skipped} 张读图失败)")
    print(f"  浏览器打开, 逐张校验, 导出 ped_signal_feedback.json")
    print(f"  再跑: python scripts/apply_ped_signal_feedback.py")


if __name__ == "__main__":
    main()
