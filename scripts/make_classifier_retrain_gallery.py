"""生成 light-state 重训数据集人工抽检画廊 (HTML, 自包含 base64, 免服务)。

用途: 抽检 datasets/classifier_retrain/labels.csv 的弱标签(walk/stand/off),
修正错误标签并标记 verified=1, 供 train_ped_signal.py (--verified-only) 训练。

分层抽样(降 Jacob 工作量, cc 决策): 默认 strategy=sample:
  - 100% impostor(off, source∈impostor/impostor_outside)
  - 100% 06/07 暗绿(walk) + 100% 04 短绿(walk)
  - 100% 段边界帧(最易误标 impostor 处)
  - 其余(prior_roi 真信号 + 背景正则)随机抽 20%
全量模式 --all 展示全部。

工作流:
  1. PYTHONPATH=src ./.venv/bin/python scripts/make_classifier_retrain_gallery.py
  2. 浏览器打开 gallery.html, 逐张选正确标签 + 保存
  3. 导出 classifier_retrain_feedback.json
  4. PYTHONPATH=src ./.venv/bin/python scripts/apply_classifier_retrain_feedback.py

用法:
  scripts/make_classifier_retrain_gallery.py [--strategy sample|all] [--max-per-video N]
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
from redlight.evaluation import gt_lookup

LABELS = ["walk", "stand", "off"]
CONF_COLOR = {"walk": "#16a34a", "stand": "#d97706", "off": "#64748b"}
DARK_GREEN_VIDEOS = {"违章06", "违章07"}
SHORT_GREEN_VIDEO = "违章04"
BOUNDARY_SEC = 2.0


def _boundary_ts_by_video(light_states_csv):
    """返回 {video: [boundary_ts, ...]}, 段边界且涉及 green<->非green 跳变。"""
    segs = gt_lookup.load_light_state_csv(light_states_csv)
    out = {}
    for v, seglist in segs.items():
        bts = []
        for i, (a, b, st, conf) in enumerate(seglist):
            prev = seglist[i - 1][2] if i > 0 else None
            nxt = seglist[i + 1][2] if i + 1 < len(seglist) else None
            involves_green = (st == "green") or (prev == "green") or (nxt == "green")
            if involves_green:
                bts.append(a)
                bts.append(b)
        out[v] = bts
    return out


def _is_boundary(video, ts, boundary_map):
    for b in boundary_map.get(video, []):
        if abs(ts - b) <= BOUNDARY_SEC:
            return True
    return False


def img_to_b64(path):
    img = robust_imread(path)
    if img is None:
        return None
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def main():
    ap = argparse.ArgumentParser(description="生成 light-state 重训数据集抽检画廊")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "classifier_retrain", "labels.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "classifier_retrain", "classifier_retrain_gallery.html"))
    ap.add_argument("--light-states", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--strategy", default="sample", choices=["sample", "all"])
    ap.add_argument("--sample-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    rows = load_labeled_crops(args.labels)
    # load_labeled_crops 把 crop_path 解析成绝对路径(便于读图); 但 labels.csv 存相对路径,
    # feedback 必须回相对路径才能被 apply_classifier_retrain_feedback.py 按 crop_path 匹配。
    dataset_root = os.path.dirname(os.path.abspath(args.labels))
    boundary_map = _boundary_ts_by_video(args.light_states)

    if args.strategy == "all":
        sel = list(rows)
    else:
        impostor, dark_green, short_green, boundary, rest = [], [], [], [], []
        for r in rows:
            v, ts, src, lab = r["video"], float(r["frame_ts"]), r["source"], r["label"]
            if src in ("impostor", "impostor_outside"):
                impostor.append(r)
            elif v in DARK_GREEN_VIDEOS and lab == "walk":
                dark_green.append(r)
            elif v == SHORT_GREEN_VIDEO and lab == "walk":
                short_green.append(r)
            elif _is_boundary(v, ts, boundary_map):
                boundary.append(r)
            else:
                rest.append(r)
        sampled_rest = random.sample(rest, max(0, int(len(rest) * args.sample_frac))) if rest else []
        sel = impostor + dark_green + short_green + boundary + sampled_rest

    sel.sort(key=lambda r: (r["video"], float(r["frame_ts"])))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cards_by_video = {}
    skipped = 0
    for r in sel:
        b64 = img_to_b64(r["crop_path"])
        if b64 is None:
            skipped += 1
            continue
        rel_cp = os.path.relpath(r["crop_path"], dataset_root)  # 回相对路径, 供 apply 匹配
        cur = r["label"]
        color = CONF_COLOR.get(cur, "#475569")
        tag = "边界" if _is_boundary(r["video"], float(r["frame_ts"]), boundary_map) else ""
        card = f"""
        <div class="crop-card" data-cp="{rel_cp}" data-video="{r['video']}" data-t="{r['frame_ts']}" data-cur="{cur}">
          <img src="data:image/jpeg;base64,{b64}"/>
          <div class="meta">{r['video']} t={r['frame_ts']}s <span class="cur" style="background:{color}">{cur}</span>{('<span class="bd">'+tag+'</span>') if tag else ''}</div>
          <div class="fb">
            <select class="verdict">
              <option value="">--改标--</option>
              <option value="walk"{' selected' if cur=='walk' else ''}>walk(绿灯/过街)</option>
              <option value="stand"{' selected' if cur=='stand' else ''}>stand(红灯/站立)</option>
              <option value="off"{' selected' if cur=='off' else ''}>off(非信号/impostor)</option>
              <option value="delete">删除(难判/废)</option>
            </select>
            <button class="save">保存</button>
            <span class="status"></span>
          </div>
        </div>"""
        cards_by_video.setdefault(r["video"], []).append(card)

    cards_html = ""
    for v in sorted(cards_by_video):
        cards_html += f'<h2>{v} <span class="cnt">({len(cards_by_video[v])}张)</span></h2><div class="grid">{"".join(cards_by_video[v])}</div>'

    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>light-state 重训数据集抽检画廊</title>
<style>
body{font-family:-apple-system,sans-serif;background:#f1f5f9;margin:0;padding:0 20px 40px;}
#tb{position:sticky;top:0;background:#0f172a;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:10;margin:0 -20px 18px;font-size:13px;}
#export{margin-left:auto;background:#fff;color:#0f172a;border:none;border-radius:4px;padding:5px 12px;cursor:pointer;font-size:12px;}
h1{color:#0f172a;margin:8px 0 4px;} .intro{color:#475569;font-size:13px;margin:0 0 14px;}
h2{color:#0f172a;font-size:16px;margin:18px 0 6px;} .cnt{color:#64748b;font-weight:400;font-size:13px;}
.grid{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;}
.crop-card{border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;width:200px;background:#fff;}
.crop-card.saved{box-shadow:0 0 0 2px #22c55e inset;}
.crop-card img{width:200px;height:200px;object-fit:cover;display:block;background:#000;}
.meta{font-size:11px;padding:4px 6px;color:#334155;}
.cur{display:inline-block;color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:4px;font-weight:700;}
.bd{display:inline-block;color:#fff;font-size:9px;padding:1px 5px;border-radius:6px;margin-left:4px;background:#9333ea;}
.fb{padding:6px;display:flex;gap:4px;align-items:center;font-size:11px;background:#f8fafc;}
.fb select{font-size:11px;padding:3px;border:1px solid #cbd5e1;border-radius:4px;}
.fb button{background:#0f172a;color:#fff;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px;}
.fb .status{font-size:10px;}
</style></head><body>
<div id="tb"><b>light-state 重训数据集抽检</b><span id="cnt">已保存 0</span><span class="hint">分层抽样: impostor+暗绿+04短绿+段边界帧全核, 其余随机20%。选正确标签→保存; 全标完点"导出反馈JSON"; 再跑 apply_classifier_retrain_feedback.py</span><button id="export">导出反馈JSON</button></div>
<h1>light-state 重训数据集抽检</h1>
<p class="intro">每张小图是 prior ROI 或信号外绿斑抠出的候选。弱标签=GT段+引擎绿读, 可能错(假绿帧藏暗真绿 / 真绿ROI偏出信号)。<b>选正确标签→保存</b>: walk=真绿过街 / stand=真红站立 / off=非信号(背心/植物/反射/车灯) / 删除=难判废图。<b>默认选中当前弱标签</b>, 对就保存, 错就改。</p>
""" + cards_html + """
<script>
const saved=new Map();
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
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='classifier_retrain_feedback.json';a.click();
});
</script></body></html>"""

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] 画廊 -> {args.out}")
    print(f"  展示 {len(sel)} 张 (跳过 {skipped} 张读图失败); 全量 {len(rows)} 张")
    print(f"  分层: impostor+暗绿+04短绿+段边界帧全核 + 其余随机{args.sample_frac:.0%}")
    print(f"  浏览器打开, 逐张校验, 导出 classifier_retrain_feedback.json")
    print(f"  再跑: PYTHONPATH=src ./.venv/bin/python scripts/apply_classifier_retrain_feedback.py")


if __name__ == "__main__":
    main()
