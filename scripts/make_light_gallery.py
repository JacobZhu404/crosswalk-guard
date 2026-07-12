"""生成灯态误差确认画廊 (HTML): 每视频并排 预测/GT 时间线色带 + 代表性误差帧信号灯特写。

用途: 用户滚动逐张判定 mismatch 帧是 "标注错 / 算法错 / 太难放弃"。

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


def crop_signal(frame_path, prior, out_path, label_txt):
    with open(frame_path, "rb") as f:
        b = f.read()
    fr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if fr is None:
        return False
    H, W = fr.shape[:2]
    if prior is None:
        crop = fr.copy()
    else:
        cx, cy, roi = prior
        ax, ay = int(cx * W), int(cy * H)
        x1, y1 = max(0, ax - roi), max(0, ay - roi)
        x2, y2 = min(W, ax + roi), min(H, ay + roi)
        crop = fr[y1:y2, x1:x2]
    # 标注文字
    cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(crop, label_txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # 字节写盘, 规避 OpenCV 中文路径 imwrite 静默失败
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        return False
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "output", "light_eval"))
    ap.add_argument("--frames-dir", default=os.path.join(ROOT, "datasets", "frames"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--priors", default=os.path.join(ROOT, "configs", "light_priors.json"))
    ap.add_argument("--max-crops", type=int, default=10)
    args = ap.parse_args()

    priors = load_priors(args.priors)
    gt = load_gt(args.gt)
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

        crop_html = ""
        crop_dir = os.path.join(args.eval_dir, "crops", v)
        os.makedirs(crop_dir, exist_ok=True)
        for m in reps:
            idx = int(m["frame_idx"])
            fp = os.path.join(args.frames_dir, v, f"frame_{idx:06d}.jpg")
            if not os.path.exists(fp):
                continue
            out = os.path.join(crop_dir, f"t{float(m['t_sec']):.1f}.jpg")
            lab = f"t={float(m['t_sec']):.1f}s 预{m['pred']}/GT{m['gt']}"
            if crop_signal(fp, priors.get(v), out, lab):
                rel = os.path.relpath(out, args.eval_dir).replace("\\", "/")
                crop_html += (f'<div style="border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;'
                              f'width:200px;"><img src="{rel}" style="width:200px;display:block;"/>'
                              f'<div style="font-size:11px;padding:3px 6px;color:#334155;">'
                              f'预测 <b>{m["pred"]}</b> / GT <b>{m["gt"]}</b> '
                              f'(g={m["g_px"]},r={m["r_px"]})</div></div>')

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

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>灯态误差确认画廊</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f1f5f9;margin:0;padding:20px;}}
h1{{color:#0f172a;}}</style></head><body>
<h1>灯态识别误差确认画廊</h1>
<p style="color:#475569;">每张裁剪图 = 信号灯 ROI 特写。请判定: <b>标注错</b>(GT 写反了) / <b>算法错</b>(检测器误判) / <b>太难</b>(放弃该帧)。
下方时间线: 绿=绿灯, 红=红灯, 灰=unknown。</p>
{''.join(cards)}
</body></html>"""
    out_html = os.path.join(args.eval_dir, "gallery.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] 画廊 -> {out_html}")


if __name__ == "__main__":
    main()
