"""Step2 信号过滤(几何保持 cc(B)): 丢信号占比<阈值的 walk/stand 正样本, 只删行不改几何。

设计(逐条承接 cc review 4 条修正):
  - 读 labels.csv 用**纯 csv.DictReader**, 不走 load_labeled_crops(它会把相对 crop_path
    绝对化, 写回破跨机可移植) -> 保相对路径(Bug1 修正)。
  - 写盘用**输入真实表头**(含 fi 的 11 列), 不由 LABELS_HEADER 兜底(它缺 fi) -> 不丢列(Bug2 修正)。
  - 写盘 open(tmp,"w") 截断 + os.replace 原子覆盖 -> 重跑不追加重复行(Bug3 修正)。
  - off 恒保留(负类不要求有信号); delete/空 label 恒丢弃; walk/stand 按生产同款 HSV 阈值算
    信号占比, <threshold 丢弃; 读图失败记 unreadable 丢弃。
  - 产出 labels.filtered.csv(可训练) + filter_manifest.json(清单) + filter_deleted_gallery.html
    (训练前 GO/NO-GO 有效性门: Jacob 审被删的是错框而非暗绿真值)。

纯函数 filter_low_signal_rows 可单测(见 tests/test_filter_low_signal_rows.py)。

用法:
  PYTHONPATH=src:scripts ./.venv/bin/python scripts/filter_low_signal_rows.py \
    --labels datasets/classifier_retrain/labels.csv \
    --out datasets/classifier_retrain/labels.filtered.csv \
    --threshold 0.005
"""
import os
import sys
import csv
import json
import random
import argparse

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analyze_signal_presence import signal_ratio  # 复用已 TDD 的 255-bug 修复版

DEFAULT_THRESHOLD = 0.005  # cc(B): 信号占比 < 0.5% 丢弃(几何保持, 不重挖)

# GO/NO-GO 画廊: 风险视频(暗绿/弱信号高浓度)全量 + 其余随机采样
RISK_VIDEO_KEYS = ("04", "07")
GALLERY_SAMPLE_REST = 60


# ------------------------------------------------------------------ 纯函数
def _is_deleted(r):
    """与 exclude_deleted 口径一致: 空/delete 行恒丢弃。"""
    return r.get("label") in (None, "", "delete")


def read_rows_relative(csv_path):
    """纯 csv.DictReader 读 labels.csv -> (rows, fieldnames)。

    ⚠️ 不走 load_labeled_crops(它会把相对 crop_path 绝对化, 写回破跨机可移植)。
    这里保相对路径, 仅训练侧 load_labeled_crops 在消费时再解析绝对。
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    return rows, fieldnames


def filter_low_signal_rows(rows, threshold, base_dir, signal_fn=signal_ratio):
    """纯函数: 过滤低信号 walk/stand 正样本。

    rows: csv.DictReader 产物(crop_path 为相对路径, 未绝对化)。
    base_dir: labels.csv 所在目录, 用于把相对 crop_path 解析成绝对路径读图(仅读图用, 不改 row)。
    返回 (kept, dropped):
      - kept: 原样 row(相对 crop_path 不变), 可直接写回 CSV。
      - dropped: {**row, "_signal_ratio": float|None, "_reason": str}。
    规则:
      - label==off            -> 恒保留(负类不要求有信号)。
      - label 空/delete       -> 恒丢弃。
      - walk/stand            -> 读图算 signal_ratio; None(读失败/空图)->unreadable 丢弃;
                                 <threshold->low_signal 丢弃; 否则保留。
      - 其它 label            -> 保守保留(避免误删未知类)。
    """
    kept, dropped = [], []
    for r in rows:
        lab = r.get("label")
        if _is_deleted(r):
            dropped.append({**r, "_signal_ratio": None, "_reason": "deleted_or_empty"})
            continue
        if lab == "off":
            kept.append(r)
            continue
        if lab not in ("walk", "stand"):
            kept.append(r)  # 未知类保守保留
            continue
        rel = r.get("crop_path", "")
        abs_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        img = cv2.imread(abs_path) if abs_path else None
        ratio = signal_fn(img, "green" if lab == "walk" else "red")
        if ratio is None:
            dropped.append({**r, "_signal_ratio": None, "_reason": "unreadable"})
        elif ratio < threshold:
            dropped.append({**r, "_signal_ratio": ratio, "_reason": "low_signal"})
        else:
            kept.append(r)
    return kept, dropped


# ------------------------------------------------------------------ 写盘(均原子)
def write_filtered_csv(kept, fieldnames, out_csv):
    """写 kept 行到 out_csv, 保相对 crop_path + 输入真实表头。

    ⚠️ open(tmp,"w") 截断(tmp 每次全新, 不追加) + os.replace 原子覆盖 -> 重跑不产生重复行。
    fieldnames 须为输入真实表头(含 fi 等 11 列), 不由 LABELS_HEADER 兜底(它缺 fi)。
    """
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_csv + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    os.replace(tmp, out_csv)  # 原子 + 截断(覆盖旧文件)
    return out_csv


def write_manifest(kept, dropped, fieldnames, threshold, out_json):
    """写过滤清单(原子): 计数 + 分视频 dropped + dropped 明细(含 signal_ratio/reason)。"""
    dropped_by_video = {}
    detail = []
    for d in dropped:
        v = d.get("video", "?")
        reason = d.get("_reason", "?")
        lab = d.get("label", "?")
        dropped_by_video.setdefault(v, {"walk": 0, "stand": 0, "off": 0, "other": 0})
        if lab in ("walk", "stand", "off"):
            dropped_by_video[v][lab] += 1
        else:
            dropped_by_video[v]["other"] += 1
        detail.append({
            "crop_path": d.get("crop_path"),
            "video": v,
            "label": lab,
            "fi": d.get("fi"),
            "signal_ratio": d.get("_signal_ratio"),
            "reason": reason,
        })
    manifest = {
        "threshold": threshold,
        "fieldnames": fieldnames,
        "total_in": len(kept) + len(dropped),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_by_video": dropped_by_video,
        "dropped_detail": detail,
    }
    out_dir = os.path.dirname(os.path.abspath(out_json))
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_json)
    return out_json


_GALLERY_TMPL = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Step2 被删集画廊 (GO/NO-GO 有效性门)</title>
<style>
 body{font-family:-apple-system,Segoe UI,sans-serif;margin:16px;background:#fafafa;color:#222}
 h1{font-size:18px} .meta{color:#666;margin-bottom:12px}
 .grid{display:flex;flex-wrap:wrap;gap:8px}
 .t{width:200px;border:1px solid #ddd;border-radius:6px;overflow:hidden;background:#fff}
 .t img{width:200px;height:120px;object-fit:cover;display:block;background:#000}
 .c{font-size:11px;padding:4px 6px;line-height:1.35}
 .risk{color:#b00;font-weight:600}
</style></head><body>
<h1>Step2 被删集画廊 — 训练前 GO/NO-GO 有效性门</h1>
<div class="meta">展示 __N__ / __TOTAL__ 张被删 crop(风险视频 04/07 全量 + 其余随机采样 __REST__)。
请确认被删的是<b>错框/无信号</b>(删得对), 而非<b>暗绿/暗红真值</b>(删错了 → NO-GO, 抬高阈值)。</div>
<div class="grid">
__TILES__
</div></body></html>"""


def build_deleted_gallery(dropped, base_dir, out_html, seed=0, sample_rest=GALLERY_SAMPLE_REST,
                          risk_keys=RISK_VIDEO_KEYS):
    """GO/NO-GO 有效性门产物: 被删 crop 画廊。风险视频全量 + 其余随机采样。

    红框标注 = 风险视频(04/07, 暗绿/弱信号高浓度); 其余为随机采样。
    每张标注 video/label/fi/signal_ratio/reason, 供 Jacob 判断"删的是错框还是暗绿真值"。
    """
    risk = [d for d in dropped if any(k in str(d.get("video", "")) for k in risk_keys)]
    rest = [d for d in dropped if d not in risk]
    rng = random.Random(seed)
    rest_sample = rng.sample(rest, min(sample_rest, len(rest)))
    sel = risk + rest_sample
    tiles = []
    for d in sel:
        rel = d.get("crop_path", "")
        abs_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        ratio = d.get("_signal_ratio")
        ratio_s = "None" if ratio is None else f"{ratio:.4f}"
        is_risk = any(k in str(d.get("video", "")) for k in risk_keys)
        cls = "t risk" if is_risk else "t"
        tiles.append(
            f'<div class="{cls}"><img src="file://{abs_path}">'
            f'<div class="c">{d.get("video", "?")} / {d.get("label", "?")} / fi={d.get("fi", "?")}<br>'
            f'signal={ratio_s} / {d.get("_reason", "?")}</div></div>'
        )
    html = (_GALLERY_TMPL
            .replace("__N__", str(len(sel)))
            .replace("__TOTAL__", str(len(dropped)))
            .replace("__REST__", str(sample_rest))
            .replace("__TILES__", "\n".join(tiles)))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


# ------------------------------------------------------------------ 入口
def main():
    ap = argparse.ArgumentParser(description="Step2 信号过滤(几何保持 cc(B), 只删行)")
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "classifier_retrain", "labels.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "classifier_retrain", "labels.filtered.csv"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "datasets", "classifier_retrain", "filter_manifest.json"))
    ap.add_argument("--gallery", default=os.path.join(ROOT, "datasets", "classifier_retrain", "filter_deleted_gallery.html"))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="walk/stand 信号占比 < 此值丢弃(默认 0.005 = 0.5%, cc(B))")
    args = ap.parse_args()

    rows, fieldnames = read_rows_relative(args.labels)
    base_dir = os.path.dirname(os.path.abspath(args.labels))
    kept, dropped = filter_low_signal_rows(rows, args.threshold, base_dir)

    write_filtered_csv(kept, fieldnames, args.out)
    write_manifest(kept, dropped, fieldnames, args.threshold, args.manifest)
    build_deleted_gallery(dropped, base_dir, args.gallery)

    by_v = {}
    for d in dropped:
        v = d.get("video", "?")
        by_v[v] = by_v.get(v, 0) + 1
    print(f"[filter] threshold={args.threshold}")
    print(f"[filter] 输入 {len(rows)} 行 -> 保留 {len(kept)} / 丢弃 {len(dropped)}")
    print(f"[filter] 各视频丢弃: " + ", ".join(f"{v}={n}" for v, n in sorted(by_v.items())))
    print(f"[filter] -> {args.out}")
    print(f"[filter] manifest -> {args.manifest}")
    print(f"[filter] GO/NO-GO 画廊 -> {args.gallery} (训练前请 Jacob 审被删是否真无信号)")


if __name__ == "__main__":
    main()
