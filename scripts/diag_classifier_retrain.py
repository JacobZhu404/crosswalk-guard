"""Phase B gate 诊断:重跑可行性诊断, 模型指向 ped_signal_v2.pt, 数据集指向 classifier_retrain。

复用:
  - diag_classifier_feasibility.scan_video (端到端产绿区判别, 走 observe() 从 signal_prior 的
    prior ROI 裁图 —— 与生产门控插入点/训练分布一致; cc 已核 scan_video:83-94)
  - ped_signal_dataset.load_labeled_crops + SignalStateClassifier.classify (域混淆矩阵)

新增(旧脚本没有):
  - 重指向 v2 模型 + 新数据集
  - train/val 分组报告(07 在 val 诚实泛化, 06 在 train 弱证据须分列)
  - 04 短暗绿探针窗口(--probe)
  - Gate A 加 walk->off 误判率(过拒真绿哨兵)
  - impostor_outside 消融对比(去 outside 重训版需另跑, 本脚本只对比两版 JSON)
  - 四关阈值判定函数(可单测, 见 tests/test_diag_classifier_retrain.py)

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/diag_classifier_retrain.py
  PYTHONPATH=src ./.venv/bin/python scripts/diag_classifier_retrain.py --probe 违章04:42.0:43.2
  PYTHONPATH=src ./.venv/bin/python scripts/diag_classifier_retrain.py --compare-json <去outside版诊断.json>
"""
import os
import sys
import argparse
import json
import csv
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2
import numpy as np

from diag_classifier_feasibility import scan_video
from redlight.models.signal_state_classifier import SignalStateClassifier, LABELS
from redlight.models.traffic_light import TrafficLightDetector
from redlight.data_pipeline.ped_signal_dataset import load_labeled_crops

MODEL = os.path.join(ROOT, "models", "ped_signal_v2.pt")
INPUT_VIDEO = os.path.join(ROOT, "input_video")
NEG_VIDEOS = {"违章01", "违章10"}          # GT 真负例(固定, 与 diag_classifier_feasibility 一致)
VIDEOS = [f"违章{i:02d}" for i in range(1, 12)]

# 消融触发阈值(§4.1): 去 outside 重训后 07(val) 收率回升 >= 此值 -> 才降权
ABLATION_RECALL_GAIN_PP = 5.0


# ------------------------------------------------------------------ 纯函数(可单测)
def parse_probe_window(s):
    """'违章04:42.0:43.2' -> ('违章04', 42.0, 43.2)。"""
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"--probe 需 video:t0:t1, 收到 {s!r}")
    return parts[0], float(parts[1]), float(parts[2])


def gate_neg_off_ratio(videos, neg_set, thr=0.75):
    """关1: 负例(01/10)拒识率 >= thr。返回 (pass, ratio)。"""
    neg = [r for r in videos if r["video"] in neg_set]
    tot = sum(r["crops"] for r in neg)
    off = sum(r["off"] for r in neg)
    ratio = off / tot if tot else None
    return (ratio is not None and ratio >= thr), ratio


def gate_true_green_recall(videos, pos_set, thr=0.90):
    """关2/关4: 正例(暗绿视频)真绿收率(walk+stand 占比)>= thr。返回 (pass, ratio)。"""
    pos = [r for r in videos if r["video"] in pos_set]
    tot = sum(r["crops"] for r in pos)
    acc = sum(r["walk"] + r["stand"] for r in pos)
    ratio = acc / tot if tot else None
    return (ratio is not None and ratio >= thr), ratio


def gate_probe_window(walk, off, other, thr=0.5):
    """关3: 04 短暗绿窗口内 walk 占比 >= thr。返回 (pass, walk_ratio)。"""
    tot = walk + off + other
    ratio = walk / tot if tot else None
    return (ratio is not None and ratio >= thr), ratio


# ------------------------------------------------------------------ 域混淆(新数据集)
def domain_confusion_v2(labels_csv, clf):
    """新数据集 crops 域混淆矩阵 + per-class acc + walk->off 误判率。
    复用 load_labeled_crops + clf.classify(与旧 domain_confusion 同一基元)。"""
    rows = exclude_deleted(load_labeled_crops(labels_csv, verified_only=False))
    cm = {a: {b: 0 for b in LABELS} for a in LABELS}
    for r in rows:
        img = cv2.imread(r["crop_path"])
        if img is None:
            continue
        lab, _ = clf.classify(img)
        if r["label"] in cm and lab in cm[r["label"]]:
            cm[r["label"]][lab] += 1
    n = sum(cm[a][b] for a in LABELS for b in LABELS)
    per = {}
    for a in LABELS:
        tot = sum(cm[a].values())
        per[a] = {"n": tot, "acc": round(cm[a][a] / tot, 3) if tot else 0}
    wtot = sum(cm["walk"].values())
    walk_to_off = (cm["walk"]["off"] / wtot) if wtot else None
    return {"n": n, "confusion": cm, "per_class": per, "walk_to_off": walk_to_off}


def exclude_deleted(rows):
    return [r for r in rows if r.get("label") not in (None, "", "delete")]


# ------------------------------------------------------------------ 04 探针窗口
def scan_window(det, clf, video, t0, t1, sample_step=8):
    """只扫 [t0,t1] 窗口内帧, 统计 walk/off/other(与 scan_video 同裁剪逻辑: prior ROI)。"""
    vp = os.path.join(INPUT_VIDEO, f"{video}.mp4")
    if not os.path.isfile(vp):
        return None
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det.set_video_prior(video)
    rec = {"video": video, "t0": t0, "t1": t1, "walk": 0, "off": 0, "other": 0, "crops": 0}
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = fi / fps
        if t0 <= ts <= t1 and fi % sample_step == 0 and frame is not None:
            h, w = frame.shape[:2]
            res = det.observe(frame)
            if res.get("obs") == "green":
                if det.signal_prior is not None:
                    px, py = det.signal_prior
                    rp = det.prior_roi_px
                    cx, cy = int(px * w), int(py * h)
                    x1, y1 = max(0, cx - rp // 2), max(0, cy - rp // 2)
                    x2, y2 = min(w, cx + rp // 2), min(h, cy + rp // 2)
                    roi = frame[y1:y2, x1:x2]
                else:
                    roi = None
                if roi is not None and roi.size > 0:
                    lab, _ = clf.classify(roi)
                    rec["crops"] += 1
                    rec[lab if lab in ("walk", "off") else "other"] += 1
        fi += 1
    cap.release()
    return rec


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="Phase B gate 可行性诊断 (ped_signal_v2.pt)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--labels", default=os.path.join(ROOT, "datasets", "classifier_retrain", "labels.csv"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "datasets", "classifier_retrain", "manifest.json"))
    ap.add_argument("--probe", default=None, help="04 短暗绿探针: 违章04:42.0:43.2")
    ap.add_argument("--compare-json", default=None, help="去 outside 重训版诊断 JSON, 做消融对比")
    args = ap.parse_args()

    clf = SignalStateClassifier(args.model, verbose=False)
    print(f"[classifier] available={clf.available} path={args.model}")
    if not clf.available:
        print("判别器不可用, 退出")
        return

    cfg = types.SimpleNamespace(
        traffic_light=None,
        models=types.SimpleNamespace(ped_signal_model=args.model),
    )
    det = TrafficLightDetector(cfg, verbose=False)

    out = {"domain": domain_confusion_v2(args.labels, clf), "videos": [], "gates": {}}

    # A) 域混淆 + walk->off 误判率
    d = out["domain"]
    print("\n=== A) 域混淆矩阵 (classifier_retrain crops) ===")
    print(f"  总样本={d['n']}  walk->off 误判率={d['walk_to_off']}")
    for a in LABELS:
        print(f"  {a}: n={d['per_class'][a]['n']} acc={d['per_class'][a]['acc']}")

    # B) 端到端产绿区判别(observe() 复现), 按 train/val 分列
    train_v, val_v = (json.load(open(args.manifest))["split"]["train"],
                      json.load(open(args.manifest))["split"]["val"])
    train_set, val_set = set(train_v), set(val_v)
    print("\n=== B) 端到端产绿区判别 (observe() + prior ROI 裁图) ===")
    for v in VIDEOS:
        rec = scan_video(det, clf, v)
        if rec is None:
            print(f"  {v}: 跳过(无视频)")
            continue
        out["videos"].append(rec)
        tag = "负例" if v in NEG_VIDEOS else ("val" if v in val_set else "train")
        c = rec["crops"]
        off_r = rec["off"] / c if c else 0
        acc_r = (rec["walk"] + rec["stand"]) / c if c else 0
        print(f"  {v}[{tag}] 裁剪={c:>4} 拒(off)={rec['off']:>4}({off_r:.2f}) "
              f"收(walk+stand)={rec['walk']+rec['stand']:>4}({acc_r:.2f}) 低置信={rec['low_conf']}")

    # 四关判定
    g = out["gates"]
    g1_p, g1 = gate_neg_off_ratio(out["videos"], NEG_VIDEOS)
    g["gate1_neg_off_ratio"] = {"pass": g1_p, "value": g1, "thr": 0.75}
    g2_06_p, g2_06 = gate_true_green_recall(out["videos"], {"违章06"})
    g2_07_p, g2_07 = gate_true_green_recall(out["videos"], {"违章07"})
    g["gate2_dark_green"] = {
        "06_train": {"pass": g2_06_p, "value": g2_06, "note": "train 弱证据, 不当作泛化证据"},
        "07_val": {"pass": g2_07_p, "value": g2_07, "note": "val 真泛化关, 必须诚实"},
        "thr": 0.90,
    }
    g4_01_p, g4_01 = gate_neg_off_ratio(out["videos"], {"违章01"})
    g4_07_p, g4_07 = gate_true_green_recall(out["videos"], {"违章07"})
    g4_11_p, g4_11 = gate_neg_off_ratio(out["videos"], {"违章11"})
    g["gate4_val_generalization"] = {
        "01_neg_off": g4_01, "07_pos_recall": g4_07, "11_neg_off": g4_11,
        "pass": bool(g4_01_p and g4_07_p and g4_11_p),
    }

    # 关3 探针
    if args.probe:
        pv, t0, t1 = parse_probe_window(args.probe)
        prec = scan_window(det, clf, pv, t0, t1)
        if prec:
            p3_p, p3 = gate_probe_window(prec["walk"], prec["off"], prec["other"])
            g["gate3_probe_04"] = {"pass": p3_p, "value": p3, "window": [t0, t1],
                                    "walk": prec["walk"], "off": prec["off"], "other": prec["other"]}
            print(f"\n=== 关3 探针 {pv} [{t0},{t1}]s ===  walk={prec['walk']} off={prec['off']} other={prec['other']} walk占比={p3}")
        else:
            print(f"\n[probe] {pv} 无视频, 跳过")

    # 消融对比(去 outside 版)
    if args.compare_json:
        base_07 = g2_07
        other = json.load(open(args.compare_json))
        o07 = other.get("gates", {}).get("gate2_dark_green", {}).get("07_val", {}).get("value")
        if base_07 is not None and o07 is not None:
            gain_pp = (o07 - base_07) * 100
            out["ablation"] = {
                "07_val_recall_base": base_07, "07_val_recall_drop_outside": o07,
                "gain_pp": round(gain_pp, 1),
                "trigger_downweight": gain_pp >= ABLATION_RECALL_GAIN_PP,
                "threshold_pp": ABLATION_RECALL_GAIN_PP,
            }
            print(f"\n=== 消融(去 outside) === 07 收率 {base_07:.3f} -> {o07:.3f} (回升 {gain_pp:.1f}pp, 触发阈值 {ABLATION_RECALL_GAIN_PP}pp)")

    # 汇总
    print("\n=== 四关判定 ===")
    print(f"  关1 拒负例(01/10): {g['gate1_neg_off_ratio']}")
    print(f"  关2 保暗绿 06train={g2_06:.3f} 07val={g2_07:.3f}")
    print(f"  关4 val泛化: {g['gate4_val_generalization']}")

    dst = os.path.join(ROOT, "data", "output", "diag_classifier_retrain.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nJSON -> {dst}")


if __name__ == "__main__":
    main()
