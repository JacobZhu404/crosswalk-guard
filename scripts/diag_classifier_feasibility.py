"""只读诊断: ped_signal.pt 判别器可行性(灯态治本方向, 第一步).

不改生产代码. 只做评估, 输出 JSON + 摘要.
对应 docs/handoff/2026-07-20-cc-direction-light-classifier-diagnosis.md §3 四问.

方法:
  A) 域内 sanity: 在 datasets/ped_signal/ 2658 张已标 crop(walk/stand/off) 上跑冻结判别器 -> 混淆矩阵.
  B) 端到端产绿区判别: 用现有 TrafficLightDetector.observe() 复现 0.889 管线看到的"绿"区域
     (3 个产绿出口: YOLO框 / prior ROI / candidates面积求和), 裁剪最大绿候选 ROI 送判别器,
     统计判别器对"引擎判绿"区域的判定分布.
     - 负例视频(01/10): 引擎看到的绿必为假绿 -> 判别器判 off(拒) = 治本有效.
     - 正例视频: 引擎看到的绿多为真绿 -> 判别器判 walk/stand(收) = 不杀真绿.
  这样无需新标注即得泛化/判别力证据(GT 已知 01/10 为负例).

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/diag_classifier_feasibility.py
"""
import sys, os, json, csv, types
sys.path.insert(0, "src")
import cv2
import numpy as np

from redlight.models.traffic_light import TrafficLightDetector
from redlight.models.signal_state_classifier import SignalStateClassifier, LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "ped_signal.pt")
CROP_DIR = os.path.join(ROOT, "datasets", "ped_signal")
INPUT_VIDEO = os.path.join(ROOT, "input_video")
VIDEOS = [f"违章{i:02d}" for i in range(1, 12)]
NEG_VIDEOS = {"违章01", "违章10"}  # GT 真负例

# ---------- A) 域内混淆矩阵 ----------
def domain_confusion(clf):
    ytrue, ypred = [], []
    with open(os.path.join(CROP_DIR, "labels.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            p = os.path.join(CROP_DIR, r["crop_path"])
            if not os.path.isfile(p):
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            lab, _ = clf.classify(img)
            ytrue.append(r["label"])
            ypred.append(lab)
    # confusion
    cm = {a: {b: 0 for b in LABELS} for a in LABELS}
    for t, p in zip(ytrue, ypred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    n = len(ytrue)
    acc = sum(1 for t, p in zip(ytrue, ypred) if t == p) / n if n else 0
    per = {}
    for a in LABELS:
        tot = sum(cm[a].values())
        cor = cm[a][a]
        per[a] = {"n": tot, "acc": round(cor / tot, 3) if tot else 0}
    return {"n": n, "acc": round(acc, 3), "confusion": cm, "per_class": per}


# ---------- B) 端到端产绿区判别 ----------
def scan_video(det, clf, video, sample_step=8):
    vp = os.path.join(INPUT_VIDEO, f"{video}.mp4")
    if not os.path.isfile(vp):
        return None
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det.set_video_prior(video)
    h0, w0 = None, None
    rec = {"video": video, "gt_neg": video in NEG_VIDEOS, "engine_green_frames": 0,
           "crops": 0, "off": 0, "walk": 0, "stand": 0, "low_conf": 0,
           "prior_mode": det.signal_prior is not None}
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % sample_step == 0 and frame is not None:
            h, w = frame.shape[:2]
            res = det.observe(frame)
            obs = res.get("obs")
            if obs == "green":
                rec["engine_green_frames"] += 1
                # 裁引擎实际判定区: prior 模式=prior ROI 框(与训练分布一致); 无prior=最大绿候选(兜底)
                if det.signal_prior is not None:
                    px, py = det.signal_prior
                    rp = det.prior_roi_px
                    cx, cy = int(px * w), int(py * h)
                    x1, y1 = max(0, cx - rp // 2), max(0, cy - rp // 2)
                    x2, y2 = min(w, cx + rp // 2), min(h, cy + rp // 2)
                    roi = frame[y1:y2, x1:x2]
                else:
                    spots = res.get("candidates") or []
                    greens = [s for s in spots if s.get("color") == "green" and s.get("area", 0) > 0]
                    if not greens:
                        roi = None
                    else:
                        s = max(greens, key=lambda x: x["area"])
                        x1, y1, x2, y2 = [int(v) for v in s["box"]]
                        roi = frame[max(0, y1):y2, max(0, x1):x2]
                if roi is not None and roi.size > 0:
                    lab, conf = clf.classify(roi)
                    rec["crops"] += 1
                    rec[lab] = rec.get(lab, 0) + 1
                    if conf < 0.5:
                        rec["low_conf"] += 1
        fi += 1
    cap.release()
    return rec


def main():
    clf = SignalStateClassifier(MODEL, verbose=False)
    print(f"[classifier] available={clf.available} path={MODEL}")
    if not clf.available:
        print("判别器不可用, 退出")
        return

    cfg = types.SimpleNamespace(
        traffic_light=None,
        models=types.SimpleNamespace(ped_signal_model=MODEL),
    )
    det = TrafficLightDetector(cfg, verbose=False)

    out = {"domain": domain_confusion(clf), "videos": []}
    print("\n=== A) 域内混淆矩阵 (datasets/ped_signal 2658 张已标 crop) ===")
    d = out["domain"]
    print(f"  总样本={d['n']} 总acc={d['acc']}")
    for a in LABELS:
        print(f"  {a}: n={d['per_class'][a]['n']} acc={d['per_class'][a]['acc']}")
    print("  混淆矩阵 (行=真值 列=预测):")
    print("        " + " ".join(f"{b:>6}" for b in LABELS))
    for a in LABELS:
        print(f"  {a:>5} " + " ".join(f"{d['confusion'][a][b]:>6}" for b in LABELS))

    print("\n=== B) 端到端产绿区判别 (observe() 复现 + 判别器) ===")
    for v in VIDEOS:
        rec = scan_video(det, clf, v)
        if rec is None:
            print(f"  {v}: 跳过(无视频)")
            continue
        out["videos"].append(rec)
        c = rec["crops"]
        off_r = rec["off"] / c if c else 0
        acc_r = (rec["walk"] + rec["stand"]) / c if c else 0
        tag = "负例" if rec["gt_neg"] else "正例"
        print(f"  {v}[{tag}] 引擎绿帧={rec['engine_green_frames']:>4} 裁剪={c:>4} "
              f"拒(off)={rec['off']:>4}({off_r:.2f}) 收(walk+stand)={rec['walk']+rec['stand']:>4}({acc_r:.2f}) "
              f"低置信={rec['low_conf']}")

    # 汇总负例/正例
    neg_off = sum(r["off"] for r in out["videos"] if r["gt_neg"])
    neg_tot = sum(r["crops"] for r in out["videos"] if r["gt_neg"])
    pos_off = sum(r["off"] for r in out["videos"] if not r["gt_neg"])
    pos_tot = sum(r["crops"] for r in out["videos"] if not r["gt_neg"])
    out["summary"] = {
        "neg_off_ratio": round(neg_off / neg_tot, 3) if neg_tot else None,
        "neg_crops": neg_tot,
        "pos_off_ratio": round(pos_off / pos_tot, 3) if pos_tot else None,
        "pos_crops": pos_tot,
    }
    print("\n=== 汇总 ===")
    print(f"  负例(01/10) 引擎绿裁剪={neg_tot} 判别器拒(off)={neg_off} 拒识率={out['summary']['neg_off_ratio']}")
    print(f"  正例       引擎绿裁剪={pos_tot} 判别器拒(off)={pos_off} 误拒率={out['summary']['pos_off_ratio']}")

    dst = os.path.join(ROOT, "data", "output", "diag_classifier_feasibility.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nJSON -> {dst}")


if __name__ == "__main__":
    main()
