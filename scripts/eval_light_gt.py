"""红绿灯状态 GT 评测 (要求#6)。

读取 datasets/gt/light_state/*_gt.csv (需人工填好 gt_state 列),
对已完成标注的片段计算信号灯状态分类的 Precision/Recall/F1。

用法:
    python scripts/eval_light_gt.py
    python scripts/eval_light_gt.py datasets/gt/light_state/违章01_gt.csv
"""
import sys
import os
import csv
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from redlight.evaluation.evaluator import Evaluator


def load_labeled(gt_csv):
    preds, gts = [], []
    with open(gt_csv, newline="") as f:
        for r in csv.DictReader(f):
            g = (r.get("gt_state") or "").strip()
            if not g:
                continue  # 未标注片段跳过
            preds.append(r["predicted_state"])
            gts.append(g)
    return preds, gts


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(
        ROOT, "datasets", "gt", "light_state", "*_gt.csv")))
    all_pred, all_gt = [], []
    ev = Evaluator()
    for p in paths:
        preds, gts = load_labeled(p)
        if not gts:
            print(f"[跳过] {os.path.basename(p)}: 无已标注片段")
            continue
        rep = ev.evaluate_light_states(preds, gts)
        print(f"\n===== {os.path.basename(p)} (n={rep['n']}) =====")
        print(f"  accuracy={rep['accuracy']:.3f}  macro_f1={rep['macro_f1']:.3f}")
        for cls, d in rep["per_class"].items():
            print(f"  {cls:9s} P={d['precision']:.3f} R={d['recall']:.3f} "
                  f"F1={d['f1']:.3f} support={d['support']}")
        all_pred += preds
        all_gt += gts
    if all_gt:
        rep = ev.evaluate_light_states(all_pred, all_gt)
        print(f"\n===== 合计 (n={rep['n']}) =====")
        print(f"  accuracy={rep['accuracy']:.3f}  macro_f1={rep['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
