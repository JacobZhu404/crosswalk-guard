#!/usr/bin/env python3
"""diag_base_leakage_per_video.py — 纯 base 路径(无判别器, τ=0)逐视频漏绿.

cc 裁定(2026-08-04-cc-verify-wb-nega-unified-caliber.md)要求坐实:
  base 扣05 漏绿 ≈ 66 (全量 80 含违章05 ~14).

口径严格对齐 canonical / eval_selection_quality.py:12 + metrics_for_tau:
  漏绿 = gt_green and not(best_conf >= tau and best_color == 'green')
  base 路径 model=None, tau=0 → best_conf>=0 恒真 → 漏绿 = gt_green and best_color != 'green'

本脚本只读、无训练、无判别器、不碰任何缓存. 复用 eval_video 生产上线路径.
"""
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import json
from redlight.models import governing_disc as gd
from redlight.models.traffic_light import TrafficLightDetector
from scripts.eval_selection_quality import eval_video, metrics_for_tau, _video_frames, GT

EXCL = "违章05"

def main():
    gt = json.load(open(GT, encoding="utf-8"))
    det = TrafficLightDetector(gd._cfg_tl(), verbose=False)
    yolo = gd._lazy_yolo()
    by_video = _video_frames(gt)
    videos = sorted(by_video)

    print(f"{'video':<10} {'GT绿帧':>6} {'漏绿':>5} {'扣05漏绿':>8}")
    print("-" * 32)
    per_video_miss = {}
    per_video_gtg = {}
    for V in videos:
        rows = eval_video(V, by_video[V], None, det, yolo, governing_weight=0.0)
        mg = metrics_for_tau(rows, 0.0)  # 全量口径 (不扣05)
        per_video_miss[V] = mg["miss"]
        per_video_gtg[V] = mg["n_green"]
        excl = (V == EXCL)
        print(f"{V:<10} {mg['n_green']:>6} {mg['miss']:>5} {'(扣)' if excl else mg['miss']:>8}")

    total_all = sum(per_video_miss.values())
    total_excl = sum(v for V, v in per_video_miss.items() if V != EXCL)
    print("-" * 32)
    print(f"{'全量':<10} {sum(per_video_gtg.values()):>6} {total_all:>5}")
    print(f"{'扣05':<10} {'':>6} {total_excl:>5}")
    print()
    print(f"cc 估计: 全量漏绿=80, 扣05漏绿≈66 (违章05贡献≈14)")
    print(f"实测:     全量漏绿={total_all}, 扣05漏绿={total_excl}")
    v05 = per_video_miss.get(EXCL)
    print(f"违章05 漏绿={v05} (全量-扣05差={total_all - total_excl}, 应≈{v05})")
    ok = (total_all == 80) and (total_excl == 66)
    print(f"坐实判定: {'PASS ✅ base 漏绿全量80/扣05 66' if ok else '数值见上, 与 cc 估计差异需复核'}")

if __name__ == "__main__":
    main()
