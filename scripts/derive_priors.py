"""数据驱动推导行人信号灯 prior(中心+ROI), 仅用已抽帧, 免 GT.

思路: 行人信号是固定实物 -> 在绝大多数帧里都有"高饱和红或绿亮点"(红段亮红/绿段亮绿).
     扫描每帧小面积(排除大车灯) + 上部(cy<0.62) 的红/绿连通块, 跨帧按网格聚类,
     频率最高且红绿都出现的网格 = 信号外壳中心 -> 作为 signal_prior.
ROI: 取该网格附近亮斑的空间散布, 给足余量(默认 160, 远小灯 220).

输出: 终端打印每视频 Top 候选 + 建议 prior; --write 直接写回 configs/light_priors.json.

用法:
  python scripts/derive_priors.py                 # 只打印建议
  python scripts/derive_priors.py --write        # 写回 light_priors.json
  python scripts/derive_priors.py --videos 违章02 违章04
"""
import os
import sys
import argparse
import json

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from scan_pedestrian_signal import small_color_blobs, robust_imread, cluster

FRAMES_ROOT = os.path.join(ROOT, "datasets", "frames")
PRIORS_PATH = os.path.join(ROOT, "configs", "light_priors.json")

RED_RANGES = [(0, 12), (168, 179)]
GREEN_RANGES = [(38, 82)]


def scan_video(video, stride=3, grid=0.03, area_max=700):
    vdir = os.path.join(FRAMES_ROOT, video)
    if not os.path.isdir(vdir):
        return None
    files = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
    recs = []  # (cx, cy, area, color)
    for i, fn in enumerate(files):
        if i % stride != 0:
            continue
        fr = robust_imread(os.path.join(vdir, fn))
        if fr is None:
            continue
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        for cx, cy, a in small_color_blobs(hsv, RED_RANGES, area_max=area_max):
            recs.append((cx, cy, a, "r"))
        for cx, cy, a in small_color_blobs(hsv, GREEN_RANGES, area_max=area_max):
            recs.append((cx, cy, a, "g"))
    if not recs:
        return None
    # 合并红绿聚类(网格)
    buckets = {}
    for cx, cy, a, col in recs:
        key = (round(cx / grid) * grid, round(cy / grid) * grid)
        buckets.setdefault(key, {"n": 0, "r": 0, "g": 0, "areas": []})
        buckets[key]["n"] += 1
        buckets[key]["areas"].append(a)
        if col == "r":
            buckets[key]["r"] += 1
        else:
            buckets[key]["g"] += 1
    res = []
    for (gx, gy), d in buckets.items():
        res.append({
            "cx": round(gx, 3), "cy": round(gy, 3),
            "n": d["n"], "r": d["r"], "g": d["g"],
            "avg_area": int(np.mean(d["areas"])),
            "both": (d["r"] > 0 and d["g"] > 0),
        })
    res.sort(key=lambda r: (-r["n"], -int(r["both"]), -r["avg_area"]))
    return res


def suggest_prior(video, top):
    if not top:
        return None
    best = top[0]
    # ROI: 视信号大小; 平均面积越小灯越远 -> ROI 放大以包容手持漂移
    aa = best["avg_area"]
    roi = 200 if aa < 60 else (160 if aa < 200 else 140)
    return [best["cx"], best["cy"], roi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--write", action="store_true", help="写回 light_priors.json")
    args = ap.parse_args()

    if args.videos:
        videos = args.videos
    else:
        videos = sorted(d for d in os.listdir(FRAMES_ROOT)
                        if os.path.isdir(os.path.join(FRAMES_ROOT, d))
                        and d.startswith("违章"))

    suggestions = {}
    for v in videos:
        top = scan_video(v, stride=args.stride)
        if not top:
            print(f"[无信号] {v}")
            continue
        b = top[0]
        print(f"\n===== {v}  扫描候选 Top3 =====")
        for r in top[:3]:
            print(f"  ({r['cx']:.3f},{r['cy']:.3f}) freq={r['n']:4d} "
                  f"r={r['r']:4d} g={r['g']:4d} both={int(r['both'])} avgArea={r['avg_area']}")
        sg = suggest_prior(v, top)
        suggestions[v] = sg
        print(f"  >> 建议 prior = {sg}")

    if args.write and suggestions:
        cur = {}
        if os.path.exists(PRIORS_PATH):
            with open(PRIORS_PATH, encoding="utf-8") as f:
                cur = json.load(f)
        for v, sg in suggestions.items():
            cur[v] = sg
        with open(PRIORS_PATH, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
        print(f"\n已写回 {PRIORS_PATH} ({len(suggestions)} 个视频)")


if __name__ == "__main__":
    main()
