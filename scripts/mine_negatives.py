"""负例质量杠杆: 从负视频(无真绿信号)矿石帧通过 observe() 的"假绿"先验 ROI 裁剪,
保存为 off 类补充训练样本。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/mine_negatives.py --videos 10 \\
      --out-dir datasets/classifier_retrain_negatives

输出:
  - {out_dir}/{video}/{fi}.jpg  — 裁剪(BGR jpg, 与主数据集相同)
  - {out_dir}/labels.csv       — 与主数据集一致的 CSV 列(crop_path,video,fi,source,label=off)
"""
import os
import sys
import argparse
import csv
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2
import numpy as np

from redlight.models.traffic_light import TrafficLightDetector

INPUT_VIDEO = os.path.join(ROOT, "input_video")
_NEG_VIDEOS = {"违章10"}  # 训练负视频(10 在 train split, 01/11 在 val 不泄露)


def mine_one(video, out_dir, sample_step=1):
    """从单视频中扫描 observe()==green 的帧, 裁剪先验 ROI, 存为 off 样本。

    返回: 产出行列表 [{crop_path, video, fi, source, label}]。
    """
    vp = os.path.join(INPUT_VIDEO, f"{video}.mp4")
    if not os.path.isfile(vp):
        print(f"  ⚠️  {video}: 视频文件不存在, 跳过")
        return []
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        print(f"  ⚠️  {video}: 无法打开, 跳过")
        return []

    cfg = types.SimpleNamespace(
        traffic_light=None,
        models=types.SimpleNamespace(ped_signal_model=None),
    )
    det = TrafficLightDetector(cfg, verbose=False)
    det.set_video_prior(video)

    # 如果这个视频没有 prior, 跳过
    if det.signal_prior is None:
        print(f"  ⚠️  {video}: 无 signal_prior, 跳过")
        cap.release()
        return []

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if h == 0 or w == 0:
        h, w = 1080, 1920  # fallback

    vid_out = os.path.join(out_dir, video)
    os.makedirs(vid_out, exist_ok=True)

    rows = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % sample_step != 0:
            fi += 1
            continue

        res = det.observe(frame)
        if res.get("obs") != "green":
            fi += 1
            continue

        # 同 scan_window 裁剪逻辑: prior ROI
        px, py = det.signal_prior
        rp = det.prior_roi_px
        cx, cy = int(px * w), int(py * h)
        x1 = max(0, cx - rp // 2)
        y1 = max(0, cy - rp // 2)
        x2 = min(w, cx + rp // 2)
        y2 = min(h, cy + rp // 2)
        if x2 <= x1 or y2 <= y1:
            fi += 1
            continue
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            fi += 1
            continue

        # 保存裁剪
        fname = f"{fi}.jpg"
        path = os.path.join(vid_out, fname)
        cv2.imwrite(path, roi)

        # 相对路径 (与 labels.csv 约定一致: 相对于 out_dir)
        rel = f"{video}/{fname}"
        rows.append({
            "crop_path": rel,
            "video": video,
            "fi": fi,
            "source": "negative_quality",
            "label": "off",        # 负视频矿石 -> off(不管 stand 真值)
        })
        fi += 1

    cap.release()
    print(f"  {video}: 产出 {len(rows)} 个假绿 off 样本")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["违章10"],
                    help="要挖矿的负视频(默认=[违章10])")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "datasets", "classifier_retrain_negatives"),
                    help="产出目录")
    ap.add_argument("--sample-step", type=int, default=1,
                    help="帧采样步长(1=每帧都检)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []
    for v in args.videos:
        rows = mine_one(v, args.out_dir, sample_step=args.sample_step)
        all_rows.extend(rows)

    if not all_rows:
        print("无产出, 退出")
        return

    csv_path = os.path.join(args.out_dir, "labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "video", "fi", "source", "label"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nlabels.csv -> {csv_path} ({len(all_rows)} 行)")


if __name__ == "__main__":
    main()
