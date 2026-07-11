"""双引擎车牌识别对比: HyperLPR3 (CRNN) vs RapidOCR (PP-OCRv4, SVTR-Transformer)

对同一视频逐帧:
  1) HyperLPR3: 检测+识别 (CRNN+CTC)
  2) RapidOCR: 通用OCR (PP-OCRv4, 含SVTR结构)
     - 用 HyperLPR3 的检测框裁剪车牌区域 -> RapidOCR 识别
     - 同时也对整帧做 RapidOCR, 筛选车牌格式文本
  3) 对比两者与真值的编辑距离/字符准确率

用法:
    python scripts/compare_ocr.py <video> [--fps 8] [--truth 京JLE560]
"""
import os
import sys
import csv
import re
import argparse
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer

try:
    from rapidocr_onnxruntime import RapidOCR
    _HAS_RAPID = True
except Exception:
    _HAS_RAPID = False


PROVINCES = set("京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新港澳")


def levenshtein(a, b):
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def char_accuracy(pred, truth):
    if not pred:
        return 0.0
    ed = levenshtein(pred, truth)
    return max(0.0, 1.0 - ed / max(len(pred), len(truth)))


def is_plate_like(text):
    if not text or len(text) < 6 or len(text) > 9:
        return False
    if text[0] not in PROVINCES:
        return False
    return True


def filter_plate(rapid_results):
    """从 RapidOCR 结果中筛选最像车牌的文本。"""
    if not rapid_results:
        return None
    candidates = []
    for item in rapid_results:
        if len(item) < 3:
            continue
        box, text, conf = item[0], str(item[1]), float(item[2])
        text = text.replace(" ", "").replace("-", "").replace("·", "")
        if is_plate_like(text):
            candidates.append((text, conf, box))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=r"E:\BaiduNetdiskDownload\违章02.mp4")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--truth", default="京JLE560")
    ap.add_argument("--start", type=float, default=100.0, help="起始时间(秒), 默认100s聚焦目标车")
    ap.add_argument("--end", type=float, default=111.0, help="结束时间(秒)")
    args = ap.parse_args()

    if not _HAS_RAPID:
        print("[错误] rapidocr_onnxruntime 未安装, 请 pip install rapidocr_onnxruntime")
        return

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(ROOT, "data", "output", "ocr_compare", video_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("无法打开视频")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / args.fps)))

    hl = PlateRecognizer(cfg)
    rapid = RapidOCR()

    rows = []
    frame_idx = 0
    proc = 0

    print(f"真值: {args.truth}")
    print(f"时间窗口: {args.start}s ~ {args.end}s")
    print("=" * 110)
    print(f"{'帧':>6s} {'时间':>7s} | {'HyperLPR3':<16s} {'conf':>6s} {'ed':>3s} | "
          f"{'RapidOCR整帧':<16s} {'conf':>6s} {'ed':>3s} | "
          f"{'RapidOCR裁剪':<16s} {'conf':>6s} {'ed':>3s}")
    print("-" * 110)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        if ts < args.start or ts > args.end:
            frame_idx += 1
            continue
        if frame_idx % interval != 0:
            frame_idx += 1
            continue
        proc += 1

        # 1) HyperLPR3
        hl_best = None
        for p in hl.detect(frame):
            if p.get("text"):
                if hl_best is None or p["conf"] > hl_best["conf"]:
                    hl_best = p

        # 2) RapidOCR 整帧
        rapid_full_text, rapid_full_conf = "", 0.0
        try:
            result, _ = rapid(frame)
            picked = filter_plate(result)
            if picked:
                rapid_full_text, rapid_full_conf, _ = picked
        except Exception as e:
            pass

        # 3) RapidOCR 裁剪车牌区域 (用 HyperLPR3 的检测框)
        rapid_crop_text, rapid_crop_conf = "", 0.0
        if hl_best is not None:
            x1, y1, x2, y2 = [int(v) for v in hl_best["xyxy"]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                pad = 8
                crop_pad = cv2.copyMakeBorder(crop, pad, pad, pad, pad,
                                              cv2.BORDER_REPLICATE)
                try:
                    result2, _ = rapid(crop_pad)
                    if result2:
                        texts = [str(r[1]).replace(" ", "").replace("-", "")
                                 for r in result2 if r and len(r) > 1]
                        joined = "".join(texts)
                        if is_plate_like(joined):
                            rapid_crop_text = joined
                            rapid_crop_conf = max(float(r[2]) for r in result2 if r and len(r) > 2)
                        elif texts:
                            rapid_crop_text = texts[0]
                            rapid_crop_conf = float(result2[0][2])
                except Exception as e:
                    pass

        hl_text = hl_best["text"] if hl_best else ""
        hl_conf = hl_best["conf"] if hl_best else 0.0
        hl_ed = levenshtein(hl_text, args.truth) if hl_text else -1
        rf_ed = levenshtein(rapid_full_text, args.truth) if rapid_full_text else -1
        rc_ed = levenshtein(rapid_crop_text, args.truth) if rapid_crop_text else -1

        flag_hl = " ✅" if hl_ed == 0 else (" ~" if 0 < hl_ed <= 2 else "")
        flag_rf = " ✅" if rf_ed == 0 else (" ~" if 0 < rf_ed <= 2 else "")
        flag_rc = " ✅" if rc_ed == 0 else (" ~" if 0 < rc_ed <= 2 else "")

        print(f"{frame_idx:6d} {ts:7.2f} | {hl_text:<16s} {hl_conf:6.3f} {hl_ed:3d}{flag_hl:2s} | "
              f"{rapid_full_text:<16s} {rapid_full_conf:6.3f} {rf_ed:3d}{flag_rf:2s} | "
              f"{rapid_crop_text:<16s} {rapid_crop_conf:6.3f} {rc_ed:3d}{flag_rc:2s}")

        rows.append({
            "frame": frame_idx, "time": round(ts, 2),
            "hl_text": hl_text, "hl_conf": round(hl_conf, 4), "hl_ed": hl_ed,
            "rapid_full_text": rapid_full_text, "rapid_full_conf": round(rapid_full_conf, 4), "rapid_full_ed": rf_ed,
            "rapid_crop_text": rapid_crop_text, "rapid_crop_conf": round(rapid_crop_conf, 4), "rapid_crop_ed": rc_ed,
        })
        frame_idx += 1

    cap.release()

    # 写 CSV
    csv_path = os.path.join(out_dir, "compare.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time", "hl_text", "hl_conf", "hl_ed",
                    "rapid_full_text", "rapid_full_conf", "rapid_full_ed",
                    "rapid_crop_text", "rapid_crop_conf", "rapid_crop_ed"])
        for r in rows:
            w.writerow([r["frame"], r["time"], r["hl_text"], r["hl_conf"], r["hl_ed"],
                        r["rapid_full_text"], r["rapid_full_conf"], r["rapid_full_ed"],
                        r["rapid_crop_text"], r["rapid_crop_conf"], r["rapid_crop_ed"]])

    # 汇总统计
    print("\n" + "=" * 110)
    print(f"采样帧数: {proc}  真值: {args.truth}")
    print(f"CSV: {csv_path}")

    def stats(name, text_key, ed_key, conf_key):
        valid = [r for r in rows if r[text_key]]
        if not valid:
            print(f"\n{name}: 无识别结果")
            return
        exact = sum(1 for r in valid if r[ed_key] == 0)
        near = sum(1 for r in valid if 0 < r[ed_key] <= 2)
        eds = [r[ed_key] for r in valid]
        char_accs = [char_accuracy(r[text_key], args.truth) for r in valid]
        confs = [r[conf_key] for r in valid if r[conf_key] is not None]
        print(f"\n=== {name} ===")
        print(f"  有效识别帧数: {len(valid)}")
        print(f"  完全匹配 (ed=0): {exact}  ({exact/len(valid)*100:.1f}%)")
        print(f"  近似匹配 (ed<=2): {near}  ({near/len(valid)*100:.1f}%)")
        print(f"  平均编辑距离: {sum(eds)/len(eds):.2f}")
        print(f"  平均字符准确率: {sum(char_accs)/len(char_accs)*100:.1f}%")
        if confs:
            print(f"  平均置信度: {sum(confs)/len(confs):.3f}")
        # Top-5
        cnt = Counter(r[text_key] for r in valid)
        print(f"  Top-5 识别结果:")
        for txt, n in cnt.most_common(5):
            ed = levenshtein(txt, args.truth)
            flag = " ✅" if ed == 0 else (" ~" if ed <= 2 else "")
            print(f"    {txt:16s} 次数={n:3d} 编辑距离={ed}{flag}")

    stats("HyperLPR3 (CRNN+CTC)", "hl_text", "hl_ed", "hl_conf")
    stats("RapidOCR 整帧 (PP-OCRv4)", "rapid_full_text", "rapid_full_ed", "rapid_full_conf")
    stats("RapidOCR 裁剪 (PP-OCRv4)", "rapid_crop_text", "rapid_crop_ed", "rapid_crop_conf")


if __name__ == "__main__":
    main()
