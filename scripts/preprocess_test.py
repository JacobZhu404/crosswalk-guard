"""图像预处理增强实验: 对高置信度帧的车牌区域做不同预处理, 看能否让J被正确识别。

预处理策略:
  A) 原图 (baseline)
  B) 2x放大 (INTER_CUBIC)
  C) 3x放大 + 锐化
  D) 灰度+CLAHE对比度增强
  E) 二值化 (Otsu)
  F) 放大+CLAHE+锐化 组合
  G) 上下颠倒 (因为车牌字符J在倒置时与L区分更明显? 验证)
"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.models.plate import PlateRecognizer
from redlight.evaluation.metrics import levenshtein
from rapidocr_onnxruntime import RapidOCR


TRUTH = "京JLE560"


def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def clahe(gray):
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return c.apply(gray)


def preprocess_variants(crop):
    """返回 {name: img} 字典。"""
    variants = {}
    # A 原图
    variants["A_原图"] = crop
    # B 2x放大
    h, w = crop.shape[:2]
    variants["B_2x放大"] = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    # C 3x放大+锐化
    big = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    variants["C_3x+锐化"] = sharpen(big)
    # D 灰度+CLAHE
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants["D_CLAHE"] = cv2.cvtColor(clahe(gray), cv2.COLOR_GRAY2BGR)
    # E Otsu二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants["E_二值化"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # F 放大+CLAHE+锐化
    big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    big_clahe = clahe(big_gray)
    big_sharp = sharpen(cv2.cvtColor(big_clahe, cv2.COLOR_GRAY2BGR))
    variants["F_组合"] = big_sharp
    # G 2x放大+CLAHE
    med = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    med_gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    med_clahe = clahe(med_gray)
    variants["G_2x+CLAHE"] = cv2.cvtColor(med_clahe, cv2.COLOR_GRAY2BGR)
    return variants


def main():
    video_path = r"E:\BaiduNetdiskDownload\违章02.mp4"
    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))

    out_dir = os.path.join(ROOT, "data", "output", "preprocess_test")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hl = PlateRecognizer(cfg)
    rapid = RapidOCR()

    # 选几个高置信度帧
    target_frames = [3192, 3200, 3204, 3212, 3220]

    print(f"真值: {TRUTH}")
    print("=" * 120)

    for tf in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, tf)
        ret, frame = cap.read()
        if not ret:
            continue
        ts = tf / fps

        # HyperLPR3 检测车牌位置
        hl_results = hl.detect(frame)
        hl_best = None
        for p in hl_results:
            if p.get("text") and (hl_best is None or p["conf"] > hl_best["conf"]):
                hl_best = p

        if hl_best is None:
            print(f"\n[帧{tf} @{ts:.2f}s] HyperLPR3 未检测到车牌")
            continue

        x1, y1, x2, y2 = [int(v) for v in hl_best["xyxy"]]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]

        print(f"\n[帧{tf} @{ts:.2f}s] HyperLPR3={hl_best['text']} conf={hl_best['conf']:.3f} "
              f"尺寸={x2-x1}x{y2-y1} 位置=({x1},{y1})")
        print(f"  {'预处理':<16s} | {'RapidOCR结果':<16s} {'conf':>6s} {'ed':>3s} | {'HyperLPR3结果':<16s} {'conf':>6s} {'ed':>3s}")
        print("  " + "-" * 90)

        variants = preprocess_variants(crop)

        for name, img in variants.items():
            # 保存预处理图像
            safe_name = name.replace("+", "_")
            cv2.imwrite(os.path.join(out_dir, f"f{tf}_{safe_name}.jpg"), img)

            # RapidOCR 识别
            rapid_text, rapid_conf = "", 0.0
            try:
                result, _ = rapid(img)
                if result:
                    texts = [str(r[1]).replace(" ", "").replace("-", "").replace("·", "")
                             for r in result if r and len(r) > 1]
                    joined = "".join(texts)
                    rapid_text = joined
                    rapid_conf = max(float(r[2]) for r in result if r and len(r) > 2)
            except Exception:
                pass

            # HyperLPR3 识别 (对预处理后的图)
            hl_text, hl_conf = "", 0.0
            try:
                hl_res = hl.detect(img)
                for p in hl_res:
                    if p.get("text"):
                        hl_text = p["text"]
                        hl_conf = p["conf"]
                        break
            except Exception:
                pass

            r_ed = levenshtein(rapid_text, TRUTH) if rapid_text else -1
            h_ed = levenshtein(hl_text, TRUTH) if hl_text else -1
            r_flag = " ✅" if r_ed == 0 else (" ~" if 0 < r_ed <= 2 else "")
            h_flag = " ✅" if h_ed == 0 else (" ~" if 0 < h_ed <= 2 else "")

            print(f"  {name:<16s} | {rapid_text:<16s} {rapid_conf:6.3f} {r_ed:3d}{r_flag:2s} | "
                  f"{hl_text:<16s} {hl_conf:6.3f} {h_ed:3d}{h_flag:2s}")

    cap.release()
    print(f"\n预处理图像已保存到: {out_dir}")


if __name__ == "__main__":
    main()
