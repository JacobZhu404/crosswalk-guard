"""斑马线检测 v8 中间产物诊断 (针对 违章01 ts=61.02 真实条纹漏检帧)。

输出每一步的二值掩膜 + 轮廓统计, 定位丢失点。
"""
import sys
import os
import cv2
import numpy as np

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config, project_root


def diagnose_frame(video_path, ts, cfg, out_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = int(round(ts * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("cannot read"); return

    h, w = frame.shape[:2]
    y0 = int(h * 0.45)
    roi = frame[y0:, :]
    roi_h, roi_w = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)

    # Step 1: white mask (raw)
    white_raw = ((S < 55) & (V > 145)).astype(np.uint8) * 255
    print(f"Step1 white_raw: {np.count_nonzero(white_raw)} px "
          f"({100*np.count_nonzero(white_raw)/(roi_h*roi_w):.1f}%)")

    # Step 1b: green exclusion
    is_green = ((H > 35) & (H < 85) & (S > 35))
    combined_raw = (~is_green & (S.astype(np.int16) < 55) &
                    (V > 145)).astype(np.uint8) * 255
    print(f"Step1b after_green_exclude: {np.count_nonzero(combined_raw)} px")

    # Step 2: open
    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(combined_raw, cv2.MORPH_OPEN, k_o, iterations=1)
    print(f"Step2 opened: {np.count_nonzero(opened)} px")

    # Step 3: close horizontal
    k_ch = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_ch, iterations=2)
    print(f"Step3 closed_h: {np.count_nonzero(closed)} px")

    # Step 4: open again
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_o, iterations=1)
    print(f"Step4 final_cleaned: {np.count_nonzero(cleaned)} px")

    # Contours analysis
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clist = list(contours)
    print(f"\nContours ({len(clist)}):")
    clist.sort(key=cv2.contourArea, reverse=True)
    for i, c in enumerate(clist[:10]):
        area = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh) if bh > 0 else 0
        print(f"  #{i} area={area:.0f} ar={ar:.1f} pos=({x},{y}) size={bw}x{bh}")

    # Save visualization images
    def save_step(name, binary_img):
        vis = roi.copy()
        m = (binary_img > 0).astype(np.uint8)
        vis[m == 1] = (0, 255, 0)
        path = os.path.join(out_dir, f"debug_{name}.jpg")
        cv2.imwrite(path, vis)
        print(f"  saved: {path}")

    save_step("s1_white_raw", white_raw)
    save_step("s1b_combined", combined_raw)
    save_step("s2_opened", opened)
    save_step("s3_closed", closed)
    save_step("s4_cleaned", cleaned)

    # Also show HSV channel values at a sample point where stripes should be
    # (manually estimate: lower-left of ROI where people stand)
    print("\n--- Sample HSV at ROI grid ---")
    for gy in range(0, roi_h, roi_h // 6):
        for gx in range(0, roi_w, roi_w // 6):
            hh, ss, vv = int(H[gy, gx]), int(S[gy, gx]), int(V[gy, gx])
            print(f"  ROI[{gy:4d},{gx:4d}] H={hh:3d} S={ss:3d} V={vv:3d}")


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    base_in = r"E:\BaiduNetdiskDownload"
    out_dir = os.path.join(project_root(), "data", "output", "cw_debug")
    os.makedirs(out_dir, exist_ok=True)
    video = os.path.join(base_in, "违章01.mp4")
    print("===== Diagnosing 违章01 ts=61.02 =====")
    diagnose_frame(video, 61.02, cfg, out_dir)


if __name__ == "__main__":
    main()
