"""v10 梯度剖面诊断 (针对 违章01 ts=61.02 条纹漏检帧)。

输出: 1D row_density 剖面图 + 阈值线 + 原始帧标注,
     用于判断斑马线条纹是否产生梯度信号。
"""
import sys
import os
import cv2
import numpy as np

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def diagnose(video_path, ts, out_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = int(round(ts * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("cannot read"); return

    h, w = frame.shape[:2]
    y0 = int(h * 0.50)
    roi = frame[y0:, :]
    roi_h, roi_w = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.abs(sobely)
    row_density = np.mean(grad_mag, axis=1)

    kernel_size = min(15, roi_h // 4)
    if kernel_size % 2 == 0: kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(row_density, kernel, mode='same')

    mean_val = np.mean(smoothed)
    std_val = np.std(smoothed)
    thresh = mean_val + 1.5 * std_val

    print(f"ROI: {roi_w}x{roi_h}  y0={y0}")
    print(f"row_density: mean={mean_val:.2f} std={std_val:.2f} thresh={thresh:.2f}")
    print(f"max_smoothed={np.max(smoothed):.2f} at row={np.argmax(smoothed)}")
    print(f"rows above thresh: {np.sum(smoothed > thresh)} / {roi_h}")

    # Save profile as text + visualization
    # Create profile visualization (side by side with ROI)
    profile_h = 300
    profile_w = roi_w
    prof_img = np.full((profile_h, profile_w, 3), 240, dtype=np.uint8)
    # Draw profile curve normalized to fit
    max_v = max(np.max(smoothed), thresh) * 1.1
    pts_x = np.linspace(0, profile_w - 1, len(smoothed)).astype(int)
    pts_y = (profile_h - 10 - (smoothed / max_v) * (profile_h - 20)).astype(int)
    pts_y_clip = np.clip(pts_y, 5, profile_h - 5)
    for i in range(len(pts_x) - 1):
        cv2.line(prof_img, (pts_x[i], pts_y_clip[i]), (pts_x[i+1], pts_y_clip[i+1]), (0, 0, 255), 1)
    # Threshold line
    ty = int(profile_h - 10 - (thresh / max_v) * (profile_h - 20))
    cv2.line(prof_img, (0, ty), (profile_w - 1, ty), (0, 200, 0), 2)

    # Combine: top=ROI with stripe region hint, bottom=profile
    roi_display = cv2.resize(roi, (roi_w, roi_h // 2))
    combined = np.vstack([roi_display,
                          np.full((10, roi_w, 3), 200, dtype=np.uint8),
                          prof_img])
    p = os.path.join(out_dir, f"grad_profile_ts{ts}.jpg")
    cv2.imwrite(p, combined)
    print(f"saved: {p}")


def main():
    base_in = r"E:\BaiduNetdiskDownload"
    out_dir = os.path.join(ROOT, "data", "output", "cw_debug")
    os.makedirs(out_dir, exist_ok=True)
    video = os.path.join(base_in, "违章01.mp4")
    print("===== Gradient Profile: 违章01 ts=61.02 =====")
    diagnose(video, 61.02, out_dir)


if __name__ == "__main__":
    main()
