"""斑马线检测终极诊断: 逐步输出, 找到确切断点."""
import cv2
import numpy as np

f = r"D:\redlight-crosswalk-violation\data\output\smoke\frame_004.jpg"
frame = cv2.imread(f)
h, w = frame.shape[:2]
y0 = int(h * 0.50)
roi = frame[y0:, :]
print(f"frame={w}x{h}, roi starts y0={y0}, roi size={roi.shape[1]}x{roi.shape[0]}")

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 试不同阈值
for th_val in [120, 130, 140, 150]:
    _, bin_img = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)
    white_px = np.count_nonzero(bin_img)
    total_px = bin_img.size
    print(f"  threshold={th_val}: white_ratio={white_px/total_px:.3f} ({white_px}/{total_px})")
    cv2.imwrite(f"D:\\redlight-crosswalk-violation\\data\\output\\smoke\\th{th_val}.jpg", bin_img)

# 用 Otsu 看看自动阈值是多少
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\otsu.jpg", otsu)
print(f"  Otsu auto threshold applied")

# 直接保存灰度ROI
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\roi_gray.jpg", gray)

# 保存原始ROI
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\roi_raw.jpg", roi)
