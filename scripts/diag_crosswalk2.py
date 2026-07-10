"""斑马线检测中间步骤诊断: 保存每一步的掩膜图, 找出哪步丢失了条纹."""
import cv2
import numpy as np

f = r"D:\redlight-crosswalk-violation\data\output\smoke\frame_004.jpg"
frame = cv2.imread(f)
h, w = frame.shape[:2]
y0 = int(h * 0.38)
roi = frame[y0:, :]

# 步骤1: LAB高亮度
lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
l_ch = lab[:, :, 0]
a_ch = lab[:, :, 1].astype(np.int16) - 128
b_ch = lab[:, :, 2].astype(np.int16) - 128
white_lab = ((l_ch > 135) & (np.abs(a_ch) < 40) & (np.abs(b_ch) < 45)).astype(np.uint8) * 255
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\step1_lab_white.jpg", white_lab)

# 步骤2: 灰度自适应阈值
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
th = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 51, 12)
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\step2_adapt_thresh.jpg", th)

# 步骤3: 合并
combined = cv2.bitwise_or(white_lab, th)
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\step3_combined.jpg", combined)

# 步骤4: 形态学后
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
combined2 = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open, iterations=1)
k_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 8))
combined2 = cv2.morphologyEx(combined2, cv2.MORPH_CLOSE, k_close_h, iterations=4)
k_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 15))
combined2 = cv2.morphologyEx(combined2, cv2.MORPH_CLOSE, k_close_v, iterations=2)
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\step4_morph.jpg", combined2)

# 轮廓统计
contours, _ = cv2.findContours(combined2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数: {len(contours)}")
for i, c in enumerate(contours[:10]):
    area = cv2.contourArea(c)
    x, y, bw, bh = cv2.boundingRect(c)
    ar = bw / float(bh) if bh > 0 else 0
    print(f"  #{i} area={area:.0f} bbox=({x},{y},{bw},{bh}) ar={ar:.2f}")

print("所有步骤图片已保存到 data/output/smoke/step*.jpg")
