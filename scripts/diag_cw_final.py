"""斑马线终极调试: 逐步打印轮廓信息 + 保存mask."""
import cv2, numpy as np, sys
sys.path.insert(0, r"D:\redlight-crosswalk-violation")
from src.crosswalk_detector import CrosswalkDetector
from src.utils import load_config

cfg = load_config(r"D:\redlight-crosswalk-violation\configs\config.yaml")
det = CrosswalkDetector(cfg)
frame = cv2.imread(r"D:\redlight-crosswalk-violation\data\output\smoke\frame_004.jpg")
h, w = frame.shape[:2]

# 手动调用内部方法看每步
y0 = int(h * 0.45)
roi = frame[y0:, :]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\cw_debug_binary.jpg", binary)

k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_h, iterations=1)
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\cw_debug_morph.jpg", binary)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours: {len(contours)}")
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    x, y, bw, bh = cv2.boundingRect(c)
    ar = bw / float(bh) if bh > 0 else 0
    print(f"  #{i} area={area:.0f} bbox=({x},{y},{bw},{bh}) ar={ar:.2f} y_in_roi={y}")

# 调用 detect 看返回的 mask
mask = det.detect(frame)
nonzero = np.count_nonzero(mask)
total = mask.size
print(f"\nFinal mask: nonzero={nonzero}/{total} ({nonzero/total*100:.1f}%)")
if nonzero > 0:
    # 在原图上叠加 mask 看
    overlay = frame.copy()
    overlay[mask > 0] = [0, 255, 255]  # 青色
    blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\cw_debug_final.jpg", blended)
    print("Saved cw_debug_final.jpg with mask overlay")
else:
    print("MASK IS EMPTY!")
