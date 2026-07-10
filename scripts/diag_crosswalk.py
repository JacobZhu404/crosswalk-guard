"""诊断斑马线区域颜色值, 用于调参."""
import cv2
import numpy as np

f = r"D:\redlight-crosswalk-violation\data\output\smoke\frame_004.jpg"
frame = cv2.imread(f)
h, w = frame.shape[:2]
# 斑马线大致在画面中下部
roi = frame[int(h*0.50):int(h*0.88), int(w*0.30):int(w*0.85)]
lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

print("ROI shape:", roi.shape)
print(f"  L  mean={lab[:,:,0].mean():.1f} std={lab[:,:,0].std():.1f}")
print(f"  A  mean={lab[:,:,1].mean():.1f} std={lab[:,:,1].std():.1f}")
print(f"  B  mean={lab[:,:,2].mean():.1f} std={lab[:,:,2].std():.1f}")
print(f"  V  mean={hsv[:,:,2].mean():.1f} std={hsv[:,:,2].std():.1f}")
print(f"  S  mean={hsv[:,:,1].mean():.1f} std={hsv[:,:,1].std():.1f}")
print(f"  H  mean={hsv[:,:,0].mean():.1f} std={hsv[:,:,0].std():.1f}")

# 采样几个点 (条纹和间隙)
for y, x in [(roi.shape[0]//4, roi.shape[1]//2),
             (roi.shape[0]//2, roi.shape[1]//2),
             (3*roi.shape[0]//4, roi.shape[1]//2)]:
    bgr = roi[y, x]
    l, a, b = lab[y, x]
    v, s, hv = hsv[y, x]
    print(f"  pixel({x},{y}) BGR=({bgr[0]},{bgr[1]},{bgr[2]}) L={l} A={a} B={b} V={v} S={s} H={hv}")

# 保存 ROI 裁剪图方便看
cv2.imwrite(r"D:\redlight-crosswalk-violation\data\output\smoke\crosswalk_roi.jpg", roi)
print("ROI saved to crosswalk_roi.jpg")
