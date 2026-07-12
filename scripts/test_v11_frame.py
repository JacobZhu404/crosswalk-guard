"""v11 单帧验证: 从原始视频抽指定时间戳的帧, 跑 v11 斑马线检测, 输出带掩膜叠加的图片。
用于快速确认 v11 掩膜是否对齐到真实斑马线位置。

用法:
    python scripts/test_v11_frame.py <video> <ts_seconds> <out.jpg>
"""
import os
import sys

os.environ["TQDM_DISABLE"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
import numpy as np

from redlight.infrastructure.config import load_config, project_root
from redlight.models.crosswalk import CrosswalkDetector
from redlight.models.vehicle import VehicleDetector

if len(sys.argv) < 4:
    print("usage: test_v11_frame.py <video.mp4> <ts_s> <out.jpg>")
    sys.exit(1)

video = sys.argv[1]
ts = float(sys.argv[2])
out = sys.argv[3]

cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
fi = int(round(ts * fps))
cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
ret, frame = cap.read()
cap.release()
if not ret:
    sys.exit(1)

# 检测车辆(传给斑马线检测器做锚定)
veh_det = VehicleDetector(cfg, verbose=False)
dets = veh_det.detect(frame)
boxes = [d['box'] for d in dets]

# 斑马线检测 v11
cw = CrosswalkDetector(cfg, verbose=True)
mask_v10 = cw._cv_v10(frame)          # 对比基线
mask_v11 = cw.detect(frame, vehicle_boxes=boxes)  # v11 + 车辆锚定

# 绘制: 左=v10, 右=v11
h, w = frame.shape[:2]
combo = np.zeros((h, w * 2, 3), dtype=np.uint8)
left = frame.copy(); right = frame.copy()

# 叠加掩膜(半透明蓝)
for m, img in [(mask_v10, left), (mask_v11, right)]:
    overlay = img.copy()
    overlay[m > 0] = [255, 200, 0]  # BGR yellow-ish blue
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    # 描边
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 150, 0), 2)

# 画车框
for b in boxes:
    for img in [left, right]:
        x1,y1,x2,y2 = map(int,b[:4])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)

# 标签信息
info = f"ts={ts:.1f}s | v10_px={np.count_nonzero(mask_v10)} ov10={0:.3f} | v11_px={np.count_nonzero(mask_v11)} n_vehicles={len(boxes)}"
cv2.rectangle(combo, (0,0),(w*2,22),(0,0,0),-1)
cv2.putText(combo, info, (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255),1,cv2.LINE_AA)
cv2.putText(left, "v10(fixed y0=0.50)", (4,h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0),1)
cv2.putText(right, "v11(multi-scan+anchor)", (4,h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0),1)

combo[:, :w] = left; combo[:, w:] = right
cv2.imwrite(out, combo)
print("wrote", out, f"| v10={np.count_nonzero(mask_v10)}px v11={np.count_nonzero(mask_v11)}px")
