"""Phase A 抽检: _dump 代表性假绿帧的 ROI 裁图(紧160px + 扩320px), 在
当前 prior(0.2,0.35) 与早前 prior-重定位调查发现的真灯位(0.50,0.10) 两处各取,
供视觉定根因(车辆绿 / 绿植 / 反光 / 真灯)。纯只读, 输出到 data/output/mine_false_green/crops。
红线: 不碰生产码/权重; GT 仅 oracle。"""
import os, cv2, numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = "/Users/jacob/personal/crosswalk-guard/input_video"
OUT = os.path.join(PROJECT_ROOT, "data", "output", "mine_false_green", "crops")
os.makedirs(OUT, exist_ok=True)
PRIOR_ROI_PX = 160
EXPAND = 320

def crop(frame, px, py, roi_px, w, h):
    cx_i, cy_i = int(px * w), int(py * h)
    x1 = max(0, cx_i - roi_px // 2); y1 = max(0, cy_i - roi_px // 2)
    x2 = min(w, cx_i + roi_px // 2); y2 = min(h, cy_i + roi_px // 2)
    return frame[y1:y2, x1:x2]

def grab(video, t, prior, label):
    cap = cv2.VideoCapture(os.path.join(INPUT_DIR, f"{video}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read(); cap.release()
    if not ret: return
    h, w = frame.shape[:2]
    t160 = crop(frame, prior[0], prior[1], PRIOR_ROI_PX, w, h)
    t320 = crop(frame, prior[0], prior[1], EXPAND, w, h)
    cv2.imwrite(os.path.join(OUT, f"{video}_{label}_t{t:.0f}_prior{prior[0]}_{prior[1]}_roi160.png"), t160)
    cv2.imwrite(os.path.join(OUT, f"{video}_{label}_t{t:.0f}_prior{prior[0]}_{prior[1]}_roi320.png"), t320)

# 09: 当前 prior(0.2,0.35) + 真灯位(0.50,0.10); suspected [72-106] 中段与 GT green [11-72] 段
for t in (20, 40, 80, 90, 95, 100):
    grab("违章09", t, (0.2, 0.35), "cur")
    grab("违章09", t, (0.50, 0.10), "true")
print("crops written to", OUT)
print(sorted(os.listdir(OUT)))
