"""从标注视频(含 v10 掩膜叠加)里抽某时间窗口的帧, 拼成联系表图片。

用于肉眼核对: 黄色斑马线掩膜 是否对到了"占用斑马线的车"上。
用法:
    python scripts/extract_window_sheet.py <annotated.mp4> <start_s> <end_s> <out.jpg> [n]
"""
import os
import sys

import cv2
import numpy as np

if len(sys.argv) < 5:
    print("usage: extract_window_sheet.py <annotated.mp4> <start_s> <end_s> <out.jpg> [n=12]")
    sys.exit(1)

VIDEO = sys.argv[1]
S, E = float(sys.argv[2]), float(sys.argv[3])
OUT = sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 12

COLS, ROWS = 4, 3
TW, TH = 400, 225
grid = np.full((TH * ROWS, TW * COLS, 3), 18, dtype=np.uint8)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(N):
    ts = S + (E - S) * (i + 0.5) / N
    fi = int(round(ts * fps))
    if fi >= total:
        fi = total - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue
    f = cv2.resize(frame, (TW, TH))
    cv2.rectangle(f, (0, 0), (TW, 18), (0, 0, 0), -1)
    cv2.putText(f, f"{ts:.1f}s", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1, cv2.LINE_AA)
    r, c = divmod(i, COLS)
    grid[r * TH:(r + 1) * TH, c * TW:(c + 1) * TW] = f
cap.release()
cv2.imwrite(OUT, grid)
print("wrote", OUT)
