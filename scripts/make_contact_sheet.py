"""从标注视频里按信号灯时间线抽取关键帧，拼成联系表图片（替代视频预览）。

- 违章01: 抽 green/flashing 相位帧（最该出违规的时刻）
- 违章02: 无 green(=E16 bug), 改为抽有活跃车辆且有信号灯状态的帧, 证明"检不到绿灯"
每个单元格标注 时间戳(s) + 信号灯状态 + 该帧最大压线 overlap。
"""
import csv
import os

import cv2
import numpy as np

BASE = "D:/redlight-crosswalk-violation"
OUT_DIR = os.path.join(BASE, "data", "output", "contact_sheets")
os.makedirs(OUT_DIR, exist_ok=True)

COLS, ROWS = 4, 4
THUMB_W, THUMB_H = 360, 202  # 16:9
N = COLS * ROWS


def pick_frames(vid, diag_csv, max_n=N):
    rows = list(csv.DictReader(open(diag_csv, encoding="utf-8")))
    # 优先 green/flashing
    cand = [r for r in rows if r["light_state"] in ("green", "flashing")]
    if not cand:
        # 退而求其次: 有活跃车 + 有信号灯状态(非全unknown空帧)
        cand = [r for r in rows if int(r["n_active"]) > 0
                and r["light_state"] in ("red", "unknown")]
    if not cand:
        cand = rows
    # 均匀抽 max_n 帧
    if len(cand) > max_n:
        step = len(cand) / max_n
        cand = [cand[int(i * step)] for i in range(max_n)]
    return cand


def build(vid, annotated_mp4, diag_csv):
    cap = cv2.VideoCapture(annotated_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = pick_frames(vid, diag_csv)
    grid = np.full((THUMB_H * ROWS, THUMB_W * COLS, 3), 18, dtype=np.uint8)
    for i, r in enumerate(frames):
        if i >= N:
            break
        ts = float(r["ts"])
        fi = int(round(ts * fps))
        if fi >= total:
            fi = total - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        f = cv2.resize(frame, (THUMB_W, THUMB_H))
        # 标签
        label = f"{ts:.1f}s|{r['light_state']}|ov={float(r['max_overlap']):.2f}|n={r['n_active']}"
        cv2.rectangle(f, (0, 0), (THUMB_W, 18), (0, 0, 0), -1)
        cv2.putText(f, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 255, 255), 1, cv2.LINE_AA)
        rr = i // COLS
        cc = i % COLS
        grid[rr * THUMB_H:(rr + 1) * THUMB_H, cc * THUMB_W:(cc + 1) * THUMB_W] = f
    cap.release()
    out = os.path.join(OUT_DIR, f"contact_{vid}.jpg")
    cv2.imwrite(out, grid)
    print("wrote", out, "frames=", len(frames))


if __name__ == "__main__":
    build("违章01",
          os.path.join(BASE, "data/output/run_v10_01/annotated.mp4"),
          os.path.join(BASE, "data/output/diag_违章01_timeline.csv"))
    build("违章02",
          os.path.join(BASE, "data/output/run_v10_02/annotated.mp4"),
          os.path.join(BASE, "data/output/diag_违章02_timeline.csv"))
