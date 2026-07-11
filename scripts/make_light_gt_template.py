"""生成红绿灯状态 GT 标注模板 (要求#6 前置步骤)。

读取 diag_*_timeline.csv (逐帧预测), 把"连续相同预测状态"切成片段,
每段抽取一张缩略图, 输出可供人工标注真值的 CSV 模板:
  segment_id, start_ts, end_ts, predicted_state, predicted_reason,
  frame_count, mid_ts, thumb, gt_state(空), note(空)

用法:
    python scripts/make_light_gt_template.py
"""
import sys
import os
import csv

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import cv2

BASE_IN = r"E:\BaiduNetdiskDownload"
DIAG_DIR = os.path.join(ROOT, "data", "output")
GT_DIR = os.path.join(ROOT, "datasets", "gt", "light_state")


def build_segments(diag_csv):
    segs = []
    cur = None  # {state, reason_counter, rows:[(ts,reason)]}
    with open(diag_csv, newline="") as f:
        for r in csv.DictReader(f):
            st = r["light_state"]
            reason = r["light_reason"]
            ts = float(r["ts"])
            if cur is None or cur["state"] != st:
                if cur is not None:
                    segs.append(cur)
                cur = {"state": st, "rows": [(ts, reason)], "reasons": {}}
            else:
                cur["rows"].append((ts, reason))
            cur["reasons"][reason] = cur["reasons"].get(reason, 0) + 1
    if cur is not None:
        segs.append(cur)
    out = []
    for i, s in enumerate(segs):
        rows = s["rows"]
        start_ts = rows[0][0]
        end_ts = rows[-1][0]
        mid_ts = (start_ts + end_ts) / 2.0
        dom_reason = max(s["reasons"].items(), key=lambda x: x[1])[0]
        out.append({
            "segment_id": i + 1,
            "start_ts": round(start_ts, 2),
            "end_ts": round(end_ts, 2),
            "predicted_state": s["state"],
            "predicted_reason": dom_reason,
            "frame_count": len(rows),
            "mid_ts": round(mid_ts, 2),
            "thumb": "",
            "gt_state": "",
            "note": "",
        })
    return out


def extract_thumb(video, mid_ts, out_path):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, mid_ts * 1000.0)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    cv2.imwrite(out_path, frame)
    return True


def main():
    os.makedirs(GT_DIR, exist_ok=True)
    for fname in ("违章01.mp4", "违章02.mp4"):
        name = os.path.splitext(fname)[0]
        diag_csv = os.path.join(DIAG_DIR, f"diag_{name}_timeline.csv")
        video = os.path.join(BASE_IN, fname)
        if not os.path.exists(diag_csv):
            print(f"[SKIP] 缺诊断文件 {diag_csv}")
            continue
        segs = build_segments(diag_csv)
        thumb_dir = os.path.join(GT_DIR, name)
        os.makedirs(thumb_dir, exist_ok=True)
        for s in segs:
            thumb_path = os.path.join(thumb_dir, f"seg_{s['segment_id']:03d}.jpg")
            if extract_thumb(video, s["mid_ts"], thumb_path):
                s["thumb"] = os.path.relpath(thumb_path, GT_DIR).replace("\\", "/")
        gt_csv = os.path.join(GT_DIR, f"{name}_gt.csv")
        with open(gt_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["segment_id", "start_ts", "end_ts",
                "predicted_state", "predicted_reason", "frame_count", "mid_ts",
                "thumb", "gt_state", "note"])
            w.writeheader()
            w.writerows(segs)
        print(f"[OK] {name}: {len(segs)} 段 -> {gt_csv} (缩略图在 {thumb_dir})")


if __name__ == "__main__":
    main()
