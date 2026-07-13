"""将用户标注反馈(plate_feedback.csv)转换为评测集格式。

用途: 用户在画廊中标注的结果(包含正确车牌corrected_plate)可直接作为回归测试集。
形成 标注→评测集→算法迭代 的闭环。

输出格式:
  datasets/plate_eval_set/
    meta.csv          # 索引文件: video, frame_idx, detected, corrected_plate, verdict, reason
    images/
      违章01_000123.jpg   # 关键帧截图

用法:
  python scripts/feedback_to_eval_set.py
"""
import os
import sys
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2


def load_feedback(path):
    d = []
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d.append({
                "video": r.get("video", ""),
                "frame_idx": int(r.get("frame_idx", 0)),
                "t_sec": float(r.get("t", 0)),
                "detected": r.get("detected", ""),
                "gt": r.get("gt", ""),
                "verdict": r.get("verdict", ""),
                "reason": r.get("reason", ""),
                "corrected_plate": r.get("corrected_plate", ""),
                "note": r.get("note", ""),
            })
    return d


def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def main():
    ap = argparse.ArgumentParser(description="将标注反馈转换为评测集")
    ap.add_argument("--feedback", default=os.path.join(ROOT, "data", "output", "annotated", "plate_feedback.csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "datasets", "plate_eval_set"))
    ap.add_argument("--video-dir", default=os.path.join(ROOT, "input_video"))
    args = ap.parse_args()

    feedback = load_feedback(args.feedback)
    if not feedback:
        print(f"警告: 未找到标注反馈文件 {args.feedback}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    meta_rows = []
    saved_count = 0

    for item in feedback:
        video = item["video"]
        frame_idx = item["frame_idx"]
        detected = item["detected"]
        corrected_plate = item["corrected_plate"]
        verdict = item["verdict"]
        reason = item["reason"]

        if not corrected_plate and verdict != "too_hard":
            continue

        video_path = os.path.join(args.video_dir, f"{video}.mp4")
        if not os.path.exists(video_path):
            print(f"警告: 视频不存在 {video_path}")
            continue

        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            print(f"警告: 无法提取帧 {video} frame_{frame_idx}")
            continue

        img_name = f"{video}_{frame_idx:06d}.jpg"
        img_path = os.path.join(images_dir, img_name)
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            with open(img_path, "wb") as f:
                f.write(buf.tobytes())
            saved_count += 1

        meta_rows.append({
            "video": video,
            "frame_idx": frame_idx,
            "image_path": f"images/{img_name}",
            "detected": detected,
            "corrected_plate": corrected_plate,
            "verdict": verdict,
            "reason": reason,
            "note": item.get("note", ""),
        })

    meta_path = os.path.join(args.out_dir, "meta.csv")
    with open(meta_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame_idx", "image_path", "detected",
                                         "corrected_plate", "verdict", "reason", "note"])
        w.writeheader()
        w.writerows(meta_rows)

    print(f"[OK] 评测集已生成 -> {args.out_dir}")
    print(f"  标注帧数: {len(feedback)}")
    print(f"  有效评测样本: {len(meta_rows)}")
    print(f"  保存图像: {saved_count}")
    print(f"  索引文件: {meta_path}")


if __name__ == "__main__":
    main()