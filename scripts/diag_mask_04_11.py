"""Step 1 只读诊断 (2026-07-16): 定位 04/11 占道 max_overlap 恒为 0 的根因。

忠实复现 dag.py 的 detect->track->crosswalk(带车辆框)->accumulate 路径,
只观测、不改代码。核心要分清两种可能:

  (a) mask 全空 (v11 在 04/11 泛化失败) -> mask_area=0 -> compute_overlap_ratio 返回 0.0
  (b) mask 有内容但错位 (坐标/车辆框对不上) -> overlap 也 0, 但原因不同

逐采样帧记录:
  - mask 非零像素数 (mask_nonzero)
  - mask 非零区 bbox (cy1,cy2,cx1,cx2) —— 空则为 None
  - 每个 active 车辆框的 overlap (footprint=0.5, denom=mask)
  - 是否存在静止车辆 (has_stationary)
落图: 每视频若干帧 (原图 + 斑马线掩膜 overlay + 车辆框), 便于肉眼判断"空 vs 错位"。

用法:
    python scripts/diag_mask_04_11.py
输出:
    /tmp/diag_04_11/<video>_timeline.csv
    /tmp/diag_04_11/<video>_<frame>_mask.jpg  (样本)
"""
import os
import sys
import csv
import cv2
import numpy as np

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config, project_root
from redlight.models.vehicle import VehicleDetector
from redlight.models.crosswalk import CrosswalkDetector
from redlight.pipeline.tracker import TrackStateManagerV2, SENSITIVITY_PRESETS
from redlight.infrastructure.geometry import compute_overlap_ratio

VIDEOS = ["违章04", "违章11"]
PRESET = "balanced"
OUT_DIR = "/tmp/diag_04_11"


def mask_bbox(mask):
    """返回非零区 (cy1,cy2,cx1,cx2) 或 None。"""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))


def diagnose(video_name, cfg):
    video_path = os.path.join(ROOT, "input_video", f"{video_name}.mp4")
    if not os.path.isfile(video_path):
        print(f"[SKIP] 找不到 {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] 无法打开 {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    interval = max(1, int(round(fps / cfg.inference.fps)))

    det = VehicleDetector(cfg, verbose=False)
    cw = CrosswalkDetector(cfg)
    track = TrackStateManagerV2(PRESET)
    overlap_thr = SENSITIVITY_PRESETS[PRESET]["overlap"]

    rows = []
    frame_idx = 0
    proc = 0
    n_empty = 0
    n_mask = 0
    n_mask_with_stationary = 0
    n_mask_with_stationary_but_zero_ov = 0
    mask_nonzero_list = []
    saved_samples = 0
    max_samples = 8
    sample_dir = os.path.join(OUT_DIR, video_name)
    os.makedirs(sample_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            proc += 1
            dets = det.detect(frame)
            states = track.update(dets, frame_idx / fps)
            vb = [d["xyxy"] for d in dets]
            mask = cw.detect(frame, vb)

            mnon = int(np.count_nonzero(mask > 0)) if mask is not None else 0
            mbbox = mask_bbox(mask) if mask is not None else None
            mask_nonzero_list.append(mnon)
            if mnon == 0:
                n_empty += 1
            else:
                n_mask += 1

            has_stationary = False
            max_ov = 0.0
            boxes_ov = []
            for s in states.values():
                if not s.get("active"):
                    continue
                ov = compute_overlap_ratio(s["box"], mask, footprint=0.5, denom="mask") if mask is not None else 0.0
                max_ov = max(max_ov, ov)
                boxes_ov.append((s["box"], round(ov, 3), bool(s.get("stationary", False))))
                if s.get("stationary", False):
                    has_stationary = True

            if mnon > 0 and has_stationary:
                n_mask_with_stationary += 1
                if max_ov == 0.0:
                    n_mask_with_stationary_but_zero_ov += 1

            rows.append({
                "ts": round(frame_idx / fps, 2),
                "frame": frame_idx,
                "mask_nonzero": mnon,
                "mask_cy1": mbbox[0] if mbbox else "",
                "mask_cy2": mbbox[1] if mbbox else "",
                "mask_cx1": mbbox[2] if mbbox else "",
                "mask_cx2": mbbox[3] if mbbox else "",
                "n_active": sum(1 for s in states.values() if s.get("active")),
                "n_stationary": sum(1 for s in states.values() if s.get("active") and s.get("stationary")),
                "max_overlap": round(max_ov, 3),
                "overlap_thr": overlap_thr,
            })

            # 落样本图: 优先"有掩膜+有静止车但 overlap=0"(最可能是错位), 其次随机带掩膜的
            if saved_samples < max_samples and mbbox is not None:
                want = (has_stationary and max_ov == 0.0) or (saved_samples < max_samples // 2)
                if want:
                    disp = frame.copy()
                    # 画掩膜 overlay (青色)
                    ov = (mask > 0)
                    disp[ov, 0] = np.minimum(disp[ov, 0].astype(int) + 120, 255).astype(np.uint8)
                    # 画车辆框
                    for b, ovv, stt in boxes_ov:
                        x1, y1, x2, y2 = [int(v) for v in b]
                        col = (0, 0, 255) if stt else (0, 255, 0)
                        cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
                        cv2.putText(disp, f"{ovv:.2f}", (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    fn = f"{video_name}_{frame_idx}_mask.jpg"
                    cv2.imwrite(os.path.join(sample_dir, fn), disp)
                    saved_samples += 1

        frame_idx += 1

    cap.release()

    csv_path = os.path.join(OUT_DIR, f"{video_name}_timeline.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mnl = np.array(mask_nonzero_list)
    print(f"\n===== {video_name} =====")
    print(f"时长~{total/fps:.0f}s  fps={fps:.1f}  interval={interval}  推理帧={proc}")
    print(f"mask 全空帧: {n_empty}/{proc} ({100.0*n_empty/proc:.1f}%)")
    print(f"mask 有内容帧: {n_mask}/{proc} ({100.0*n_mask/proc:.1f}%)")
    if len(mnl) > 0:
        print(f"mask_nonzero 像素: min={mnl.min()}  median={int(np.median(mnl))}  max={mnl.max()}")
    print(f"有掩膜且有静止车的帧: {n_mask_with_stationary}")
    print(f"  -> 其中 overlap 仍=0 (疑似错位): {n_mask_with_stationary_but_zero_ov}")
    print(f"样本图: {sample_dir}/  (共 {saved_samples} 张)")
    print(f"[CSV] {csv_path}")


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    os.makedirs(OUT_DIR, exist_ok=True)
    for v in VIDEOS:
        diagnose(v, cfg)


if __name__ == "__main__":
    main()
