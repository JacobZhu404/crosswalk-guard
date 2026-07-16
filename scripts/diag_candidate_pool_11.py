"""Step 0 只读前置检查 (2026-07-16, 对应 cc 复核阻塞1):

不写码之前的决定性检查 —— 回答"风险 A: 11 漏检是'候选缺失'还是'选择输了'"。

忠实复现 crosswalk.py:_cv_v11 的候选生成段 (行 61-151), 不改生产代码。
对 11 视频绿灯窗 (ts 20.0-28.3, cc 确认的 63 帧) 每帧, 收集所有候选条带的
(abs_y1, abs_y2, score), 统计有多少帧的候选池里存在与真斑马线 Y 区间 [373,463]
重叠的条纹带:

  - 多数帧有该候选 -> 是"选择输了" -> v3 占道池化对症, 可放行写码
  - 多数帧无该候选 -> 是"候选缺失" -> occ 池化只能救被选中的少数帧, 停, 升级给 Jacob

注意: 候选生成段 (行 81-151) 中车辆加分只影响 score, 不影响候选的 (abs_y1,abs_y2),
因此此处不传 vehicle_boxes, 纯看条纹候选是否生成 —— 恰好隔离"候选缺失 vs 选择输了"。
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

VIDEO = "违章11"
GREEN_T0, GREEN_T1 = 20.0, 28.3   # 绿灯窗 (cc 确认, ts 区间)
TRUE_BAND = (373, 463)            # 真斑马线 Y 区间 (cc 确认, 静止车脚下)
OUT_CSV = "/tmp/diag_04_11/违章11_candidate_pool.csv"


def gen_candidates(frame):
    """忠实复现 _cv_v11:61-151 的候选生成, 返回 [(score,abs_y1,abs_y2,run_len),...]。

    不调用生产 _cv_v11, 纯复制逻辑以保证候选 (abs_y1,abs_y2) 与生产完全一致;
    不传 vehicle_boxes (行 140-147 跳过), 因此 score 仅反映纯条纹纹理,
    但候选的 Y 区间与带车辆框时完全相同。
    """
    h, w = frame.shape[:2]
    if h < 120 or w < 160:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.abs(sobely)

    BAND_FRAC = 0.30
    STEP_FRAC = 0.12
    MIN_RUN_FRAC = 0.06
    band_h = max(40, int(h * BAND_FRAC))
    step = max(20, int(h * STEP_FRAC))
    min_run_px = max(5, int(band_h * MIN_RUN_FRAC))

    candidates = []
    for y0 in range(0, h - band_h + 1, step):
        y1 = min(y0 + band_h, h)
        actual_h = y1 - y0

        row_density = np.mean(grad_mag[y0:y1, :], axis=1)
        ks = min(15, actual_h // 4)
        if ks % 2 == 0:
            ks += 1
        kernel = np.ones(ks) / ks
        smoothed = np.convolve(row_density, kernel, mode='same')

        mean_val = np.mean(smoothed)
        std_val = np.std(smoothed)
        if std_val < 1e-6:
            continue
        thresh = mean_val + 1.5 * std_val
        candidate_rows = smoothed > thresh

        runs = []
        in_run = False
        start = 0
        for i, val in enumerate(candidate_rows):
            if val and not in_run:
                start = i
                in_run = True
            elif not val and in_run:
                runs.append((start, i))
                in_run = False
        if in_run:
            runs.append((start, len(candidate_rows)))

        if not runs:
            continue
        best_run = max(runs, key=lambda r: (r[1] - r[0]))
        run_len = best_run[1] - best_run[0]
        if run_len < min_run_px:
            continue

        run_slice = frame[y0 + best_run[0]:y0 + best_run[1], :]
        run_gray = cv2.cvtColor(run_slice, cv2.COLOR_BGR2GRAY) \
            if len(run_slice.shape) == 3 else run_slice
        avg_brightness = np.mean(run_gray)
        if avg_brightness < 100:
            continue

        run_mean_grad = np.mean(smoothed[best_run[0]:best_run[1]])
        score = float(run_len) * (run_mean_grad - mean_val) / std_val + avg_brightness / 255.0

        abs_y1 = y0 + best_run[0]
        abs_y2 = y0 + best_run[1]
        candidates.append((round(score, 3), int(abs_y1), int(abs_y2), int(run_len)))
    return candidates


def overlaps_true_band(cands):
    t1, t2 = TRUE_BAND
    return [c for c in cands if c[1] <= t2 and c[2] >= t1]


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    fps = cfg.inference.fps
    cap = cv2.VideoCapture(os.path.join(ROOT, "input_video", f"{VIDEO}.mp4"))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(vfps / fps)))

    rows = []
    total = 0
    with_true = 0
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = fi / vfps
        if GREEN_T0 <= ts <= GREEN_T1 and fi % interval == 0:
            cands = gen_candidates(frame)
            ov = overlaps_true_band(cands)
            total += 1
            has = len(ov) > 0
            if has:
                with_true += 1
            top = max(cands, key=lambda c: c[0]) if cands else None
            rows.append({
                "ts": round(ts, 2),
                "frame": fi,
                "n_cands": len(cands),
                "has_true_cand": int(has),
                "n_true_cand": len(ov),
                "true_cand_y": ";".join(f"{c[1]}-{c[2]}" for c in ov),
                "top_score": round(top[0], 3) if top else 0,
                "top_cy": f"{top[1]}-{top[2]}" if top else "",
            })
        fi += 1
    cap.release()

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n===== {VIDEO} 绿灯窗候选池 dump (Step 0 / 风险 A) =====")
    print(f"窗口 ts∈[{GREEN_T0},{GREEN_T1}]  帧间隔={interval}  采样帧={total}")
    print(f"候选池含真带(373-463)候选的帧: {with_true}/{total} ({100.0*with_true/total:.1f}%)")
    if total > 0 and with_true >= 0.5 * total:
        print(">>> 结论: 多数帧'有真带候选' -> 是'选择输了' -> v3 占道池化对症, 可放行写码")
    else:
        print(">>> 结论: 多数帧'无真带候选' -> 是'候选缺失' -> occ 池化救不了, 停, 升级给 Jacob")
    print(f"[CSV] {OUT_CSV}")


if __name__ == "__main__":
    main()
