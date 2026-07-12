"""逐帧可视化红绿灯检测 — 把检测器"看到"的灯框在视频上。

输出:
  data/output/annotated/<video>_lightboxes.mp4   带框标注视频
  data/output/annotated/<video>_shots/          每 ~5s 截图(png)

绘制内容:
  1. 原始候选亮斑 (candidates): 绿/红框 + 面积 -> 看"灯找没找对位置/颜色判对没"
  2. 跟踪轨迹 (tracks): 圆点 + 持久度(frames_seen) + 最后颜色 -> 看哪盏灯被持续跟踪
  3. 选中轨迹 (track): 黄色圆圈 -> 看最终驱动判定的那盏灯 (READING)
  3b. 空间锚 (anchor): 青色方框 -> 稳定器锁定的信号灯位置, 应全程不动(防衣服/树/倒计时劫持)
  4. 左上角文字: 时刻 / 最终state / reason / 候选数 / 轨迹数
"""
import sys, os, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from redlight.models.traffic_light import TrafficLightDetector
from redlight.infrastructure.config import load_config


def draw(video, out_dir, max_sec=45, every_shot=5.0, draw_min_area=15, prior=None, prior_roi=None):
    cfg = load_config("configs/config.yaml")
    det = TrafficLightDetector(cfg, verbose=False)
    if prior is not None:
        # 行人信号位置先验(由 identify_pedestrian_signal.py 用 GT 标定得出): 仅在此附近选灯
        det.signal_prior = tuple(prior)
        print(f"[prior] 锁定行人信号 @ {prior}")
    if prior_roi is not None:
        det.prior_roi_px = int(prior_roi)
        print(f"[prior] ROI = {prior_roi}px")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(out_dir, exist_ok=True)
    shot_dir = os.path.join(out_dir, "shots")
    os.makedirs(shot_dir, exist_ok=True)

    vname = os.path.splitext(os.path.basename(video))[0]
    out_path = os.path.join(out_dir, f"{vname}_lightboxes.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    fi = 0
    last_shot = -10
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if t > max_sec:
            break
        res = det.detect(frame)
        cands = res.get("candidates", [])
        tracks = det.tracks
        best = res.get("track")

        # --- 1. 原始候选 ---
        for c in cands:
            if c["area"] < draw_min_area:
                continue
            x1, y1, x2, y2 = c["box"]
            if c["color"] == "green":
                col = (0, 255, 0)
            elif c["color"] == "red":
                col = (0, 0, 255)
            else:
                col = (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1)
            cv2.putText(frame, f"{c['color']}:{c['area']}",
                        (x1, max(0, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # --- 2. 跟踪轨迹 (每个持久候选信号头, 标注颜色字母+持久度) ---
        for tr in tracks:
            if tr["frames_seen"] < 3:
                continue
            cx, cy = int(tr["cx"] * W), int(tr["cy"] * H)
            side = max(6, int((tr.get("last_area", 0) ** 0.5)))
            emitting = tr.get("max_bright", 0.0) > 0.0
            if tr["last_color"] == "green":
                tcol = (0, 255, 0)
            elif tr["last_color"] == "red":
                tcol = (0, 0, 255)
            else:
                tcol = (200, 200, 200)
            # 真发射(点亮)的轨迹画实线粗圈, 仅反射/暗信号的画细圈, 便于区分
            cv2.circle(frame, (cx, cy), max(4, side // 2),
                       tcol, 2 if emitting else 1)
            letter = {"green": "G", "red": "R"}.get(tr["last_color"], "?")
            cv2.putText(frame, f"{letter} fs{tr['frames_seen']}",
                        (cx + side // 2 + 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, tcol, 1)

        # --- 3. 选中信号头 (黄色圆圈) ---
        if best is not None:
            cx, cy = int(best["cx"] * W), int(best["cy"] * H)
            cv2.circle(frame, (cx, cy), 34, (0, 255, 255), 3)
            cv2.putText(frame, "READING", (cx - 30, cy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # --- 3b. 空间锚(稳定器): 锁定静态信号灯的位置(青色方框) ---
        anc = res.get("anchor")
        if anc and anc.get("cx") is not None:
            ax, ay = int(anc["cx"] * W), int(anc["cy"] * H)
            s = 48
            cv2.rectangle(frame, (ax - s, ay - s), (ax + s, ay + s),
                          (255, 255, 0), 2)
            cv2.putText(frame, f"ANCHOR:{anc.get('last_dom', '-')}",
                        (ax - 46, ay - s - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # --- 4. 文字 ---
        sel_col = best["last_color"] if best else "-"
        sel_area = int(best["last_area"]) if best else 0
        ng = sum(1 for c in cands if c["color"] == "green" and c["area"] >= draw_min_area)
        nr = sum(1 for c in cands if c["color"] == "red" and c["area"] >= draw_min_area)
        txt = (f"t={t:5.1f}s  state={res['state']:8s}  reason={res['reason']}"
               f"  boxG={ng} boxR={nr}  sel={sel_col}({sel_area})  tracks={len(tracks)}")
        cv2.rectangle(frame, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(frame, txt, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        vw.write(frame)
        if t - last_shot >= every_shot:
            last_shot = t
            cv2.imwrite(os.path.join(shot_dir, f"{vname}_{t:05.1f}s.png"), frame)
        fi += 1

    cap.release()
    vw.release()
    print(f"[OK] video : {out_path}")
    print(f"[OK] shots : {shot_dir}")
    print(f"frames={fi} fps={fps:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="input_video/违章07.mp4")
    ap.add_argument("--out", default="data/output/annotated")
    ap.add_argument("--sec", type=float, default=45.0)
    ap.add_argument("--prior", type=float, nargs=2, default=None,
                    metavar=("CX", "CY"),
                    help="行人信号位置先验(归一化 cx cy), 仅在此附近选灯, 永不重锚")
    ap.add_argument("--roi", type=int, default=None,
                    help="HSV直采ROI边长px(小灯视频加大, 如04用220)")
    args = ap.parse_args()
    draw(args.video, args.out, max_sec=args.sec, prior=args.prior, prior_roi=args.roi)
