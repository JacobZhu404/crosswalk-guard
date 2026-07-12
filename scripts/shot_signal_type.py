"""信号类型诊断: 只抽关键帧截图(ASCII文件名, 避免GBK乱码), 不编码整段视频。

每帧画: 候选框 + 持久轨迹 + 黄色READING圈 + 青色ANCHOR框 + 顶部state文字。
截图名 shot_<视频名>_<t*10>.png, 便于直接 Read 肉眼判断锁的是行人信号(走路/站立人图标)
还是机动车信号(车/箭头图标), 以及相位是否与 GT 行人灯态一致。
"""
import sys, os, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from redlight.models.traffic_light import TrafficLightDetector
from redlight.infrastructure.config import load_config


def shot(video, out_dir, max_sec=90, every_shot=4.0, draw_min_area=15):
    cfg = load_config("configs/config.yaml")
    det = TrafficLightDetector(cfg, verbose=False)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(out_dir, exist_ok=True)
    import re
    vname_raw = os.path.splitext(os.path.basename(video))[0]
    digits = re.sub(r'\D', '', vname_raw)
    vname = f"v{digits}" if digits else re.sub(r'[^A-Za-z0-9]', '_', vname_raw)
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

        for tr in tracks:
            if tr["frames_seen"] < 3:
                continue
            cx, cy = int(tr["cx"] * W), int(tr["cy"] * H)
            side = max(6, int((tr.get("last_area", 0) ** 0.5)))
            if tr["last_color"] == "green":
                tcol = (0, 255, 0)
            elif tr["last_color"] == "red":
                tcol = (0, 0, 255)
            else:
                tcol = (200, 200, 200)
            cv2.circle(frame, (cx, cy), max(4, side // 2), tcol, 2)

        if best is not None:
            cx, cy = int(best["cx"] * W), int(best["cy"] * H)
            cv2.circle(frame, (cx, cy), 34, (0, 255, 255), 3)
            cv2.putText(frame, "READING", (cx - 30, cy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        anc = res.get("anchor")
        if anc and anc.get("cx") is not None:
            ax, ay = int(anc["cx"] * W), int(anc["cy"] * H)
            s = 48
            cv2.rectangle(frame, (ax - s, ay - s), (ax + s, ay + s), (255, 255, 0), 2)
            cv2.putText(frame, f"ANCHOR:{anc.get('last_dom', '-')}",
                        (ax - 46, ay - s - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        sel_col = best["last_color"] if best else "-"
        sel_area = int(best["last_area"]) if best else 0
        ng = sum(1 for c in cands if c["color"] == "green" and c["area"] >= draw_min_area)
        nr = sum(1 for c in cands if c["color"] == "red" and c["area"] >= draw_min_area)
        txt = (f"t={t:5.1f}s  state={res['state']:8s}  reason={res['reason']}"
               f"  boxG={ng} boxR={nr}  sel={sel_col}({sel_area})")
        cv2.rectangle(frame, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(frame, txt, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if t - last_shot >= every_shot:
            last_shot = t
            fname = f"shot_{vname}_{int(t * 10):05d}.png"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            axp = anc.get("cx", -1.0) if anc else -1.0
            ayp = anc.get("cy", -1.0) if anc else -1.0
            print(f"[shot] {fname} state={res['state']} anchor=({axp:.2f},{ayp:.2f})",
                  flush=True)
        fi += 1
    cap.release()
    print(f"frames={fi} fps={fps:.1f} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="data/output/annotated/shots")
    ap.add_argument("--sec", type=float, default=90.0)
    ap.add_argument("--every", type=float, default=4.0)
    args = ap.parse_args()
    shot(args.video, args.out, max_sec=args.sec, every_shot=args.every)
