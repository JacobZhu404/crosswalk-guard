"""诊断 v6 灯态检测器偏向 green 的根因。

对指定视频逐采样帧打印: 预测灯态(GT), 以及当前持久信号轨迹的颜色/面积/位置/是否被点亮。
重点看:
  - GT=red 的帧, 检测器是否检到 red 轨迹? 是否被 green 轨迹抢走?
  - green 轨迹是真实信号灯, 还是草地/车漆/反光?
"""
import sys
import os
import csv
import glob

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
from redlight.infrastructure.config import load_config, project_root
from redlight.models.traffic_light import TrafficLightDetector


SAMPLE_FPS = 8


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "违章02"
    V = os.path.join(ROOT, "input_video", f"{name}.mp4")
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    cap = cv2.VideoCapture(V)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / SAMPLE_FPS)))

    # GT 段
    gt = []
    with open(os.path.join(ROOT, "datasets", "gt", "events.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["video"] == name:
                gt.append((float(r["start_s"]), float(r["end_s"]), r["light_state"]))

    def gt_at(ts):
        for a, b, s in gt:
            if a <= ts <= b:
                return s
        return "unknown"

    det = TrafficLightDetector(cfg, verbose=False)
    fi = 0
    print(f"===== {name} (fps={fps:.1f}, interval={interval}) =====")
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if fi % interval == 0:
            ts = fi / fps
            res = det.detect(fr)
            pred = res["state"]
            g = gt_at(ts)
            # 持久轨迹摘要
            tr_info = []
            for t in det.tracks:
                if t["frames_seen"] >= 3:
                    tr_info.append(f"{t['last_color']}(a={t.get('last_area',0):.0f},"
                                   f"cy={t['cy']:.2f},fs={t['frames_seen']},"
                                   f"lit={int(t.get('lit_this_frame',False))},"
                                   f"ss={len(t.get('states_seen',set()))})")
            flag = "  <-- mismatch" if pred != g and g != "unknown" else ""
            print(f"[{ts:6.1f}s] pred={pred:8s} gt={g:8s}{flag}  tracks={tr_info}")
        fi += 1
    cap.release()


if __name__ == "__main__":
    main()
