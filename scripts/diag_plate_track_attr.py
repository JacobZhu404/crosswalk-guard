"""qw 车牌诊断: 事件车牌归因工具(cc 硬条件③ — 定位"事件回填的牌来自哪个 track")。

用法:
  python scripts/diag_plate_track_attr.py [--video 违章02] [--plate 京A14672]

输出:
  - 事件结构(代表 track / member_tracks / 窗口)
  - 目标车牌(默认回填牌)每帧读取归属 tid(plate 框中心与车辆框时空匹配)
  - 关键 tid 的轨迹摘要(质心/时间窗) —— 区分白车/过路车
"""
import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.app import cli
from redlight.models.plate import PlateRecognizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="违章02")
    ap.add_argument("--plate", default=None, help="要归因的车牌(默认取事件回填的牌)")
    args = ap.parse_args()

    cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
    cfg.output.annotated_video = False
    cfg.output.evidence_images = False
    video_path = os.path.join(ROOT, "input_video", f"{args.video}.mp4")

    events, track_samples = cli.run(cfg, video_path,
                                    os.path.join(ROOT, "data", "output", "qw", f"attr_{args.video}"),
                                    preset="balanced", return_track_samples=True)

    # 事件结构
    for ev in events:
        if ev["status"] != "confirmed":
            continue
        print(f"事件 track_id={ev['track_id']} 窗口=[{ev['start_ts']:.2f},{ev['end_ts']:.2f}]")
        print(f"  member_tracks({len(ev.get('member_tracks', []))}): {ev.get('member_tracks')}")
        print(f"  回填 plate: {ev.get('plate', '')}")

    # 重采 plate 帧(带 box)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / cfg.inference.fps)))
    plate = PlateRecognizer(cfg, verbose=False)
    targets = set()
    if args.plate:
        targets.add(args.plate)
    else:
        for ev in events:
            if ev["status"] == "confirmed" and ev.get("plate"):
                targets.add(ev["plate"])
    plate_frames = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % interval == 0:
            ts = fi / fps
            for p in plate.detect(frame):
                t = p.get("text", "")
                if t in targets:
                    plate_frames.append((ts, t, p.get("conf", 0.0),
                                         [int(v) for v in p.get("xyxy", [0, 0, 0, 0])]))
        fi += 1
    cap.release()

    def in_box(px, py, b):
        return b[0] - 5 <= px <= b[2] + 5 and b[1] - 5 <= py <= b[3] + 5

    from collections import defaultdict
    assign = defaultdict(lambda: defaultdict(int))
    unmatched = defaultdict(int)
    for ts, text, conf, box in plate_frames:
        px, py = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        hit = None
        for tid, samples in track_samples.items():
            for s in samples:
                if abs(s["ts"] - ts) < 0.3 and in_box(px, py, s["box"]):
                    hit = tid
                    break
            if hit:
                break
        if hit:
            assign[text][hit] += 1
        else:
            unmatched[text] += 1

    print("\n=== 目标车牌读取帧的 tid 归属 ===")
    for text in sorted(targets):
        print(f"{text}: {len([1 for f in plate_frames if f[1] == text])} 帧, 未匹配={unmatched.get(text, 0)}")
        for tid, c in sorted(assign[text].items(), key=lambda x: -x[1])[:6]:
            print(f"   tid {tid}: {c} 帧")

    print("\n=== 相关 tid 轨迹摘要 ===")
    member = set()
    for ev in events:
        member.update(ev.get("member_tracks", []))
    seen_tids = set()
    for text in assign:
        seen_tids.update(assign[text].keys())
    for tid in sorted(seen_tids | member):
        samples = track_samples.get(tid, [])
        if not samples:
            continue
        ts0, ts1 = samples[0]["ts"], samples[-1]["ts"]
        cx = sum((s["box"][0] + s["box"][2]) / 2 for s in samples) / len(samples)
        cy = sum((s["box"][1] + s["box"][3]) / 2 for s in samples) / len(samples)
        tag = "member" if tid in member else ""
        print(f"  tid {tid} [{tag}]: ts=[{ts0:.1f},{ts1:.1f}] n={len(samples)} 质心=({cx:.0f},{cy:.0f})")


if __name__ == "__main__":
    main()
