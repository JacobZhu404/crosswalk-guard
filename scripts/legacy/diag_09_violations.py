#!/usr/bin/env python3
"""09 违章区间诊断: 在 T=6(#3 retune)下跑 09, 打印每个 confirmed/review 违章的
时间区间, 并对照 GT(违章09: [0,10]red无违章, [11,72]green真违章, [72,106.4]unknown无违章)。

直接回答 P: confirmed 违章是否全部落在 [11,72] (GT 真违章窗) -> P=1.000;
若有 confirmed 落在 [0,10] 或 [72,106.4] -> FP。
"""
import os, sys, json
from redlight.infrastructure.config import load_config, project_root
from redlight.app import cli as cli_mod

VIDEO = "违章09"
GT = [(0.0, 10.0, "red(无违章)"), (11.0, 72.0, "green(真违章)"), (72.0, 106.4, "unknown(无违章)")]


def window_of(t):
    for s, e, name in GT:
        if s - 1e-6 <= t <= e + 1e-6:
            return name
    return "?"


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    video_path = os.path.join("/Users/jacob/personal/crosswalk-guard/input_video", VIDEO + ".mp4")
    out = "/tmp/diag_09_viol"
    events, _ = cli_mod.run(cfg, video_path, out, "balanced",
                           return_track_samples=True, min_persistent_green_run_s=6.0)
    print("=== event 样例 keys ===", list(events[0].keys()) if events else "empty")
    print("=== 09 在 T=6 下违章事件 (status, [start,end], 落入GT窗) ===")
    conf = rev = 0
    fp_conf = []
    for e in events:
        st = e.get("status")
        s = e.get("start_ts", e.get("start_s", e.get("start")))
        en = e.get("end_ts", e.get("end_s", e.get("end")))
        if s is None or en is None:
            continue
        w = window_of((s + en) / 2.0)
        if st == "confirmed":
            conf += 1
            if not (11.0 - 1e-6 <= (s + en) / 2.0 <= 72.0 + 1e-6):
                fp_conf.append((s, en, w))
            print(f"  CONFIRMED [{s:6.2f},{en:6.2f}] 中点窗={w} light={e.get('light_state')}")
        elif st == "review":
            rev += 1
            print(f"  review   [{s:6.2f},{en:6.2f}] 中点窗={w} light={e.get('light_state')}")
    print(f"\nconfirmed={conf} review={rev}")
    print(f"confirmed 落 [0,10]或[72,106.4](FP)={len(fp_conf)}")
    if fp_conf:
        for s, en, w in fp_conf:
            print(f"   ！！FP confirmed [{s:.2f},{en:.2f}] 窗={w}")


if __name__ == "__main__":
    main()
