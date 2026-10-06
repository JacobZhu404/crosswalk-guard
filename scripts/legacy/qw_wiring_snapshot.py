"""qw 接线护栏①快照工具: 跑全 11 视频的 cli.run 默认路径, 落 violations.csv + sha256 清单。

用途(接线 C3 验收, cc gate 2026-08-04):
  - 接线前: 连跑两次(--tag run1/run2)立"确定性地板"(run-to-run 必须 0 diff);
  - 接线后: --config 指到 version=v11+occ_denom=mask 的回退配置, 与接线前快照逐字节 diff。

口径: **不传 crosswalk_detector/occ_denom**(走 cli.run 默认分支)——接线前默认=v11+mask,
接线后默认=v2+box; 回退验证时用 version=v11 的 config 让默认分支走回 v11+mask。
"""
import argparse
import hashlib
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.app import cli
from redlight.infrastructure.config import load_config

VIDEOS = [f"违章{i:02d}" for i in range(1, 12)]


def run_one(args):
    cfg, video, out_dir = args
    video_path = os.path.join(ROOT, "input_video", f"{video}.mp4")
    if not os.path.isfile(video_path):
        return video, "MISSING"
    os.makedirs(out_dir, exist_ok=True)
    events = cli.run(cfg, video_path, out_dir, preset="balanced")  # 不传参 -> 默认分支
    confirmed = sum(1 for e in events if e.get("status") == "confirmed")
    return video, f"confirmed={confirmed}"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--tag", required=True, help="如 run1/run2/post")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "data", "output", "qw", "wiring_pre"))
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # 快照只关心 violations.csv: 禁标注视频/证据截图提速(不影响事件判定与 CSV 内容)
    cfg.output.annotated_video = False
    cfg.output.evidence_images = False
    out_root = os.path.join(args.out_root, args.tag)
    os.makedirs(out_root, exist_ok=True)

    tasks = [(cfg, v, os.path.join(out_root, f"run_{v}")) for v in VIDEOS]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs) as pool:
        results = pool.map(run_one, tasks, chunksize=1)

    print("=== 运行摘要 ===")
    for video, r in results:
        print(f"  {video}: {r}")

    print("=== sha256 ===")
    lines = []
    ok = True
    for video, _ in results:
        csv_path = os.path.join(out_root, f"run_{video}", "violations.csv")
        if not os.path.isfile(csv_path):
            print(f"  {video}: 无 violations.csv")
            ok = False
            continue
        h = sha256_of(csv_path)
        lines.append(f"{h}  {video}")
        print(f"  {video}: {h[:16]}...")
    with open(os.path.join(out_root, "sha256.txt"), "w") as f:
        f.write("\n".join(sorted(lines)) + "\n")
    print(f"[完成] tag={args.tag} 快照 -> {out_root} (ok={ok})")


if __name__ == "__main__":
    main()
