"""CLI 便捷入口: 委托给 v2.0 分层架构的 redlight.app.cli.run。

用法:
    python scripts/run_video.py <video.mp4> [output_dir] [--preset balanced|strict|loose]
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config, project_root
from redlight.app.cli import run


def main():
    ap = argparse.ArgumentParser(description="斑马线压线检测 (v2.0)")
    ap.add_argument("video")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--preset", default="balanced",
                    choices=["strict", "balanced", "loose", "very_loose"])
    ap.add_argument("--mode", default="red_light",
                    choices=["red_light", "pedestrian_green"])
    ap.add_argument("--config", default=os.path.join(project_root(), "configs", "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output is None:
        name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(project_root(), "data", "output", f"run_{name}_{args.preset}")
    run(cfg, args.video, args.output, args.preset, mode=args.mode)


if __name__ == "__main__":
    main()
