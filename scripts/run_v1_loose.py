import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import run
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser(description="红灯停车压斑马线检测 (v1, 宽松模式)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    
    cfg.stationary.speed_px_per_sec = 50
    cfg.stationary.sustain_frames = 3
    cfg.violation.duration_frames = 3
    cfg.crosswalk.overlap_ratio = 0.15
    
    run(cfg, args.video, args.output)


if __name__ == "__main__":
    main()
