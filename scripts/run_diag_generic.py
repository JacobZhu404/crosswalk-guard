"""通用信号灯时间线诊断启动器 (2026-07-11)。

用法:
    python scripts/run_diag_generic.py <video.mp4> <out.csv> [preset]
"""
import sys
import os

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diag_signal_timeline import diagnose
from redlight.infrastructure.config import load_config, project_root

if __name__ == "__main__":
    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        project_root(), "data", "output", f"diag_{os.path.splitext(os.path.basename(video))[0]}_timeline.csv")
    preset = sys.argv[3] if len(sys.argv) > 3 else "balanced"
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    diagnose(video, out, cfg, preset)
