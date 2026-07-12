"""针对 违章03 跑信号灯时间线诊断 (复用 diag_signal_timeline.diagnose)。"""
import os
import sys

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diag_signal_timeline import diagnose
from redlight.infrastructure.config import load_config, project_root

cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
video = r"E:\BaiduNetdiskDownload\违章03.mp4"
out = os.path.join(project_root(), "data", "output", "diag_违章03_timeline.csv")
diagnose(video, out, cfg, "balanced")
