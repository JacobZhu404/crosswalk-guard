"""重跑脚本 (2026-07-11 语义反转后): 用修正后的正确逻辑(行人绿灯压线=违规)
端到端处理 违章01 / 违章02, 输出标注视频 + 证据截图 + CSV, 并落盘日志。

用法 (工程根目录):
    python scripts/rerun_reversed_01_02.py
"""
import sys
import os

os.environ["TQDM_DISABLE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.app.cli import run
from redlight.infrastructure.config import load_config, project_root


class Tee:
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def write(self, s):
        self.f1.write(s)
        self.f2.write(s)
        self.f1.flush()
        self.f2.flush()

    def flush(self):
        self.f1.flush()
        self.f2.flush()


def run_one(video, out_dir, cfg, preset):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    log_f = open(log_path, "w", encoding="utf-8")
    old = sys.stdout
    sys.stdout = Tee(sys.stdout, log_f)
    try:
        print(f"\n===== 处理 {video} -> {out_dir} (preset={preset}) =====")
        events = run(cfg, video, out_dir, preset)
        confirmed = sum(1 for e in events if e["status"] == "confirmed")
        review = sum(1 for e in events if e["status"] == "review")
        print(f"[SUMMARY] 确认违规={confirmed} 待复核={review}")
        return events
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return None
    finally:
        sys.stdout = old
        log_f.close()


def main():
    cfg = load_config(os.path.join(project_root(), "configs", "config.yaml"))
    preset = "balanced"
    base_in = r"E:\BaiduNetdiskDownload"
    base_out = os.path.join(project_root(), "data", "output")
    targets = [
        ("违章01.mp4", "run_违章01_balanced"),
        ("违章02.mp4", "run_违章02_balanced"),
    ]
    for fname, out_name in targets:
        video = os.path.join(base_in, fname)
        out_dir = os.path.join(base_out, out_name)
        if not os.path.exists(video):
            print(f"[SKIP] 找不到视频: {video}")
            continue
        run_one(video, out_dir, cfg, preset)


if __name__ == "__main__":
    main()
