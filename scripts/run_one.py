"""运行完整pipeline: 处理指定视频 (无tqdm, 输出到日志)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 禁用 tqdm
os.environ["TQDM_DISABLE"] = "1"

from src.pipeline import run
from src.utils import load_config

cfg = load_config("configs/config.yaml")
video = r"E:\BaiduNetdiskDownload\违章10.mp4"
out_dir = os.path.join("data", "output", "run10")

log_path = os.path.join(out_dir, "run.log")
os.makedirs(out_dir, exist_ok=True)

# 重定向print到文件和stdout
class Tee:
    def __init__(self, f1, f2):
        self.f1 = f1; self.f2 = f2
    def write(self, s): self.f1.write(s); self.f2.write(s); self.f1.flush(); self.f2.flush()
    def flush(self): self.f1.flush(); self.f2.flush()

log_f = open(log_path, "w", encoding="utf-8")
old_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_f)

try:
    events = run(cfg, video, out_dir)
    print(f"\n=== 违规事件 ({len(events)}) ===")
    for ev in events:
        print(ev)
finally:
    sys.stdout = old_stdout
    log_f.close()
print(f"Done. Log: {log_path}")
