import sys, os, io, contextlib
root = r"D:\redlight-crosswalk-violation"
sys.path.insert(0, root)
os.chdir(root)
from src.pipeline import run
from src.utils import load_config

cfg = load_config("configs/config.yaml")
video = sys.argv[1] if len(sys.argv) > 1 else r"E:\BaiduNetdiskDownload\违章10.mp4"
out = sys.argv[2] if len(sys.argv) > 2 else r"D:\redlight-crosswalk-violation\data\output\run10b"

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        events = run(cfg, video, out)
        print(f"EVENTS={len(events)}")
    except Exception as e:
        import traceback
        print("RUN ERROR:\n" + traceback.format_exc())

log = buf.getvalue()
print(log)
with open(r"D:\redlight-crosswalk-violation\data\run_log.txt", "w", encoding="utf-8") as f:
    f.write(log)
