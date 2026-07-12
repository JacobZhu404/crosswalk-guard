"""本地画廊服务: 托管 data/output/light_eval/ 并接收标注反馈。

用途: 运行后浏览器打开 http://localhost:<port> , 在画廊里点"保存"标注误差帧,
反馈实时追加写入 data/output/annotated/light_feedback.csv (回归集原料).

用法:
  python scripts/serve_gallery.py            # 默认 8765
  python scripts/serve_gallery.py 9000       # 自定义端口
"""
import os
import sys
import json
import csv
import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "data", "output", "light_eval")
FEEDBACK_CSV = os.path.join(ROOT, "data", "output", "annotated", "light_feedback.csv")
HEADER = ["video", "t_sec", "frame_idx", "pred", "gt", "verdict", "reason", "note", "ts"]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=EVAL_DIR, **kwargs)

    def do_GET(self):
        if self.path in ("/", ""):
            self.path = "/gallery.html"
        return super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") == "/feedback":
            try:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                d = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
                return
            row = [
                d.get("video", ""), d.get("t", ""), d.get("idx", ""),
                d.get("pred", ""), d.get("gt", ""), d.get("verdict", ""),
                d.get("reason", ""), d.get("note", ""),
                datetime.datetime.now().isoformat(timespec="seconds"),
            ]
            try:
                os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
                write_header = not os.path.exists(FEEDBACK_CSV)
                with open(FEEDBACK_CSV, "a", encoding="utf-8-sig", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(HEADER)
                    w.writerow(row)
                self._json(200, {"ok": True})
            except Exception as e:
                self._json(500, {"ok": False, "error": str(e)})
        else:
            self._json(404, {"ok": False})

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _json(self, code, obj):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # 安静


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"[serve] {EVAL_DIR}")
    print(f"[serve] 画廊: http://localhost:{port}")
    print(f"[serve] 反馈写入: {FEEDBACK_CSV}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
