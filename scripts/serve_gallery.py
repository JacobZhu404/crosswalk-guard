"""本地画廊服务: 托管一个 eval_dir 并接收标注反馈(通用版)。

用途: 运行后浏览器打开 http://localhost:<port> , 在画廊里点"保存"标注,
反馈实时追加写入 feedback CSV。POST 的任意字段都会透传落盘(动态表头),
故灯态(verdict/reason/note)、车牌(corrected_plate)、斑马线(y0/y1)、
跟踪(box)等不同画廊共用同一服务, 无需改服务端。

向后兼容: 不带参数时 = 旧行为(托管 light_eval, 写 light_feedback.csv)。

用法:
  python scripts/serve_gallery.py                       # 灯态(默认, 8765)
  python scripts/serve_gallery.py 9000                  # 自定义端口(位置参数, 兼容旧用法)
  python scripts/serve_gallery.py --eval-dir data/output/crosswalk_eval \
      --feedback-csv datasets/gt/crosswalk_feedback.csv --port 8766   # 斑马线标注
"""
import os
import sys
import json
import csv
import argparse
import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 表头首选顺序(出现即按此序; 未列出的额外字段追加在 ts 之前)。
_PREFERRED = ["video", "t_sec", "frame_idx", "pred", "gt",
              "verdict", "reason", "note", "corrected_plate", "y0", "y1", "box"]


def _fieldnames(rows):
    """动态表头 = 首选序 + 出现过的额外字段, ts 恒在最后。"""
    seen = set()
    for r in rows:
        seen.update(r.keys())
    ordered = [c for c in _PREFERRED if c in seen]
    extras = sorted(c for c in seen if c not in _PREFERRED and c != "ts")
    return ordered + extras + (["ts"] if "ts" in seen or True else [])


def _append_feedback(feedback_csv, row):
    """追加/原地更新一条标注(按 video+t_sec+frame_idx 去重); 任意字段透传, 动态表头。"""
    os.makedirs(os.path.dirname(feedback_csv), exist_ok=True)
    rows = []
    if os.path.exists(feedback_csv):
        with open(feedback_csv, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    key = (row.get("video", ""), str(row.get("t_sec", "")), str(row.get("frame_idx", "")))
    replaced = False
    for i, r in enumerate(rows):
        if (r.get("video", ""), str(r.get("t_sec", "")), str(r.get("frame_idx", ""))) == key:
            r.update(row)  # 合并: 保留旧列, 覆盖新值
            replaced = True
            break
    if not replaced:
        rows.append(dict(row))
    fields = _fieldnames(rows)
    with open(feedback_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return replaced


def make_handler(eval_dir, feedback_csv):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=eval_dir, **kwargs)

        def do_GET(self):
            if self.path.rstrip("/") == "/feedback":
                try:
                    rows = []
                    if os.path.exists(feedback_csv):
                        with open(feedback_csv, encoding="utf-8-sig", newline="") as f:
                            rows = list(csv.DictReader(f))
                    self._json(200, {"ok": True, "rows": rows})
                except Exception as e:
                    self._json(500, {"ok": False, "error": str(e)})
                return
            if self.path in ("/", ""):
                self.path = "/gallery.html"
            return super().do_GET()

        def do_POST(self):
            if self.path.rstrip("/") == "/feedback":
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    d = json.loads(self.rfile.read(n).decode("utf-8"))
                except Exception as e:
                    self._json(400, {"ok": False, "error": str(e)})
                    return
                # t/idx 映射为 t_sec/frame_idx; 其余字段(verdict/y0/y1/box/...) 全部透传
                row = {
                    "video": d.get("video", ""),
                    "t_sec": d.get("t", d.get("t_sec", "")),
                    "frame_idx": d.get("idx", d.get("frame_idx", "")),
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                for k, v in d.items():
                    if k in ("video", "t", "idx", "t_sec", "frame_idx"):
                        continue
                    row[k] = v
                try:
                    replaced = _append_feedback(feedback_csv, row)
                    self._json(200, {"ok": True, "replaced": bool(replaced)})
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
            pass

    return Handler


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port_pos", nargs="?", type=int, default=None,
                    help="端口(位置参数, 兼容旧用法 `serve_gallery.py 9000`)")
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "data", "output", "light_eval"))
    ap.add_argument("--feedback-csv",
                    default=os.path.join(ROOT, "data", "output", "annotated", "light_feedback.csv"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    port = args.port_pos if args.port_pos is not None else args.port
    eval_dir = os.path.abspath(args.eval_dir)
    feedback_csv = os.path.abspath(args.feedback_csv)
    print(f"[serve] {eval_dir}")
    print(f"[serve] 画廊: http://localhost:{port}")
    print(f"[serve] 反馈写入: {feedback_csv}")
    HTTPServer(("127.0.0.1", port), make_handler(eval_dir, feedback_csv)).serve_forever()
