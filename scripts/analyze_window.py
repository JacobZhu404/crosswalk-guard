"""通用窗口分析: 从 diag 时间线 CSV 统计某视频某时间窗口内的信号灯/压线情况。

用法:
    python scripts/analyze_window.py 违章02 21 85
    python scripts/analyze_window.py 违章03 78 133
"""
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

if len(sys.argv) < 4:
    print("usage: analyze_window.py <视频名无后缀> <win_start_s> <win_end_s>")
    sys.exit(1)

VID = sys.argv[1]
WIN0, WIN1 = float(sys.argv[2]), float(sys.argv[3])

csv_path = f"D:/redlight-crosswalk-violation/data/output/diag_{VID}_timeline.csv"
if not os.path.exists(csv_path):
    print("CSV 不存在:", csv_path)
    sys.exit(1)

rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
print(f"[{VID}] 全片推理帧数: {len(rows)}  时间跨度: {rows[0]['ts']}s - {rows[-1]['ts']}s")

full = Counter(r["light_state"] for r in rows)
print("[全片] light_state 直方图:", dict(full.most_common()))

win = [r for r in rows if WIN0 <= float(r["ts"]) <= WIN1]
print(f"\n===== 窗口 {WIN0:.0f}s-{WIN1:.0f}s: {len(win)} 帧 =====")
wst = Counter(r["light_state"] for r in win)
print("[窗口] light_state 直方图:", dict(wst.most_common()))
print("[窗口] light_reason 直方图:", dict(Counter(r["light_reason"] for r in win).most_common()))

ocs_frames = [r for r in win if int(r["n_on_crosswalk_stationary"]) > 0]
print(f"[窗口] '压线静止车>0' 的帧数: {len(ocs_frames)}")
if ocs_frames:
    print(f"  首帧 ts={float(ocs_frames[0]['ts']):.2f}  末帧 ts={float(ocs_frames[-1]['ts']):.2f}")
    print(f"  同时压线静止车最大数量: {max(int(r['n_on_crosswalk_stationary']) for r in ocs_frames)}")

print("\n[窗口] 逐帧明细(每~3s抽1):")
step = max(1, len(win) // 30)
for i, r in enumerate(win):
    if i % step != 0:
        continue
    print(f"  ts={float(r['ts']):6.2f} state={r['light_state']:9s} reason={r['light_reason']:14s} "
          f"n_act={r['n_active']:>2} n_stat={r['n_stationary']:>2} "
          f"n_ocs={r['n_on_crosswalk_stationary']:>2} max_ov={float(r['max_overlap']):.3f} "
          f"cw_p={r['crosswalk_present']} cw_o={r['crosswalk_occluded']}")
