"""分析 违章03 的 1:18-2:13 (78s-133s) 窗口, 与用户 GT 比对。

用户 GT: 该窗口内有两台车占用斑马线(黑车车牌不可见 + 银灰车 京ABV...)
本脚本从 diag 时间线 CSV 统计:
  - 窗口内信号灯状态分布 (是否检出 green/flashing)
  - 每帧: 活跃车数 / 静止车数 / 压线静止车数(n_on_crosswalk_stationary) / 最大 overlap
  - 压线静止车出现的帧区间
同时输出全片 light_state 直方图, 判断 03 是否也中 E16(无 green)。
"""
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

WIN0, WIN1 = 78.0, 133.0  # 1:18 - 2:13

csv_path = r"D:\redlight-crosswalk-violation\data\output\diag_违章03_timeline.csv"
if not os.path.exists(csv_path):
    print("CSV 不存在:", csv_path)
    sys.exit(1)

rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
print(f"全片推理帧数: {len(rows)}  时间跨度: {rows[0]['ts']}s - {rows[-1]['ts']}s")

# 全片直方图
full = Counter(r["light_state"] for r in rows)
print("全片 light_state 直方图:", dict(full.most_common()))

# 窗口统计
win = [r for r in rows if WIN0 <= float(r["ts"]) <= WIN1]
print(f"\n===== 窗口 {WIN0:.0f}s-{WIN1:.0f}s: {len(win)} 帧 =====")
wst = Counter(r["light_state"] for r in win)
print("窗口 light_state 直方图:", dict(wst.most_common()))

ocs_frames = [r for r in win if int(r["n_on_crosswalk_stationary"]) > 0]
print(f"窗口内 '压线静止车>0' 的帧数: {len(ocs_frames)}")
if ocs_frames:
    print("  首帧 ts=%.2f  末帧 ts=%.2f" % (float(ocs_frames[0]["ts"]), float(ocs_frames[-1]["ts"])))
    # 统计该区间内同时出现的压线车最大数量
    max_n = max(int(r["n_on_crosswalk_stationary"]) for r in ocs_frames)
    print(f"  窗口内同时压线静止车的最大数量: {max_n}")

print("\n窗口逐帧(每 ~3s 抽1帧)明细:")
step = max(1, len(win) // 30)
for i, r in enumerate(win):
    if i % step != 0:
        continue
    print(f"  ts={float(r['ts']):6.2f} state={r['light_state']:9s} "
          f"n_active={r['n_active']:>2} n_stat={r['n_stationary']:>2} "
          f"n_ocs={r['n_on_crosswalk_stationary']:>2} max_ov={float(r['max_overlap']):.3f} "
          f"cw_present={r['crosswalk_present']} cw_occ={r['crosswalk_occluded']}")
