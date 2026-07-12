"""
诊断每个 GT 段的"信号灯可见性"——数据驱动地定 light_evidence。

方法: 用当前检测器(sat_min=130)逐帧取候选灯块, 对每段按 2s 分箱,
统计箱内"出现期望颜色且面积>=MIN_VIS 的灯块"的帧占比。
- 占比 >= PERSIST -> 该箱"灯可见"
- 连续可见箱合并为 visible 子段; 间隙为 inferred/occluded 子段。

输出: 每个视频每段的可见子段清单, 供迁移 events.csv 时定 light_evidence。
"""
import sys, os, csv, collections, numpy as np, cv2
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from redlight.models.traffic_light import TrafficLightDetector
import types

MIN_VIS = 40          # 真实灯块面积下限(px), 噪声 floor=20 被排除
PERSIST = 0.30        # 箱内"可见帧"占比阈值
BIN = 2.0             # 分箱宽度(秒)

def _cfg():
    return types.SimpleNamespace(traffic_light=types.SimpleNamespace(
        smoothing_window=24, value_floor=60, sat_min=130, min_area_px=20,
        max_area_ratio=0.008, max_aspect_ratio=3.5, color_s_min=22,
        match_radius_ratio=0.06, min_persist_frames=5, track_persist_min=0.08,
        signal_cy_cutoff=0.6, flicker_toggle_count=4))

rows = list(csv.DictReader(open("datasets/gt/events.csv", encoding="utf-8")))
segs = [r for r in rows if r["light_state"] in ("green", "red", "flashing")]

for r in segs:
    v = r["video"]; a = float(r["start_s"]); b = float(r["end_s"])
    cap = cv2.VideoCapture(f"input_video/{v}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    det = TrafficLightDetector(_cfg(), verbose=False)
    fi = 0
    # bin_index -> (visible_frames, total_frames)
    bins = collections.defaultdict(lambda: [0, 0])
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        fi += 1
        if t < a or t > b:
            continue
        if fi % 3 != 0:
            continue
        res = det.detect(frame)
        cands = res.get("candidates", [])
        want = r["light_state"]
        # 期望颜色的可见灯块?
        vis = any(c["color"] == want and c["area"] >= MIN_VIS for c in cands)
        bi = int((t - a) // BIN)
        bins[bi][1] += 1
        if vis:
            bins[bi][0] += 1
    cap.release()
    # 推导可见子段
    if not bins:
        print(f"{v} [{a}-{b}] {r['light_state']}: (无采样帧)")
        continue
    maxbi = max(bins)
    visible_ranges = []
    cur_s = cur_e = None
    def flush():
        global cur_s, cur_e
        if cur_s is not None:
            visible_ranges.append((cur_s, cur_e))
            cur_s = cur_e = None
    for bi in range(maxbi + 1):
        vf, tf = bins.get(bi, [0, 0])
        frac = (vf / tf) if tf else 0
        bs = a + bi * BIN
        be = min(b, a + (bi + 1) * BIN)
        if frac >= PERSIST:
            if cur_s is None:
                cur_s, cur_e = bs, be
            else:
                cur_e = be
        else:
            flush()
    flush()
    # 把可见子段格式化为字符串
    vr_str = "; ".join(f"{s:.0f}-{e:.0f}s" for s, e in visible_ranges) or "(无可见灯块)"
    # 推断间隙
    gaps = []
    prev = a
    for s, e in visible_ranges:
        if s > prev:
            gaps.append(f"{prev:.0f}-{s:.0f}s")
        prev = e
    if prev < b:
        gaps.append(f"{prev:.0f}-{b:.0f}s")
    gap_str = "; ".join(gaps) or "(无间隙)"
    print(f"{v} [{a}-{b}] {r['light_state']}:")
    print(f"    可见子段: {vr_str}")
    print(f"    推断/遮挡间隙: {gap_str}")
