"""轻量可见性扫描: 仅 08 / 11, 定细粒度拆分与绿灯可见性。"""
import sys, os, csv, collections, numpy as np, cv2
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from redlight.models.traffic_light import TrafficLightDetector
import types

MIN_VIS = 250
PERSIST = 0.30
BIN = 2.0

def _cfg():
    return types.SimpleNamespace(traffic_light=types.SimpleNamespace(
        smoothing_window=24, value_floor=60, sat_min=130, min_area_px=20,
        max_area_ratio=0.008, max_aspect_ratio=3.5, color_s_min=22,
        match_radius_ratio=0.06, min_persist_frames=5, track_persist_min=0.08,
        signal_cy_cutoff=0.6, flicker_toggle_count=4))

rows = list(csv.DictReader(open("datasets/gt/events.csv", encoding="utf-8")))
for v in ["违章08", "违章11"]:
    for r in rows:
        if r["video"] != v or r["light_state"] not in ("green", "red", "flashing"):
            continue
        a = float(r["start_s"]); b = float(r["end_s"]); want = r["light_state"]
        cap = cv2.VideoCapture(f"input_video/{v}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 8
        det = TrafficLightDetector(_cfg(), verbose=False)
        fi = 0
        bins = collections.defaultdict(lambda: [0, 0])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = fi / fps; fi += 1
            if t < a or t > b or fi % 3 != 0:
                continue
            cands = det.detect(frame).get("candidates", [])
            vis = any(c["color"] == want and c["area"] >= MIN_VIS for c in cands)
            bi = int((t - a) // BIN)
            bins[bi][1] += 1
            if vis:
                bins[bi][0] += 1
        cap.release()
        maxbi = max(bins) if bins else -1
        vr = []; cs = ce = None
        def flush():
            global cs, ce
            if cs is not None:
                vr.append((cs, ce)); cs = ce = None
        for bi in range(maxbi + 1):
            vf, tf = bins.get(bi, [0, 0]); frac = vf / tf if tf else 0
            bs, be = a + bi * BIN, min(b, a + (bi + 1) * BIN)
            if frac >= PERSIST:
                if cs is None: cs, ce = bs, be
                else: ce = be
            else:
                flush()
        flush()
        vr_s = "; ".join(f"{s:.0f}-{e:.0f}s" for s, e in vr) or "(无可见灯块)"
        gaps = []; prev = a
        for s, e in vr:
            if s > prev: gaps.append(f"{prev:.0f}-{s:.0f}s")
            prev = e
        if prev < b: gaps.append(f"{prev:.0f}-{b:.0f}s")
        gap_s = "; ".join(gaps) or "(无间隙)"
        print(f"{v} [{a}-{b}] {want}:  可见={vr_s}  |  间隙={gap_s}")
