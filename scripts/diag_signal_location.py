"""诊断探针: 不依赖位置先验, 找出每视频里"稳定的紧凑亮斑"(疑似信号灯)。

方法:
  1. 自适应亮度阈值 bright = V >= mean_V + 1.0*std_V (适配不同曝光)
  2. 形态学清理 + 连通分量, 取紧凑(aspect<3, solidity 不卡) 且面积[25, max] 的亮斑
  3. 按 bbox 内 HSV 均值分类 红/绿/其它
  4. 跨帧把亮斑按 (量化x,量化y) 聚类, 统计每簇: 出现帧数(persistence)、主导色、平均 cy、平均面积
     -> 真实信号灯 = 高持续性 + 颜色与 GT 一致 的簇。

输出每视频前若干最持续簇, 以及 GT 对照(来自 events.csv)。
"""
import sys, os, glob, csv
os.environ["TQDM_DISABLE"] = "1"
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import cv2, numpy as np
from collections import Counter
from redlight.infrastructure.config import load_config, project_root

ROOT = project_root()
SAMPLE_FPS = 8
VIDEOS = ["违章01", "违章02", "违章07", "违章11"]  # 对照 02 + 失败三

def classify(mean_h, mean_s):
    if mean_s < 25:
        return "white"
    if 35 <= mean_h <= 95:
        return "green"
    if mean_h <= 18 or mean_h >= 160:
        return "red"
    return "other"

def find_spots(roi):
    rh, rw = roi.shape[:2]
    if rh == 0 or rw == 0:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    vmean, vstd = v.mean(), v.std()
    th = vmean + 1.0 * vstd
    bright = (v >= th).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    num, _, stats, cents = cv2.connectedComponentsWithStats(bright, 8)
    spots = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 25:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH]); bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        aspect = max(bw, bh) / min(bw, bh)
        if aspect > 3.0:
            continue
        cy = (y + bh/2.0) / rh
        patch = hsv[y:y+bh, x:x+bw]
        mh = float(np.mean(patch[:, :, 0])); ms = float(np.mean(patch[:, :, 1]))
        col = classify(mh, ms)
        spots.append({"x": x+bw//2, "y": y+bh//2, "cy": cy, "a": a, "col": col})
    return spots

def gt_for(name):
    g = []
    p = os.path.join(ROOT, "datasets", "gt", "events.csv")
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["video"] == name:
                g.append((float(r["start_s"]), float(r["end_s"]), r["light_state"]))
    return g

if __name__ == "__main__":
    for name in VIDEOS:
        V = os.path.join(ROOT, "input_video", name + ".mp4")
        cap = cv2.VideoCapture(V)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(round(fps / SAMPLE_FPS)))
        fi = 0
        # 簇: key=(qx,qy) -> {count, cols:Counter, cys:[], areas:[]}
        clusters = {}
        nframes = 0
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if fi % interval == 0:
                nframes += 1
                spots = find_spots(fr)
                for s in spots:
                    qx = int(s["x"] // 60); qy = int(s["y"] // 60)
                    k = (qx, qy)
                    c = clusters.setdefault(k, {"count": 0, "cols": Counter(), "cys": [], "areas": []})
                    c["count"] += 1
                    c["cols"][s["col"]] += 1
                    c["cys"].append(s["cy"])
                    c["areas"].append(s["a"])
            fi += 1
        cap.release()
        # 持久簇(出现 >= 20% 帧)
        pers = [(k, c) for k, c in clusters.items() if c["count"] >= max(3, nframes*0.2)]
        pers.sort(key=lambda kv: -kv[1]["count"])
        print(f"\n===== {name} (frames={nframes}, GT={gt_for(name)}) =====")
        print(f"  持久亮斑簇(>=20%帧): {len(pers)} 个")
        for k, c in pers[:8]:
            dom = c["cols"].most_common(1)[0]
            cys = np.array(c["cys"]); areas = np.array(c["areas"])
            print(f"    loc~({k[0]*60},{k[1]*60}) persist={c['count']}/{nframes} "
                  f"dom={dom[0]}({dom[1]}) cy_mean={cys.mean():.2f} area_mean={areas.mean():.0f}")
        if not pers:
            print("  !!! 无任何持久亮斑 -> v5 的位置/亮度假设在该视频完全失效")
