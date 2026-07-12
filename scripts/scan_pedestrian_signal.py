"""扫描视频里"持续存在的小红/绿亮点" = 行人信号灯候选位置.

专门解决 04 这类"信号灯太小太远、自动聚类/面积过滤失败"的视频:
  直接对每帧做 HSV 阈值, 取"小面积(排除大车灯反光)+ 上部区域(cy<0.6)"的
  红/绿连通块, 跨帧统计每个位置的"出现频率".
  真实信号灯(红段常亮红/绿段亮绿)在绝大多数帧都出现 -> 频率最高;
  车灯反光/玻璃反射是偶发的 -> 频率低. 频率最高且面积稳定的位置 = 行人信号.

用法:
  python scripts/scan_pedestrian_signal.py 违章04 --max-sec 200
  python scripts/scan_pedestrian_signal.py 违章02 --max-sec 110
"""
import sys, os, argparse
import numpy as np
import cv2

FRAMES_ROOT = os.path.join("datasets", "frames")


def robust_imread(path):
    with open(path, "rb") as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def small_color_blobs(hsv, hue_ranges, s_min=120, v_min=70, v_max=240,
                      area_min=12, area_max=700, cy_max=0.62):
    """返回在给定 hue 范围内、面积适中、位于上部的连通块中心(归一化)列表."""
    H, W = hsv.shape[:2]
    out = []
    for (h0, h1) in hue_ranges:
        if h0 <= h1:
            m = (hsv[:, :, 0] >= h0) & (hsv[:, :, 0] <= h1)
        else:  # 跨 180 边界(红)
            m = (hsv[:, :, 0] >= h0) | (hsv[:, :, 0] <= h1)
        m &= (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min) & (hsv[:, :, 2] <= v_max)
        m = m.astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a < area_min or a > area_max:
                continue
            cy_px = cents[i, 1]
            if cy_px / H > cy_max:
                continue
            cx = cents[i, 0] / W
            cy = cy_px / H
            out.append((round(cx, 3), round(cy, 3), int(a)))
    return out


def cluster(records, grid=0.03):
    """records: list of (cx,cy,area). 按 grid 网格聚合, 返回 (cx,cy,n,avg_area)."""
    buckets = {}
    for cx, cy, a in records:
        key = (round(cx / grid) * grid, round(cy / grid) * grid)
        buckets.setdefault(key, []).append(a)
    res = []
    for (gx, gy), areas in buckets.items():
        res.append((round(gx, 3), round(gy, 3), len(areas), int(np.mean(areas))))
    res.sort(key=lambda r: -r[2])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-sec", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=3, help="每 N 帧抽 1 帧扫描")
    args = ap.parse_args()

    vdir = os.path.join(FRAMES_ROOT, args.video)
    if not os.path.isdir(vdir):
        print(f"找不到抽帧目录: {vdir}")
        sys.exit(1)
    files = sorted([f for f in os.listdir(vdir) if f.endswith(".jpg")])

    red_ranges = [(0, 12), (168, 179)]   # 红(跨边界)
    green_ranges = [(38, 82)]             # 绿

    red_recs, green_recs = [], []
    n = 0
    for i, fn in enumerate(files):
        if i % args.stride != 0:
            continue
        idx = int(fn.split("_")[1].split(".")[0])  # frame_000123.jpg -> 123
        path = os.path.join(vdir, fn)
        fr = robust_imread(path)
        if fr is None:
            continue
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        red_recs += small_color_blobs(hsv, red_ranges)
        green_recs += small_color_blobs(hsv, green_ranges)
        n += 1

    print(f"\n扫描 {args.video}: 分析了 {n} 帧")
    print(f"\n=== 红光斑 频率 Top10 (位置, 出现帧数, 平均面积px) ===")
    for cx, cy, cnt, area in cluster(red_recs)[:10]:
        print(f"  ({cx:.3f},{cy:.3f})  freq={cnt:4d}  avgArea={area:4d}")
    print(f"\n=== 绿光斑 频率 Top10 (位置, 出现帧数, 平均面积px) ===")
    for cx, cy, cnt, area in cluster(green_recs)[:10]:
        print(f"  ({cx:.3f},{cy:.3f})  freq={cnt:4d}  avgArea={area:4d}")


if __name__ == "__main__":
    main()
