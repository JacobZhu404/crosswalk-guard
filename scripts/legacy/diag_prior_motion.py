"""诊断 09/06(及全11视频) 行人灯是否动相机 —— 位置级 canonical GT 测真实漂移(修复版)。

护栏#1 第一问: 09/06 是「固定机位、prior 只在 ranking 错排」(可 re-center)
还是「动相机、任何固定点皆错」(需逐帧)?

⚠️ 修复(cc 2026-08-04 plan-gate 硬伤①): 旧版把每视频 ALL governing 框展平取 range,
把「同帧多灯间距」与「相机运动」混在一起(09 实测 654px 虚高)。本版改为:
  - 按 IoU 跨帧关联成 tracklet(每盏灯一条轨迹), 报每 tracklet 的中心漂移(range/std, 像素);
  - 单列「单灯帧」(该帧仅 1 个 governing 框)的漂移, 排除同帧多灯干扰, 得到干净相机运动。

判定: 单灯帧漂移 >150px => 动相机 => 固定 prior 重定心死路。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/diag_prior_motion.py [--videos 违章09,违章06]
"""
import json
import statistics
import argparse
from collections import defaultdict

CANONICAL = "datasets/gt/light_canonical_gt.json"
W, H = 1280, 720  # 归一化→像素换算


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    ba = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + ba - inter)


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None, help="逗号分隔; 空=全部11视频")
    args = ap.parse_args()
    d = json.load(open(CANONICAL))
    byv = defaultdict(list)
    for f in d["frames"]:
        byv[f["video"]].append(f)
    videos = args.videos.split(",") if args.videos else sorted(byv)

    print(f"canonical GT: n_frames={d.get('n_frames')} n_governing={d.get('n_governing')}\n")
    print(f"{'video':<8} {'tracklet数':>9} {'tracklet最大漂移px':>16} {'单灯帧':>6} {'单灯帧漂移px':>12} {'判定'}")
    for v in videos:
        frames = sorted(byv[v], key=lambda f: int(f["source_fi"]))
        # tracklet 关联
        tracklets = []  # list of list[center]
        n_single = 0
        single_centers = []
        for f in frames:
            gboxes = [tuple(b["box_norm"]) for b in f.get("boxes", [])
                      if b.get("governing") and b.get("type") == "pedestrian"]
            if len(gboxes) == 1:
                n_single += 1
                single_centers.append(center(gboxes[0]))
            # 匹配到已有 tracklet(IoU>=0.5); 否则开新 tracklet
            matched = set()
            for b in gboxes:
                best_t, best_iou = None, 0.0
                for ti, tk in enumerate(tracklets):
                    if ti in matched:
                        continue
                    last = tk[-1]
                    i = iou(b, last)
                    if i > best_iou:
                        best_t, best_iou = ti, i
                if best_t is not None and best_iou >= 0.5:
                    tracklets[best_t].append(b)
                    matched.add(best_t)
                else:
                    tracklets.append([b])
        # 每 tracklet 漂移(用框中心)
        tracklet_drifts = []
        for tk in tracklets:
            if len(tk) < 2:
                continue
            cs = [center(b) for b in tk]
            xs = [c[0] for c in cs]
            ys = [c[1] for c in cs]
            drift = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5 * max(W, H)
            tracklet_drifts.append(drift)
        max_tk = max(tracklet_drifts) if tracklet_drifts else 0.0
        # 单灯帧漂移
        if len(single_centers) >= 2:
            sxs = [c[0] for c in single_centers]
            sys_ = [c[1] for c in single_centers]
            single_drift = ((max(sxs) - min(sxs)) ** 2 + (max(sys_) - min(sys_)) ** 2) ** 0.5 * max(W, H)
        else:
            single_drift = None
        moving = (single_drift is not None and single_drift > 150) or max_tk > 150
        verdict = "动相机✗re-center死路" if moving else "近似固定(可考虑re-center)"
        single_str = f"{single_drift:.0f}px" if single_drift is not None else "测不出(单灯帧<2)"
        mark = "  <== 重点" if v in ("违章06", "违章09") else ""
        print(f"{v:<8} {len(tracklets):>9} {max_tk:>14.0f}px {n_single:>6} {single_str:>14} {verdict}{mark}")


if __name__ == "__main__":
    main()
