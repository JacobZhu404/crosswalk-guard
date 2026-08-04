"""诊断 09/06(及全11视频) 行人灯是否动相机 —— 用位置级 canonical GT 测真实漂移。

护栏#1 第一问: 09/06 是「固定机位、prior 只在 ranking 错排」(可 re-center)
还是「动相机、任何固定点皆错」(需逐帧)?

方法: 读 datasets/gt/light_canonical_gt_v1 (399帧全11视频逐帧行人灯框),
对每个视频 aggregate 其 governing pedestrian 框中心, 量跨帧 std 与 x/y 范围(像素)。
大漂移 => 动相机 => 固定 prior 重定心死路。

注意: datasets/light_location_gt.json 每视频仅3帧标注, 会掩盖帧间运动, 不可用作
运动判定(本脚本只用 canonical 全帧 GT)。

用法:
  PYTHONPATH=src ./.venv/bin/python scripts/diag_prior_motion.py
"""
import json
import statistics
from collections import defaultdict

CANONICAL = "datasets/gt/light_canonical_gt.json"
W, H = 1280, 720  # 归一化→像素换算假设


def main():
    d = json.load(open(CANONICAL))
    pts = defaultdict(list)
    nframes = defaultdict(int)
    for f in d["frames"]:
        v = f["video"]
        nframes[v] += 1
        for b in f.get("boxes", []):
            if b.get("governing") and b.get("type") == "pedestrian":
                x1, y1, x2, y2 = b["box_norm"]
                pts[v].append(((x1 + x2) / 2, (y1 + y2) / 2))

    print(f"canonical GT: n_frames={d.get('n_frames')} n_governing={d.get('n_governing')}")
    print(f"{'video':<8} {'n帧':>4} {'gov框':>6} {'cx±std':>12} {'cy±std':>12} {'Δx':>7} {'Δy':>7} {'判定'}")
    for v in sorted(pts):
        boxes = pts[v]
        if not boxes:
            print(f"{v:<8} {nframes[v]:>4} {'0':>6} {'-':>12} {'-':>12} {'':>7} {'':>7} 无governing框")
            continue
        cxs = [c[0] for c in boxes]
        cys = [c[1] for c in boxes]
        sx = statistics.pstdev(cxs)
        sy = statistics.pstdev(cys)
        rx = (max(cxs) - min(cxs)) * W
        ry = (max(cys) - min(cys)) * H
        moving = rx > 150 or ry > 150  # >150px 视为动相机
        verdict = "动相机✗re-center死路" if moving else "近似固定(可考虑re-center)"
        mark = "  <== 重点" if v in ("违章06", "违章09") else ""
        print(f"{v:<8} {nframes[v]:>4} {len(boxes):>6} {cxs[0]:>6.3f}±{sx:<5.3f} {cys[0]:>6.3f}±{sy:<5.3f} {rx:>5.0f}px {ry:>5.0f}px {verdict}{mark}")


if __name__ == "__main__":
    main()
