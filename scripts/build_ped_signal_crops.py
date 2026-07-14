"""构建 M1 行人信号灯 crop 训练集 (Phase2 spec §5)。

从预抽帧(datasets/frames, 推荐)或源视频, 在可见灯态段抽候选 ROI, 按 GT 段灯态 + 先验
弱标签(walk/stand/off), 落到 datasets/ped_signal/(crop 图 + labels.csv)。
crop 数据集应版本化进 git(小, 属训练数据), 一次构建跨机 pull 复用。

用法:
    # 预抽帧模式(推荐; 先 python scripts/extract_frames.py --all --fps 4)
    python scripts/build_ped_signal_crops.py --frames-dir datasets/frames

    # 源视频模式
    python scripts/build_ped_signal_crops.py --video-dir input_video --fps 4

弱标签依赖: datasets/gt/light_states.csv(段灯态) + configs/light_priors.json(信号位置先验,
仅 02/03/04 有)。无先验的视频 -> crop 标 None, 需灯态画廊人工标注后再训练。
"""
import os
import sys
import json
import argparse
import functools

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.infrastructure.config import load_config
from redlight.evaluation import gt_lookup
from redlight.evaluation.frame_dataset import FrameDataset
from redlight.models.traffic_light import TrafficLightDetector
from redlight.data_pipeline.ped_signal_dataset import extract_crops, write_labels_csv


def _hsv_candidates_fn(tl):
    """用 TrafficLightDetector 的 HSV 亮斑作候选(cx,cy 已归一化)。"""
    def fn(frame):
        return [{"box": s["box"], "cx": s["cx"], "cy": s["cy"], "source": "hsv"}
                for s in tl._candidates(frame)]
    return fn


def _video_frames(video_path, sample_fps):
    """源视频抽帧迭代 (frame_idx, ts, frame)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(src_fps / sample_fps)))
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % interval == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx, idx / src_fps, frame
        idx += 1
    cap.release()


def _load_priors(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # light_priors.json: {video: [cx, cy, roi_px]} -> 取 (cx, cy)
    return {v: (a[0], a[1]) for v, a in raw.items() if isinstance(a, (list, tuple)) and len(a) >= 2}


def main():
    ap = argparse.ArgumentParser(description="构建行人信号灯 crop 训练集")
    ap.add_argument("--frames-dir", default=None, help="预抽帧目录(datasets/frames)")
    ap.add_argument("--video-dir", default=None, help="源视频目录(input_video)")
    ap.add_argument("--fps", type=int, default=4, help="视频模式采样帧率")
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "ped_signal"))
    ap.add_argument("--light-states", default=os.path.join(ROOT, "datasets", "gt", "light_states.csv"))
    ap.add_argument("--priors", default=os.path.join(ROOT, "configs", "light_priors.json"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    args = ap.parse_args()

    if not args.frames_dir and not args.video_dir:
        print("请指定 --frames-dir 或 --video-dir")
        return

    cfg = load_config(args.config)
    tl = TrafficLightDetector(cfg, verbose=False)
    cand_fn = _hsv_candidates_fn(tl)
    segs_by_video = gt_lookup.load_light_state_csv(args.light_states)
    priors = _load_priors(args.priors)
    os.makedirs(args.out, exist_ok=True)
    labels_path = os.path.join(args.out, "labels.csv")

    if args.frames_dir:
        fds = FrameDataset(args.frames_dir)
        videos = fds.videos()
    else:
        videos = [os.path.splitext(f)[0] for f in sorted(os.listdir(args.video_dir))
                  if f.endswith(".mp4")]

    total = 0
    for video in videos:
        segs = segs_by_video.get(video)
        if not segs:
            print(f"[跳过] {video}: 无 light_states GT 段")
            continue
        state_at_fn = functools.partial(gt_lookup.state_at, segs)
        prior = priors.get(video)
        if args.frames_dir:
            frame_iter = fds.iter_video(video)
        else:
            frame_iter = _video_frames(os.path.join(args.video_dir, f"{video}.mp4"), args.fps)
        tl.__init__(cfg, verbose=False)  # 重置检测器逐视频状态
        cand_fn = _hsv_candidates_fn(tl)
        # 有 prior 的视频走"先验 ROI 直抽": 每帧 prior 位置1个(walk/stand) + prior 外随机1个(off),
        # 正负~1:1, 避免旧路径一帧10+ HSV噪声 off 稀释训练集。无 prior 回退旧路径(全图候选)。
        prior_roi_px = int(prior[2]) if prior and len(prior) > 2 else 160
        rows = extract_crops(frame_iter, video, state_at_fn, cand_fn, args.out,
                             prior=(prior[0], prior[1]) if prior else None,
                             prior_roi_mode=True, prior_roi_px=prior_roi_px)
        write_labels_csv(rows, labels_path)
        auto = sum(1 for r in rows if r["label"] in ("walk", "stand"))
        off = sum(1 for r in rows if r["label"] == "off")
        print(f"[{video}] crops={len(rows)} (walk/stand={auto} off={off}) prior={'有(ROI直抽)' if prior else '无(全图候选, 需人工标)'}")
        total += len(rows)

    print(f"\n完成: {total} crops -> {args.out}  索引: {labels_path}")
    print("下一步: 灯态画廊人工校验 verified=1, 再 python scripts/train_ped_signal.py")


if __name__ == "__main__":
    main()
