"""预抽帧数据集封装：统一 manifest 读取与帧迭代。

支持两种数据源:
1. manifest.csv: 权威索引，含 video/frame_idx/timestamp/(可选 file_path/gt_plates)
2. 目录扫描: frames_dir/<video>/frame_*.jpg (manifest 缺失时的降级)

用法:
    ds = FrameDataset("datasets/frames")
    for frame_idx, timestamp, frame in ds.iter_video("违章02"):
        results = detector.detect(frame)
"""

import csv
import glob
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..infrastructure.image_utils import robust_imread


class FrameDataset:
    def __init__(self, frames_dir: str):
        """初始化预抽帧数据集。

        Args:
            frames_dir: 预抽帧根目录，期望结构:
                frames_dir/
                  manifest.csv
                  <video>/
                    frame_000001.jpg
                    ...
        """
        self.frames_dir = frames_dir
        self._manifest: Dict[Tuple[str, int], dict] = {}
        self._video_entries: Dict[str, List[dict]] = {}
        self._load_manifest()

    def _load_manifest(self):
        """加载 manifest.csv（如果存在）。"""
        mp = os.path.join(self.frames_dir, "manifest.csv")
        if not os.path.exists(mp):
            return
        with open(mp, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    video = row["video"]
                    frame_idx = int(row["frame_idx"])
                except (KeyError, ValueError):
                    continue
                entry = dict(row)
                entry["frame_idx"] = frame_idx
                try:
                    entry["timestamp"] = float(row.get("timestamp", 0))
                except ValueError:
                    entry["timestamp"] = 0.0
                self._manifest[(video, frame_idx)] = entry
                self._video_entries.setdefault(video, []).append(entry)
        # 每视频按 frame_idx 升序
        for v in self._video_entries:
            self._video_entries[v].sort(key=lambda e: e["frame_idx"])

    def has_manifest(self) -> bool:
        """是否有 manifest.csv。"""
        return bool(self._manifest)

    def iter_video(self, video_name: str) -> Iterator[Tuple[int, float, Optional[np.ndarray]]]:
        """迭代指定视频的所有帧。

        优先使用 manifest 中的 file_path；若无 manifest 或 file_path 缺失，
        则扫描 frames_dir/<video>/frame_*.jpg 并推断 frame_idx。

        Args:
            video_name: 视频名称，如 "违章02"

        Yields:
            (frame_idx, timestamp, frame_array)
            frame_array 为 None 表示图像读取失败
        """
        if video_name in self._video_entries:
            for entry in self._video_entries[video_name]:
                frame_idx = entry["frame_idx"]
                ts = entry.get("timestamp", 0.0)
                # 优先使用 manifest 中的 file_path
                fp = entry.get("file_path")
                if fp and os.path.exists(fp):
                    frame = robust_imread(fp)
                else:
                    # 回退到标准目录结构
                    fp = os.path.join(self.frames_dir, video_name, f"frame_{frame_idx:06d}.jpg")
                    frame = robust_imread(fp) if os.path.exists(fp) else None
                yield frame_idx, ts, frame
            return

        # manifest 无此视频，扫描目录
        vdir = os.path.join(self.frames_dir, video_name)
        if not os.path.isdir(vdir):
            return
        files = sorted(glob.glob(os.path.join(vdir, "frame_*.jpg")))
        for fp in files:
            basename = os.path.basename(fp)
            try:
                frame_idx = int(basename[6:-4])  # frame_000123.jpg -> 123
            except ValueError:
                continue
            # 无 manifest 时 timestamp 无法确定，退化为 0
            frame = robust_imread(fp)
            yield frame_idx, 0.0, frame

    def get_frame(self, video_name: str, frame_idx: int) -> Optional[np.ndarray]:
        """读取指定视频的单帧。

        Args:
            video_name: 视频名称
            frame_idx: 帧索引

        Returns:
            BGR 图像数组，失败返回 None
        """
        # 优先查 manifest
        entry = self._manifest.get((video_name, frame_idx))
        if entry:
            fp = entry.get("file_path")
            if fp and os.path.exists(fp):
                return robust_imread(fp)
        # 回退到标准路径
        fp = os.path.join(self.frames_dir, video_name, f"frame_{frame_idx:06d}.jpg")
        if os.path.exists(fp):
            return robust_imread(fp)
        return None

    def videos(self) -> List[str]:
        """返回有数据的所有视频名列表。"""
        known = set(self._video_entries.keys())
        if os.path.isdir(self.frames_dir):
            for name in os.listdir(self.frames_dir):
                vdir = os.path.join(self.frames_dir, name)
                if os.path.isdir(vdir):
                    known.add(name)
        return sorted(known)
