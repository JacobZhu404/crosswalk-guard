"""FrameDataset 单元测试。

覆盖: manifest 加载、按视频迭代、单帧读取、视频列表。
使用 mock 隔离 cv2.imdecode。
"""
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation.frame_dataset import FrameDataset


def _make_manifest(frames_dir, rows):
    mp = os.path.join(frames_dir, "manifest.csv")
    with open(mp, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["video", "frame_idx", "timestamp", "file_path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_frame_file(vdir, frame_idx):
    os.makedirs(vdir, exist_ok=True)
    fp = os.path.join(vdir, f"frame_{frame_idx:06d}.jpg")
    with open(fp, "wb") as f:
        f.write(b"fakejpg")
    return fp


class TestLoadManifest:
    def test_basic(self):
        with tempfile.TemporaryDirectory() as td:
            _make_manifest(td, [
                {"video": "违章02", "frame_idx": "0", "timestamp": "0.0", "file_path": ""},
                {"video": "违章02", "frame_idx": "10", "timestamp": "1.25", "file_path": ""},
                {"video": "违章03", "frame_idx": "5", "timestamp": "0.625", "file_path": ""},
            ])
            ds = FrameDataset(td)
            assert ds.has_manifest()
            assert "违章02" in ds.videos()
            assert "违章03" in ds.videos()

    def test_sorting(self):
        with tempfile.TemporaryDirectory() as td:
            _make_manifest(td, [
                {"video": "违章02", "frame_idx": "20", "timestamp": "2.5", "file_path": ""},
                {"video": "违章02", "frame_idx": "5", "timestamp": "0.625", "file_path": ""},
                {"video": "违章02", "frame_idx": "10", "timestamp": "1.25", "file_path": ""},
            ])
            ds = FrameDataset(td)
            entries = ds._video_entries["违章02"]
            assert [e["frame_idx"] for e in entries] == [5, 10, 20]

    def test_no_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "违章02"))
            ds = FrameDataset(td)
            assert not ds.has_manifest()
            assert ds.videos() == ["违章02"]


class TestIterVideo:
    @patch("redlight.evaluation.frame_dataset.robust_imread")
    def test_with_manifest(self, mock_read):
        mock_read.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            vdir = os.path.join(td, "违章02")
            fp0 = _make_frame_file(vdir, 0)
            fp10 = _make_frame_file(vdir, 10)
            _make_manifest(td, [
                {"video": "违章02", "frame_idx": "0", "timestamp": "0.0", "file_path": fp0},
                {"video": "违章02", "frame_idx": "10", "timestamp": "1.25", "file_path": fp10},
            ])
            ds = FrameDataset(td)
            results = list(ds.iter_video("违章02"))
            assert len(results) == 2
            assert results[0] == (0, 0.0, mock_read.return_value)
            assert results[1] == (10, 1.25, mock_read.return_value)
            # 应使用 manifest 中的 file_path
            assert mock_read.call_args_list[0][0][0] == fp0

    @patch("redlight.evaluation.frame_dataset.robust_imread")
    def test_without_manifest(self, mock_read):
        mock_read.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            vdir = os.path.join(td, "违章02")
            _make_frame_file(vdir, 5)
            _make_frame_file(vdir, 15)
            ds = FrameDataset(td)
            results = list(ds.iter_video("违章02"))
            assert len(results) == 2
            assert results[0][0] == 5
            assert results[1][0] == 15
            # 无 manifest 时 timestamp 为 0
            assert results[0][1] == 0.0

    def test_missing_video(self):
        with tempfile.TemporaryDirectory() as td:
            ds = FrameDataset(td)
            assert list(ds.iter_video("不存在的视频")) == []


class TestGetFrame:
    @patch("redlight.evaluation.frame_dataset.robust_imread")
    def test_from_manifest(self, mock_read):
        mock_read.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            vdir = os.path.join(td, "违章02")
            fp = _make_frame_file(vdir, 42)
            _make_manifest(td, [
                {"video": "违章02", "frame_idx": "42", "timestamp": "5.25", "file_path": fp},
            ])
            ds = FrameDataset(td)
            frame = ds.get_frame("违章02", 42)
            assert frame is mock_read.return_value
            mock_read.assert_called_once_with(fp)

    @patch("redlight.evaluation.frame_dataset.robust_imread")
    def test_fallback_path(self, mock_read):
        mock_read.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            vdir = os.path.join(td, "违章02")
            fp = _make_frame_file(vdir, 7)
            ds = FrameDataset(td)  # 无 manifest
            frame = ds.get_frame("违章02", 7)
            assert frame is mock_read.return_value
            mock_read.assert_called_once_with(fp)

    def test_missing_frame(self):
        with tempfile.TemporaryDirectory() as td:
            ds = FrameDataset(td)
            assert ds.get_frame("违章02", 999) is None


class TestVideoDuration:
    def test_from_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            _make_manifest(td, [
                {"video": "违章02", "frame_idx": "0", "timestamp": "0.0", "file_path": ""},
                {"video": "违章02", "frame_idx": "10", "timestamp": "1.25", "file_path": ""},
                {"video": "违章02", "frame_idx": "20", "timestamp": "2.50", "file_path": ""},
            ])
            ds = FrameDataset(td)
            assert ds.video_duration("违章02") == 2.50

    def test_no_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "违章02"))
            ds = FrameDataset(td)
            assert ds.video_duration("违章02") == 0.0

    def test_missing_video(self):
        with tempfile.TemporaryDirectory() as td:
            ds = FrameDataset(td)
            assert ds.video_duration("不存在") == 0.0


class TestVideos:
    def test_combines_manifest_and_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "违章02"))
            _make_manifest(td, [
                {"video": "违章03", "frame_idx": "0", "timestamp": "0.0", "file_path": ""},
            ])
            ds = FrameDataset(td)
            assert ds.videos() == ["违章02", "违章03"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
