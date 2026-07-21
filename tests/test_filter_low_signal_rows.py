"""filter_low_signal_rows.py TDD: 锁定 cc review 的 4 条落地修正。

  (Bug1) 写盘须保相对 crop_path(不调 load_labeled_crops 以免绝对化破跨机)。
  (Bug2) 写盘用输入真实 11 列表头(含 fi), 不丢列。
  (Bug3) 写盘截断非追加(重跑不翻倍)。
  (框架) off 恒保留 / delete·空 丢弃 / 读图失败 unreadable / 阈值单调性。

跑法:
  PYTHONPATH=src:scripts ./.venv/bin/python -m pytest tests/test_filter_low_signal_rows.py -q
"""
import os
import sys
import csv

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from filter_low_signal_rows import (read_rows_relative, filter_low_signal_rows,
                                    write_filtered_csv)


def _img_with(frac, channel, h=20, w=20):
    """灰底 + 前 k 个像素填 channel 色, k=round(h*w*frac)。frac=0 -> 全灰(无信号)。"""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    k = int(round(h * w * frac))
    if k > 0:
        img.reshape(-1, 3)[0:k] = channel
    return img


GREEN = (0, 255, 0)
RED = (0, 0, 255)


def _mk(tmp_path):
    d = tmp_path / "vid"
    d.mkdir()
    crops = {
        "g0.jpg": _img_with(0.0, GREEN),    # walk, 0% 绿
        "g1.jpg": _img_with(0.01, GREEN),   # walk, 1% 绿 (>0.005 保留)
        "r0.jpg": _img_with(0.0, RED),      # stand, 0% 红
        "r5.jpg": _img_with(0.05, RED),     # stand, 5% 红
        "off0.jpg": _img_with(0.0, GREEN),  # off, 0%(恒保留)
        "off5.jpg": _img_with(0.5, GREEN),  # off, 50%(恒保留)
        # miss.jpg 故意不生成 -> 读图失败 unreadable
        "del.jpg": _img_with(0.0, GREEN),   # delete 行
        # empty.jpg 故意不生成 -> 空 label 行
    }
    for name, img in crops.items():
        cv2.imwrite(str(d / name), img)

    rows = [
        {"crop_path": "vid/g0.jpg", "video": "违章02", "frame_ts": "0.0", "fi": "0",
         "x1": "1", "y1": "1", "x2": "2", "y2": "2", "source": "prior_roi", "label": "walk", "verified": "0"},
        {"crop_path": "vid/g1.jpg", "video": "违章02", "frame_ts": "0.2", "fi": "1",
         "x1": "1", "y1": "1", "x2": "2", "y2": "2", "source": "prior_roi", "label": "walk", "verified": "0"},
        {"crop_path": "vid/r0.jpg", "video": "违章02", "frame_ts": "0.4", "fi": "2",
         "x1": "1", "y1": "1", "x2": "2", "y2": "2", "source": "prior_roi", "label": "stand", "verified": "0"},
        {"crop_path": "vid/r5.jpg", "video": "违章02", "frame_ts": "0.6", "fi": "3",
         "x1": "1", "y1": "1", "x2": "2", "y2": "2", "source": "prior_roi", "label": "stand", "verified": "0"},
        {"crop_path": "vid/off0.jpg", "video": "违章02", "frame_ts": "0.8", "fi": "4",
         "x1": "1", "y1": "1", "x2": "2", "y2": "2", "source": "prior_roi", "label": "off", "verified": "0"},
        {"crop_path": "vid/off5.jpg", "video": "违章02", "frame_ts": "1.0", "fi": "5",
         "x1": "1", "y1": "1", "x2": "2", "y2": "2", "source": "prior_roi", "label": "off", "verified": "0"},
        {"crop_path": "vid/miss.jpg", "video": "违章02", "frame_ts": "1.2", "fi": "6",
         "x1": "1", "y1": "1", "x2": "2", "y2": "2", "source": "prior_roi", "label": "walk", "verified": "0"},
        {"crop_path": "vid/del.jpg", "video": "违章02", "frame_ts": "1.4", "fi": "7",
         "x1": "1", "y1": "1", "x2": "2", "y2": "2", "source": "prior_roi", "label": "delete", "verified": "0"},
        {"crop_path": "vid/empty.jpg", "video": "违章02", "frame_ts": "1.6", "fi": "8",
         "x1": "1", "y1": "1", "x2": "2", "y2": "2", "source": "prior_roi", "label": "", "verified": "0"},
    ]
    fieldnames = ["crop_path", "video", "frame_ts", "fi", "x1", "y1", "x2", "y2",
                  "source", "label", "verified"]
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return tmp_path, csv_path, fieldnames


# ------------------------------------------------------------------ 核心过滤
def test_walk_zero_signal_dropped(tmp_path):
    _, csv_path, _ = _mk(tmp_path)
    rows, fn = read_rows_relative(str(csv_path))
    kept, dropped = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    paths = {d["crop_path"] for d in dropped}
    assert "vid/g0.jpg" in paths
    g0 = [d for d in dropped if d["crop_path"] == "vid/g0.jpg"]
    assert g0 and g0[0]["_reason"] == "low_signal"


def test_walk_one_pct_kept(tmp_path):
    _, csv_path, _ = _mk(tmp_path)
    rows, fn = read_rows_relative(str(csv_path))
    kept, _ = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    assert "vid/g1.jpg" in {k["crop_path"] for k in kept}


def test_stand_red_zero_dropped_kept(tmp_path):
    _, csv_path, _ = _mk(tmp_path)
    rows, fn = read_rows_relative(str(csv_path))
    kept, dropped = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    assert "vid/r0.jpg" in {d["crop_path"] for d in dropped}
    assert "vid/r5.jpg" in {k["crop_path"] for k in kept}


def test_off_always_kept(tmp_path):
    _, csv_path, _ = _mk(tmp_path)
    rows, fn = read_rows_relative(str(csv_path))
    kept, _ = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    kept_paths = {k["crop_path"] for k in kept}
    assert "vid/off0.jpg" in kept_paths and "vid/off5.jpg" in kept_paths


def test_delete_and_empty_dropped(tmp_path):
    _, csv_path, _ = _mk(tmp_path)
    rows, fn = read_rows_relative(str(csv_path))
    kept, dropped = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    dropped_paths = {d["crop_path"] for d in dropped}
    assert "vid/del.jpg" in dropped_paths and "vid/empty.jpg" in dropped_paths
    assert "vid/del.jpg" not in {k["crop_path"] for k in kept}


def test_unreadable_dropped(tmp_path):
    _, csv_path, _ = _mk(tmp_path)
    rows, fn = read_rows_relative(str(csv_path))
    kept, dropped = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    miss = [d for d in dropped if d["crop_path"] == "vid/miss.jpg"]
    assert miss and miss[0]["_reason"] == "unreadable" and miss[0]["_signal_ratio"] is None


# ------------------------------------------------------------------ cc 4 条修正
def test_relative_path_preserved_bug1(tmp_path):
    """写盘须保相对 crop_path: 对照原始相对输入, kept 的 crop_path 不得被绝对化。"""
    _, csv_path, _ = _mk(tmp_path)
    with open(str(csv_path), encoding="utf-8-sig", newline="") as f:
        orig = {r["crop_path"] for r in csv.DictReader(f)}  # 原始相对输入
    rows, fn = read_rows_relative(str(csv_path))
    kept, _ = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    for k in kept:
        assert not os.path.isabs(k["crop_path"]), f"crop_path 被绝对化(Bug1): {k['crop_path']}"
        assert k["crop_path"] in orig, "kept crop_path 与原始相对输入不一致"


def test_eleven_col_schema_roundtrip_bug2(tmp_path):
    """写盘用输入真实 11 列表头(含 fi): 回读不得丢列、fi 值须保留。"""
    _, csv_path, fn = _mk(tmp_path)
    rows, fieldnames = read_rows_relative(str(csv_path))
    kept, _ = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    out = str(csv_path.parent / "labels.filtered.csv")
    write_filtered_csv(kept, fieldnames, out)
    with open(out, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        assert "fi" in r.fieldnames, "写盘丢了 fi 列(应为 11 列, Bug2)"
        assert len(r.fieldnames) == 11, f"表头列数错: {r.fieldnames}"
        rows2 = list(r)
    fi_vals = {row["crop_path"]: row["fi"] for row in rows2}
    assert fi_vals.get("vid/g1.jpg") == "1", "fi 值未保留"


def test_no_append_truncate_bug3(tmp_path):
    """写盘截断非追加: 同 out 重跑两次, 行数须 == kept(不翻倍)。"""
    _, csv_path, fn = _mk(tmp_path)
    rows, fieldnames = read_rows_relative(str(csv_path))
    kept, _ = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    out = str(csv_path.parent / "labels.filtered.csv")
    write_filtered_csv(kept, fieldnames, out)
    write_filtered_csv(kept, fieldnames, out)  # 重跑
    with open(out, encoding="utf-8-sig", newline="") as f:
        n = sum(1 for _ in csv.DictReader(f))
    assert n == len(kept), f"重跑后行数翻倍(append bug, Bug3): {n} != {len(kept)}"


def test_threshold_monotonic(tmp_path):
    """阈值越严删越多: 0.02 丢弃数 >= 0.005 丢弃数。"""
    _, csv_path, _ = _mk(tmp_path)
    rows, fn = read_rows_relative(str(csv_path))
    _, dropped005 = filter_low_signal_rows(rows, 0.005, str(csv_path.parent))
    _, dropped020 = filter_low_signal_rows(rows, 0.02, str(csv_path.parent))
    assert len(dropped020) >= len(dropped005)
