"""Phase A 数据集构建 TDD (cc 238b2e3 批准): 无泄漏 / 类平衡(豁免负例+无红GT视频) / impostor 只对非真绿 / 硬覆盖 / ROI 尺度。

跑法: PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_mine_classifier_retrain.py -q

关键修正(对应 cc B 点 + 挖掘脚本 confidence/evidence bug):
  - walk>0 只要求正例视频(01/10 负例无真绿, walk==0 正确)。
  - stand>0 只要求"GT 有 red confirmed 段"的视频(05/08 的 GT 只标了绿段, 无 red, stand==0 正确)。
  - impostor 判定复用 light_states.csv 的 confidence=="confirmed" 语义(不用 expand_light_evidence,
    后者把 confidence 当 evidence 解释, 会恒真通过)。
"""
import os
import sys
import csv
import json

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redlight.evaluation import gt_lookup  # noqa: E402

VIDEOS = [f"违章{i:02d}" for i in range(1, 12)]
NEG_VIDEOS = {"违章01", "违章10"}
POS_VIDEOS = {v for v in VIDEOS if v not in NEG_VIDEOS}
DATA_DIR = os.path.join(ROOT, "datasets", "classifier_retrain")
LABELS = os.path.join(DATA_DIR, "labels.csv")
MANIFEST = os.path.join(DATA_DIR, "manifest.json")
LIGHT_STATES = os.path.join(ROOT, "datasets", "gt", "light_states.csv")

OFF_RATIO_MIN, OFF_RATIO_MAX = 0.20, 0.60


def _load_rows():
    rows = []
    with open(LABELS, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _per_video(rows):
    pv = {v: {"walk": 0, "stand": 0, "off": 0, "impostor": 0,
              "impostor_outside": 0, "n": 0} for v in VIDEOS}
    for r in rows:
        v = r["video"]
        pv.setdefault(v, {"walk": 0, "stand": 0, "off": 0, "impostor": 0,
                          "impostor_outside": 0, "n": 0})
        pv[v][r["label"]] = pv[v].get(r["label"], 0) + 1
        if r["source"] == "impostor":
            pv[v]["impostor"] += 1
        elif r["source"] == "impostor_outside":
            pv[v]["impostor_outside"] += 1
        pv[v]["n"] += 1
    return pv


def _red_confirmed_videos():
    """GT 含 red confirmed 段的视频(这些视频才要求 stand>0)。"""
    segs = gt_lookup.load_light_state_csv(LIGHT_STATES)
    red = set()
    for v, lst in segs.items():
        for a, b, stt, conf in lst:
            if stt == "red" and conf == "confirmed":
                red.add(v)
    return red


def _is_true_green_visible(video, ts):
    """复用 light_states.csv 语义: confirmed green 且可见。"""
    segs = gt_lookup.load_light_state_csv(LIGHT_STATES)
    for a, b, stt, conf in segs.get(video, []):
        if a <= ts <= b:
            return (stt == "green") and (conf == "confirmed")
    return False


def test_dataset_exists():
    assert os.path.isfile(LABELS), f"labels.csv 缺失: {LABELS}"
    assert os.path.isfile(MANIFEST), f"manifest.json 缺失: {MANIFEST}"


def test_no_leakage():
    """整视频进 train 或 val, 不按帧混切; 并集 = 全 11。"""
    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    split = m["split"]
    train, val = set(split["train"]), set(split["val"])
    assert train & val == set(), f"train/val 泄漏: {train & val}"
    assert (train | val) == set(VIDEOS), f"split 未覆盖全 11: 缺 {set(VIDEOS) - (train | val)}"
    rows = _load_rows()
    for r in rows:
        v = r["video"]
        assert (v in train) or (v in val), f"video {v} 不在 split 中"


def test_class_balance_exempt_neg():
    """正例视频 walk>0 且 off>0; 负例 01/10 walk==0(豁免)。
    stand>0 仅要求 GT 有 red confirmed 段(05/08 无红 GT, stand==0 正确)。
    每视频 off 占比 ∈ [0.2,0.6] 仅对"有自然 impostor 供给"(imp+outside>0)的视频生效——
    08 类干净视频引擎从无误绿、无信号外绿斑, 仅背景正则, 强压占比会逼注入假 impostor(不诚实)。
    全局 off 占比必须 ∈ [0.2,0.6](整体数据集目标)。
    """
    rows = _load_rows()
    pv = _per_video(rows)
    red_videos = _red_confirmed_videos()
    tot_w = tot_s = tot_o = 0
    for v in VIDEOS:
        walk, stand, off = pv[v]["walk"], pv[v]["stand"], pv[v]["off"]
        tot_w += walk
        tot_s += stand
        tot_o += off
        if v in NEG_VIDEOS:
            assert walk == 0, f"负例 {v} 不应有 walk(真绿), 实为 {walk}"
        else:
            assert walk > 0, f"正例 {v} 缺 walk(真绿, 尤其 06/07/04)"
        if v in red_videos:
            assert stand > 0, f"{v} GT 有 red confirmed 但缺 stand(真红)"
        assert off > 0, f"{v} 缺 off(impostor)"
        natural_imp = pv[v].get("impostor", 0) + pv[v].get("impostor_outside", 0)
        total = walk + stand + off
        if total > 0 and natural_imp > 0:
            ratio = off / total
            assert OFF_RATIO_MIN <= ratio <= OFF_RATIO_MAX, \
                f"{v} off 占比 {ratio:.2f} 不在 [{OFF_RATIO_MIN},{OFF_RATIO_MAX}] (w={walk} s={stand} o={off})"
    # 全局 off 占比
    g = tot_w + tot_s + tot_o
    gr = tot_o / g
    assert OFF_RATIO_MIN <= gr <= OFF_RATIO_MAX, \
        f"全局 off 占比 {gr:.2f} 不在 [{OFF_RATIO_MIN},{OFF_RATIO_MAX}] (w={tot_w} s={tot_s} o={tot_o})"


def test_impostor_only_on_non_green():
    """impostor 源(engine 读绿但非真绿) 的 crop, GT 在该 ts 不能是 confirmed 绿可见段。"""
    rows = _load_rows()
    checked = 0
    for r in rows:
        if r["source"] not in ("impostor", "impostor_outside"):
            continue
        assert not _is_true_green_visible(r["video"], float(r["frame_ts"])), \
            f"impostor crop 误标在真绿可见段: {r['video']} t={r['frame_ts']}"
        checked += 1
    assert checked > 0, "未挖到任何 impostor(负例假绿/误绿扫描失败)"


def test_hard_coverage():
    """06/07 暗绿 + 04 短绿必须进训练集(walk>0)。"""
    rows = _load_rows()
    pv = _per_video(rows)
    for v in ("违章06", "违章07", "违章04"):
        assert pv[v]["walk"] > 0, f"{v} 暗绿/短绿未进训练集(walk=0)"


def test_roi_scale_consistent():
    """prior_roi 信号 crop 的 bbox 尺寸 == 该视频 prior_roi_px。"""
    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    prior_px = m["prior_roi_px"]
    rows = _load_rows()
    checked = 0
    for r in rows:
        if r["source"] != "prior_roi":
            continue
        w = int(r["x2"]) - int(r["x1"])
        h = int(r["y2"]) - int(r["y1"])
        px = prior_px[r["video"]]
        assert w == px and h == px, \
            f"{r['video']} prior_roi crop 尺寸 {w}x{h} != prior_roi_px {px}"
        checked += 1
    assert checked > 0, "无 prior_roi crop"
