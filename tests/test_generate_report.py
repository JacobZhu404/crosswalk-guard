#!/usr/bin/env python3
"""generate_report.py 报告层单测(车牌归一化 / ✓✗ 汇总 / GT 加载 / 聚合一致)。

复用 redlight.evaluation.violation_eval 的 match_violation_events, 这里只测报告层
自有逻辑: normalize_plate + summarize_video/_flags 的子标记 + 并列 GT 加载器。
"""
import os
import sys
import csv
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import generate_report as gr


class TestNormalizePlate(unittest.TestCase):
    def test_spaces(self):
        self.assertEqual(gr.normalize_plate("京 LNE 560"), "京LNE560")

    def test_hyphen(self):
        self.assertEqual(gr.normalize_plate("ABC-123"), "ABC123")

    def test_dot_bullet(self):
        self.assertEqual(gr.normalize_plate("粤B·1234"), "粤B1234")

    def test_lower_to_upper(self):
        self.assertEqual(gr.normalize_plate("abc123"), "ABC123")

    def test_empty_and_none(self):
        self.assertEqual(gr.normalize_plate(""), "")
        self.assertEqual(gr.normalize_plate(None), "")


class TestSummarizePositive(unittest.TestCase):
    def test_tp_with_plate_hit(self):
        pred = [{"start_ts": 0.0, "end_ts": 10.0, "plate": "京 LNE 560"}]
        gt = [{"start_s": 0.0, "end_s": 12.0, "plates": ["京LNE560"]}]
        res = gr.summarize_video(pred, gt, has_violation=True, min_overlap_s=0.5)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertTrue(res["is_violation_hit"])
        self.assertIn("is_violation✓", res["flags"])
        self.assertEqual(res["plate_hits"], 1)
        self.assertEqual(res["plate_total"], 1)

    def test_fn_leak(self):
        # 正例但管线没抓到 -> fn
        pred = []
        gt = [{"start_s": 0.0, "end_s": 12.0, "plates": ["京LNE560"]}]
        res = gr.summarize_video(pred, gt, has_violation=True, min_overlap_s=0.5)
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fn"], 1)
        self.assertFalse(res["is_violation_hit"])
        self.assertIn("is_violation✗(漏)", res["flags"])

    def test_fp_on_positive(self):
        # 正例 GT 存在, 但预测事件时间不重叠 GT 窗 -> fp(碎片/窗外)
        pred = [{"start_ts": 100.0, "end_ts": 110.0, "plate": "X"}]
        gt = [{"start_s": 0.0, "end_s": 12.0, "plates": ["京LNE560"]}]
        res = gr.summarize_video(pred, gt, has_violation=True, min_overlap_s=0.5)
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)


class TestSummarizeNegative(unittest.TestCase):
    def test_negative_no_pred(self):
        pred = []
        gt = []
        res = gr.summarize_video(pred, gt, has_violation=False, min_overlap_s=0.5)
        self.assertEqual(res["fp"], 0)
        self.assertIn("负例✓(未发现违章)", res["flags"])

    def test_negative_false_positive(self):
        # 负例视频却检出 confirmed -> 真误报(neg_true_fp)
        pred = [{"start_ts": 0.0, "end_ts": 10.0, "plate": "X"}]
        gt = []
        res = gr.summarize_video(pred, gt, has_violation=False, min_overlap_s=0.5)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["neg_count"], 1)
        self.assertIn("负例误报✗", res["flags"])


class TestGtLoaders(unittest.TestCase):
    def test_source_annotation(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["input_video", "gt_description", "result"])
            w.writerow(["违章01", "全程红灯,无违章", ""])
            w.writerow(["违章02", "绿灯占道违章", ""])
            path = f.name
        try:
            d = gr.load_source_annotation(path)
            self.assertEqual(d["违章01"], "全程红灯,无违章")
            self.assertEqual(d["违章02"], "绿灯占道违章")
        finally:
            os.remove(path)

    def test_events_rows(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "start_s", "end_s", "is_violation", "violating_plates", "note"])
            w.writerow(["违章02", "21", "68", "1", "京LNE560", "绿灯占道"])
            w.writerow(["违章02", "0", "21", "0", "", "红灯不算"])
            path = f.name
        try:
            rows = gr.load_events_rows(path)
            self.assertEqual(len(rows), 2)
            viol = [r for r in rows if r["video"] == "违章02" and r["is_violation"] == "1"]
            self.assertEqual(len(viol), 1)
            self.assertEqual(viol[0]["violating_plates"], "京LNE560")
        finally:
            os.remove(path)


class TestAggregateConsistency(unittest.TestCase):
    def _sum(self, pred, gt, has_v):
        return gr.summarize_video(pred, gt, has_v, min_overlap_s=0.5)

    def test_aggregate_8tp_1fp_1fn_shape(self):
        # 构造 10 正例(8 tp + 1 fn + 1 fp) + 1 负例(0 fp) = 8tp/1fp/1fn
        results = []
        # 8 个完美 tp
        for _ in range(8):
            results.append(self._sum(
                [{"start_ts": 0, "end_ts": 10, "plate": "A"}],
                [{"start_s": 0, "end_s": 12, "plates": ["A"]}], True))
        # 1 个 fn(漏)
        results.append(self._sum(
            [], [{"start_s": 0, "end_s": 12, "plates": ["A"]}], True))
        # 1 个 fp(负例视频却检出 confirmed, 无 GT -> 真误报, fn=0)
        results.append(self._sum(
            [{"start_ts": 100, "end_ts": 110, "plate": "B"}],
            [], False))
        # 1 个负例无误报
        results.append(self._sum([], [], False))
        agg = gr.aggregate(results)
        self.assertEqual((agg["tp"], agg["fp"], agg["fn"]), (8, 1, 1))
        self.assertEqual(agg["true_fp_total"], 1)   # 唯一 fp 是负例真误报
        self.assertEqual(agg["fragment_total"], 0)


if __name__ == "__main__":
    unittest.main()
