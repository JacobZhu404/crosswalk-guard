#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_gt.py 单元测试（构造微型输入，不依赖 593 行真实数据）。"""
import os
import sys
import unittest

# 把 scripts/ 加入 path 以便直接 import build_gt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import build_gt as bg  # noqa: E402


def mk_seg(start, end, state, confidence="confirmed", note="base"):
    return {"start": start, "end": end, "state": state,
            "confidence": confidence, "note": note}


def mk_point(t, gt, src="feedback", video="V", reason="r", verdict="algo_wrong", note="n"):
    return {"t": t, "gt": gt, "src": src, "video": video,
            "reason": reason, "verdict": verdict, "note": note, "row": {}}


class TestMergeOverridePoints(unittest.TestCase):
    def test_merge_same_gt_within_tol(self):
        # 真实反馈约 0.23s/帧；用 0.3s 间距模拟（< tol 0.5）
        pts = [mk_point(5.0, "green"), mk_point(5.3, "green"), mk_point(5.6, "green")]
        ivs = bg.merge_override_points(pts, gap_tol=0.5, margin=0.1)
        self.assertEqual(len(ivs), 1)
        self.assertAlmostEqual(ivs[0]["t0"], 4.9, places=5)
        self.assertAlmostEqual(ivs[0]["t1"], 5.7, places=5)
        self.assertEqual(ivs[0]["gt"], "green")
        self.assertEqual(ivs[0]["n"], 3)

    def test_split_different_gt(self):
        pts = [mk_point(5.0, "green"), mk_point(6.0, "red")]  # 间隔> tol 且 gt 不同
        ivs = bg.merge_override_points(pts, gap_tol=0.5, margin=0.1)
        self.assertEqual(len(ivs), 2)
        self.assertEqual(ivs[0]["gt"], "green")
        self.assertEqual(ivs[1]["gt"], "red")


class TestAggression(unittest.TestCase):
    def test_agg1_feedback_green_on_red_base(self):
        base = [mk_seg(0, 20, "red")]
        # 真实反馈约 0.23s/帧，用 <tol 间距；合并后区间 [4.9,5.7]
        pts = [mk_point(5.0, "green"), mk_point(5.3, "green"), mk_point(5.6, "green")]
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=1)
        segs = res["segments"]
        self.assertEqual(len(segs), 3)
        self.assertEqual((segs[0]["start"], segs[0]["state"]), (0, "red"))
        self.assertEqual(segs[1]["state"], "green")
        self.assertAlmostEqual(segs[1]["start"], 4.9, places=4)
        self.assertAlmostEqual(segs[1]["end"], 5.7, places=4)
        self.assertEqual(segs[2]["state"], "red")
        self.assertEqual(segs[0]["src"], "base")
        self.assertEqual(segs[1]["src"], "feedback")

    def test_agg2_isolated_tentative(self):
        base = [mk_seg(0, 20, "red")]
        pts = [mk_point(10.0, "green")]  # 单点 → n=1 < K=3 → feedback-tentative
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=1)
        segs = res["segments"]
        green = [s for s in segs if s["state"] == "green"][0]
        self.assertEqual(green["confidence"], "feedback-tentative")
        self.assertAlmostEqual(green["start"], 9.9, places=4)
        self.assertAlmostEqual(green["end"], 10.1, places=4)

    def test_agg3_feedback_eq_base_no_change(self):
        base = [mk_seg(0, 20, "red")]
        pts = [mk_point(10.0, "red")]  # gt == base → 不改写
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=1)
        self.assertEqual(len(res["segments"]), 1)
        self.assertEqual(res["segments"][0]["src"], "base")
        self.assertEqual(res["changed"], [])


class TestRewriteAlign(unittest.TestCase):
    def test_no_note_pollution_when_gt_eq_base(self):
        base = [mk_seg(0, 20, "red", note="orig note")]
        pts = [mk_point(10.0, "red")]  # gt == base → 不应写入 merge 溯源
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=1)
        self.assertNotIn("merge(feedback)", res["segments"][0]["note"])
        self.assertEqual(res["segments"][0]["note"], "orig note")


class TestPriority(unittest.TestCase):
    def test_feedback_over_base(self):
        base = [mk_seg(0, 20, "red")]
        fb = bg.merge_override_points([mk_point(10.0, "green")])
        res = bg.build_light_video(base, fb, has_violation=1)
        self.assertEqual([s["state"] for s in res["segments"] if s["src"] == "feedback"][0], "green")

    def test_badcase_over_feedback(self):
        base = [mk_seg(0, 20, "red")]
        fb = bg.merge_override_points([mk_point(10.0, "green", src="feedback")])
        bc = bg.merge_override_points([mk_point(10.0, "red", src="badcase")])
        res = bg.build_light_video(base, fb + bc, has_violation=1)
        # badcase 顶替 feedback：最终该 span 应为 red(badcase)
        changed = [s for s in res["segments"] if s["src"] != "base"]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["state"], "red")
        self.assertEqual(changed[0]["src"], "badcase")

    def test_idempotent_no_input(self):
        base = [mk_seg(0, 20, "red", note="a"), mk_seg(20, 40, "green", note="b")]
        res = bg.build_light_video(base, [], has_violation=1)
        # 无输入时输出 == base（仅附加内部 src="base" 字段，业务字段一致）
        self.assertEqual([(s["start"], s["end"], s["state"], s["note"]) for s in res["segments"]],
                         [(s["start"], s["end"], s["state"], s["note"]) for s in base])
        self.assertEqual(res["changed"], [])


class TestProvenance(unittest.TestCase):
    def test_note_contains_merge_ref(self):
        base = [mk_seg(0, 20, "red", note="base note")]
        pts = [mk_point(10.0, "green", video="违章02", reason="reading_point")]
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=1)
        ch = res["changed"][0]
        self.assertIn("merge(feedback)", ch["note"])
        self.assertIn("违章02", ch["note"])
        self.assertIn("gt=green", ch["note"])
        self.assertIn("reading_point", ch["note"])
        # 审计：old→new 可映射
        self.assertEqual(ch["old_state"], "red")
        self.assertEqual(ch["src_rows"][0]["gt"], "green")


class TestGuardNegative(unittest.TestCase):
    def test_negative_flip_to_green_aborts(self):
        base = [mk_seg(0, 20, "red")]  # has_violation=0 视频
        pts = [mk_point(10.0, "green")]  # 给负例标 green → 应 abort
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=0)
        self.assertTrue(res["aborted"])
        self.assertFalse(res["neg_ok"])
        self.assertEqual(len(res["neg_offenders"]), 1)

    def test_negative_red_ok(self):
        base = [mk_seg(0, 20, "red")]
        pts = [mk_point(10.0, "red")]  # gt==base，无变更
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=0)
        self.assertTrue(res["neg_ok"])
        self.assertFalse(res["aborted"])


class TestGuardHighRisk(unittest.TestCase):
    def test_high_risk_flip_on_violation_video(self):
        base = [mk_seg(0, 20, "green")]  # has_violation=1
        pts = [mk_point(10.0, "red")]     # green→red 翻转
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=1)
        self.assertEqual(len(res["high_risk"]), 1)
        self.assertEqual(res["high_risk"][0]["old_state"], "green")
        self.assertEqual(res["high_risk"][0]["new_state"], "red")

    def test_low_risk_red_unknown_on_negative_no_block(self):
        # has_violation=0 视频发生 red→unknown（不产生 green）→ 负例断言通过，不进高危
        base = [mk_seg(0, 20, "red")]
        pts = [mk_point(10.0, "unknown")]
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=0)
        self.assertTrue(res["neg_ok"])
        self.assertEqual(len(res["high_risk"]), 0)
        self.assertEqual(len(res["changed"]), 1)

    def test_high_risk_not_triggered_for_unknown_red_on_violation(self):
        # green 未参与：red→unknown 不撑起违章资格变化 → 非高危
        base = [mk_seg(0, 20, "red")]
        pts = [mk_point(10.0, "unknown")]
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=1)
        self.assertEqual(len(res["high_risk"]), 0)

    def test_clamp_removes_boundary_artifact(self):
        # feedback 绿簇整体落在同态绿 base 内 → 仅确认，零变更、零高危
        # 模拟 违章02：base [0,20.9]red [20.9,85]green；绿反馈 21/22/23（簇在绿窗内）
        base = [mk_seg(0, 20.9, "red"), mk_seg(20.9, 85, "green")]
        pts = [mk_point(21.0, "green"), mk_point(22.0, "green"), mk_point(23.0, "green")]
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=1)
        self.assertEqual(res["changed"], [])
        self.assertEqual(res["high_risk"], [])

    def test_genuine_correction_still_applies(self):
        # 真校正：绿反馈簇不在任何绿 base 内（base 全红）→ 仍覆盖为绿（高危待签）
        base = [mk_seg(0, 20, "red")]
        pts = [mk_point(5.0, "green"), mk_point(5.3, "green"), mk_point(5.6, "green")]
        ivs = bg.merge_override_points(pts)
        res = bg.build_light_video(base, ivs, has_violation=1)
        self.assertEqual(len(res["changed"]), 1)
        self.assertEqual(res["high_risk"][0]["old_state"], "red")
        self.assertEqual(res["high_risk"][0]["new_state"], "green")


class TestLoadLightStates(unittest.TestCase):
    def _write_tmp(self, lines):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".csv", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def test_unquote_robustness_malformed_and_quoted(self):
        # 同一 note 用三种写法，读出应得到相同内容（目标段放在 index 1）
        note = "人工画廊标注:48-62.5s信号灯被遮挡(60帧other),系统判unknown正确"
        lines1 = ["video,start_s,end_s,state,confidence,note",
                  "违章01,0,48,red,confirmed," + note,
                  '违章01,48,62.5,red,occluded,"' + note + '"',
                  '违章01,62.5,999,red,confirmed,' + note]
        lines3 = ["video,start_s,end_s,state,confidence,note",
                  "违章01,0,48,red,confirmed," + note,
                  '违章01,48,62.5,red,occluded,""' + note + '""',
                  '违章01,62.5,999,red,confirmed,' + note]
        for lines in (lines1, lines3):
            p = self._write_tmp(lines)
            segs = bg.load_light_states(p)
            os.remove(p)
            self.assertEqual(segs["违章01"][1]["note"], note)

    def test_idempotent_rewrite_no_double_quote(self):
        # 含逗号 note 经 loader 读出再 csv 写回：raw 行应为单引号且内容无双重引号
        note = "人工画廊标注:48-62.5s信号灯被遮挡(60帧other),系统判unknown正确"
        import tempfile, os
        p = self._write_tmp(["video,start_s,end_s,state,confidence,note",
                             "违章01,0,48,red,confirmed," + note])
        segs = bg.load_light_states(p)
        out = os.path.join(tempfile.gettempdir(), "bg_out_test.csv")
        bg.write_light_csv({"违章01": segs["违章01"]}, out)
        with open(out, encoding="utf-8") as f:
            raw = f.read().splitlines()[1]  # 第二行（首行 header）
        os.remove(p); os.remove(out)
        # raw 行应含单引号包裹的 note，且不含双重引号
        self.assertIn('"' + note + '"', raw)
        self.assertNotIn('""' + note, raw)


class TestPolyIoU(unittest.TestCase):
    def test_identical(self):
        p = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertAlmostEqual(bg.poly_iou(p, p), 1.0, places=5)

    def test_disjoint(self):
        p1 = [[0, 0], [1, 0], [1, 1], [0, 1]]
        p2 = [[5, 5], [6, 5], [6, 6], [5, 6]]
        self.assertEqual(bg.poly_iou(p1, p2), 0.0)

    def test_partial(self):
        p1 = [[0, 0], [2, 0], [2, 2], [0, 2]]
        p2 = [[1, 1], [3, 1], [3, 3], [1, 3]]
        # 交叠 [1,2]x[1,2]=1，并=4+4-1=7 → 1/7
        self.assertAlmostEqual(bg.poly_iou(p1, p2), 1.0 / 7.0, places=5)

    def test_clockwise_poly_iou_identical(self):
        # CW 朝向（负有向面积）多边形：修复前误报 0
        p = [[718, 697], [435, 251], [3, 279], [7, 713]]  # 与 base 同坐标，CW
        self.assertAlmostEqual(bg.poly_iou(p, p), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
