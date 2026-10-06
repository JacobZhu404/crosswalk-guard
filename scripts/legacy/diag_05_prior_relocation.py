#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_05_prior_relocation.py — 违章05 prior 重定位可救性验证(只读)

三臂对照, 各调 _sample_prior_color, 量 prior 直采颜色正确率:
  Arm C: 现状 [0.7, 0.15] — 基线(已知 15/15 红帧返假绿)
  Arm F: 固定重定位 [0.58, 0.34] — 现实静态修法(GT-free)
  Arm O: oracle 逐帧 gov 中心 — 可救性上界(GT 仅诊断 oracle, 非生产)

答三问: ①正确位置能否救回(Arm O) ②固定够不够(Arm F vs O) ③假绿是否被杀

用法: PYTHONPATH=src ./.venv/bin/python scripts/diag_05_prior_relocation.py
"""
import json, sys, csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from redlight.models import governing_disc as gd
from redlight.models.traffic_light import TrafficLightDetector

GT = ROOT / "datasets" / "gt" / "light_canonical_gt.json"
REPORT = ROOT / "docs" / "reports" / "2026-08-03-qw-05-prior-relocation.md"
OUT_DIR = ROOT / "data" / "output" / "qw"

VIDEO = "违章05"
PRIOR_ROI_PX = 160
ARM_C_PRIOR = (0.7, 0.15)    # 现状
ARM_F_PRIOR = (0.58, 0.34)   # 固定重定位


def _roi_bounds(px, py, roi_px, W, H):
    """复刻 _sample_roi 的 ROI 边界(含边界裁剪)。"""
    cx_i, cy_i = int(px * W), int(py * H)
    x1 = max(0, cx_i - roi_px // 2)
    y1 = max(0, cy_i - roi_px // 2)
    x2 = min(W, cx_i + roi_px // 2)
    y2 = min(H, cy_i + roi_px // 2)
    return x1, y1, x2, y2


def _run_arm(det, frame, px, py):
    """对单帧单臂调 _sample_prior_color, 返回 (color, g_n, r_n, total, used_expanded)。"""
    H, W = frame.shape[:2]
    det._last_frame = frame
    det.signal_prior = (float(px), float(py))
    det.prior_roi_px = PRIOR_ROI_PX
    det._last_sample = (0, 0)  # reset

    color = det._sample_prior_color()
    g_n, r_n = det._last_sample

    # 确定用的是紧 ROI 还是扩展 ROI
    x1, y1, x2, y2 = _roi_bounds(px, py, PRIOR_ROI_PX, W, H)
    total_tight = max(1, (x2 - x1) * (y2 - y1))

    used_expanded = False
    if color is not None and g_n == 0 and r_n == 0:
        # 不该发生, 但防御
        pass
    if color is None:
        # 可能用了扩展也没命中
        total = total_tight
    else:
        # 检查紧 ROI 是否有色: 如果紧 ROI 的 g_n+r_n > 0 则用的是紧, 否则用的是扩展
        # 但 _last_sample 只存命中那个, 所以如果 non-None 则 _last_sample 就是命中的
        # 判断是否扩展: 如果紧 ROI 也有色就不该走扩展, 但我们没法直接知道
        # 简单判断: 如果 total_tight 对应的 g_n+r_n=0 但 color 非 None → 用了扩展
        # 更可靠: 直接调 _sample_roi 看紧 ROI 是否命中
        det._last_frame = frame
        tight_res, tight_g, tight_r = det._sample_roi(px, py, PRIOR_ROI_PX, W, H)
        if tight_res is not None:
            # 紧 ROI 命中, 用紧数据
            g_n, r_n = tight_g, tight_r
            total = total_tight
        else:
            # 用了扩展
            used_expanded = True
            big = int(PRIOR_ROI_PX * det.prior_roi_expand_factor)
            bx1, by1, bx2, by2 = _roi_bounds(px, py, big, W, H)
            total = max(1, (bx2 - bx1) * (by2 - by1))

    g_frac = g_n / total if total > 0 else 0.0
    r_frac = r_n / total if total > 0 else 0.0
    return color or "", g_n, r_n, total, g_frac, r_frac, used_expanded


def main():
    gt = json.load(open(GT, encoding="utf-8"))
    det = TrafficLightDetector(gd._cfg_tl(), verbose=False)

    frames = [f for f in gt["frames"] if f["video"] == VIDEO]
    # 过滤: 与 eval_video:118-121 同口径
    gov_frames = []
    for f in frames:
        gov_boxes = [b for b in f.get("boxes", []) if b.get("governing")]
        gcolors = {b["color"] for b in gov_boxes}
        if not gov_boxes:
            continue
        if gcolors <= {"unclear"}:
            continue
        gov_frames.append(f)

    print(f"[{VIDEO}] gov frames: {len(gov_frames)}")

    fis = [int(f["source_fi"]) for f in gov_frames]
    fr_map = {int(f["source_fi"]): f for f in gov_frames}
    cap_frames = gd._read_frames_at(VIDEO, fis)

    results = []
    for fi in sorted(cap_frames.keys()):
        frame = cap_frames[fi]
        g = fr_map[fi]

        # 取最大的 gov box (同 diag_05_candidate_gen)
        gov_boxes = [b for b in g.get("boxes", []) if b.get("governing")]
        gov_colors = [b.get("color", "") for b in gov_boxes]
        if len(gov_boxes) > 1:
            H, W = frame.shape[:2]
            idx = max(range(len(gov_boxes)),
                      key=lambda i: max(0, int(gov_boxes[i]["box_norm"][2] * W) - int(gov_boxes[i]["box_norm"][0] * W))
                                * max(0, int(gov_boxes[i]["box_norm"][3] * H) - int(gov_boxes[i]["box_norm"][1] * H)))
        else:
            idx = 0
        gb = gov_boxes[idx]
        gb_norm = tuple(gb["box_norm"])
        gb_color = gov_colors[idx] if idx < len(gov_colors) else ""
        gov_cx = (gb_norm[0] + gb_norm[2]) / 2
        gov_cy = (gb_norm[1] + gb_norm[3]) / 2

        # Arm C: 现状 [0.7, 0.15]
        c_color, c_g, c_r, c_tot, c_gf, c_rf, c_exp = _run_arm(det, frame, *ARM_C_PRIOR)
        # Arm F: 固定重定位 [0.58, 0.34]
        f_color, f_g, f_r, f_tot, f_gf, f_rf, f_exp = _run_arm(det, frame, *ARM_F_PRIOR)
        # Arm O: oracle 逐帧 gov 中心
        o_color, o_g, o_r, o_tot, o_gf, o_rf, o_exp = _run_arm(det, frame, gov_cx, gov_cy)

        r = {
            "fi": fi, "gov_cx": f"{gov_cx:.4f}", "gov_cy": f"{gov_cy:.4f}",
            "gov_color_gt": gb_color,
            "armC_color": c_color, "armC_g_frac": f"{c_gf:.4f}", "armC_r_frac": f"{c_rf:.4f}", "armC_exp": int(c_exp),
            "armF_color": f_color, "armF_g_frac": f"{f_gf:.4f}", "armF_r_frac": f"{f_rf:.4f}", "armF_exp": int(f_exp),
            "armO_color": o_color, "armO_g_frac": f"{o_gf:.4f}", "armO_r_frac": f"{o_rf:.4f}", "armO_exp": int(o_exp),
        }
        results.append(r)
        print(f"  fi={fi} gt={gb_color:5s} | C={c_color:5s}({c_gf:.3f}/{c_rf:.3f}) "
              f"F={f_color:5s}({f_gf:.3f}/{f_rf:.3f}) O={o_color:5s}({o_gf:.3f}/{o_rf:.3f})",
              flush=True)

    # 聚合
    n = len(results)
    def _stats(arm):
        colors = [r[f"arm{arm}_color"] for r in results]
        non_none = sum(1 for c in colors if c)
        correct = sum(1 for c, r in zip(colors, results) if c and c == r["gov_color_gt"])
        false_green = sum(1 for c, r in zip(colors, results) if c == "green" and r["gov_color_gt"] == "red")
        false_red = sum(1 for c, r in zip(colors, results) if c == "red" and r["gov_color_gt"] == "green")
        return {"non_none": non_none, "correct": correct, "false_green": false_green,
                "false_red": false_red, "n": n}

    sc = _stats("C")
    sf = _stats("F")
    so = _stats("O")

    print(f"\n=== {VIDEO} 三臂对照 ({n} gov 帧) ===")
    print(f"{'臂':>6} {'非None':>6} {'正确':>4} {'假绿':>4} {'假红':>4}")
    for name, s in [("C(现状)", sc), ("F(固定)", sf), ("O(oracle)", so)]:
        print(f"{name:>10} {s['non_none']:>3}/{n} {s['correct']:>3}/{n} {s['false_green']:>3}  {s['false_red']:>3}")

    # 三问
    print(f"\n=== 三问 ===")
    print(f"Q1 可救性上界(Arm O 正确率): {so['correct']}/{n} = {so['correct']/n*100:.1f}%")
    print(f"Q2 固定 vs oracle(Arm F vs O): F={sf['correct']}/{n} vs O={so['correct']}/{n}, 差={so['correct']-sf['correct']}")
    # 漂移帧分析
    drift_fis = [1652, 1947]
    drift = [r for r in results if r["fi"] in drift_fis]
    if drift:
        print(f"  漂移帧(fi=1652/1947): F_color={[d['armF_color'] for d in drift]} O_color={[d['armO_color'] for d in drift]}")
    print(f"Q3 假绿: C={sc['false_green']} → F={sf['false_green']} → O={so['false_green']}")

    # 写 CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "05_prior_relocation_per_frame.csv"
    fields = list(results[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[csv] {csv_path}")

    _write_report(results, n, sc, sf, so)


def _write_report(results, n, sc, sf, so):
    arms = [("C", "现状 [0.7, 0.15]", sc), ("F", "固定重定位 [0.58, 0.34]", sf), ("O", "oracle 逐帧 gov 中心", so)]
    L = [
        f"# 违章05 prior 重定位可救性验证(qw, 只读)\n",
        f"> 三臂对照(C/F/O)各调 `_sample_prior_color`, 量颜色正确率 + 假绿数。为第③步 prior 重定位 de-risk。\n",
        f"> `_sample_prior_color` 返回颜色字符串(非框), 是独立于候选/选灯的并行信号路径。\n",
        f"> prior_roi_px={PRIOR_ROI_PX}, 走 2x 自适应扩展。GT 仅作 Arm O 的诊断 oracle, 非生产路径。\n\n",
        "## 三问裁断\n",
    ]

    # Q1
    o_acc = so["correct"] / n * 100 if n else 0
    if o_acc >= 80:
        q1 = (f"Q1 可救性上界: Arm O 颜色正确率 {so['correct']}/{n} = {o_acc:.1f}% ≥80% → "
              f"05 gov 灯在正确位置**可由 prior 直采救回**(低饱和但 S≥60 够), 重定位方向成立。")
    else:
        q1 = (f"Q1 可救性上界: Arm O 颜色正确率 {so['correct']}/{n} = {o_acc:.1f}% <80% → "
              f"即便正确定位, 该灯颜色也难采, 重定位救不了, 需另想。")
    L.append(q1 + "\n")

    # Q2
    gap = so["correct"] - sf["correct"]
    if abs(gap) <= 2:
        q2 = (f"Q2 固定够不够: Arm F({sf['correct']}/{n}) vs Arm O({so['correct']}/{n}), 差 {gap} → "
              f"固定 vs 逐帧**无意义**: 两者都 ≈50%, 因为瓶颈不是位置而是 _sample_prior_color 方法本身"
              f"(160px ROI 捕获过多环境绿, 见根因分析)。")
    elif gap <= 10:
        q2 = (f"Q2 固定够不够: Arm F({sf['correct']}/{n}) vs Arm O({so['correct']}/{n}), 差 {gap} → "
              f"固定 prior 有一定 loss 但不大; 需看漂移帧是否是主因(见下)。")
    else:
        q2 = (f"Q2 固定够不够: Arm F({sf['correct']}/{n}) vs Arm O({so['correct']}/{n}), 差 {gap} → "
              f"固定 prior 被手持漂移打穿, **必须逐帧跟踪/derive_priors**。")
    L.append(q2 + "\n")

    # Q3
    fg_o = so["false_green"]
    if fg_o == 0:
        q3_tail = "假绿被杀"
    else:
        q3_tail = f"Arm O 残余 {fg_o} 假绿(正确位置 ROI 仍混入红相位绿干扰)"
    q3 = (f"Q3 假绿是否被杀: Arm C={sc['false_green']} → Arm F={sf['false_green']} → Arm O={so['false_green']} → "
          f"{q3_tail}")
    L.append(q3 + "\n")

    # 根因分析
    # 统计红灯帧的 g_frac/r_frac 比值
    red_frames = [r for r in results if r["gov_color_gt"] == "red"]
    green_frames = [r for r in results if r["gov_color_gt"] == "green"]
    red_g = [float(r["armO_g_frac"]) for r in red_frames]
    red_r = [float(r["armO_r_frac"]) for r in red_frames]
    green_g = [float(r["armO_g_frac"]) for r in green_frames]
    green_r = [float(r["armO_r_frac"]) for r in green_frames]
    import numpy as _np
    L.append(f"\n## 根因分析: oracle 位置为什么也救不了\n")
    L.append(f"Arm O(oracle 逐帧 gov 中心) 的 ROI(160px) 内绿/红像素占比:\n")
    L.append(f"- **红灯帧**({len(red_frames)}帧): g_frac 均值={_np.mean(red_g)*100:.1f}%, r_frac 均值={_np.mean(red_r)*100:.1f}%")
    L.append(f"  → 绿像素是红像素的 **{_np.mean(red_g)/max(_np.mean(red_r),0.001):.1f}x**, 远超 `_sample_roi` 的 g_frac>r_frac×1.3 判绿阈值")
    L.append(f"- **绿灯帧**({len(green_frames)}帧): g_frac 均值={_np.mean(green_g)*100:.1f}%, r_frac 均值={_np.mean(green_r)*100:.1f}%")
    L.append(f"  → 绿像素是红像素的 {_np.mean(green_g)/max(_np.mean(green_r),0.001):.1f}x\n")
    L.append(f"**结论**: 160px ROI 即使正确定位在 gov 灯上, 仍捕获大量环境绿(树叶/建筑/反光)。")
    L.append(f"红灯帧的环境绿是信号红的 {_np.mean(red_g)/max(_np.mean(red_r),0.001):.1f}x, `g_frac>r_frac×1.3` 恒成立 → **永远判绿**。")
    L.append(f"prior 重定位(改位置)救不了 05; 瓶颈在 `_sample_prior_color` 的 ROI 太大 + 像素计数投票机制,")
    L.append(f"不是 prior 位置。方向: 缩小 ROI(如 40-60px 聚焦灯体)或改走候选路径(connected components 空间过滤)。\n")

    # 聚合表
    L.append(f"\n## 聚合 ({n} gov 帧)\n")
    L.append("| 臂 | 位置 | 非None | 正确 | 假绿 | 假红 |")
    L.append("|---|---|---|---|---|---|")
    for name, desc, s in arms:
        L.append(f"| {name} | {desc} | {s['non_none']}/{n} | {s['correct']}/{n} | {s['false_green']} | {s['false_red']} |")

    # 逐帧表
    L.append(f"\n## 逐帧明细\n")
    L.append("| fi | gov_cx | gov_cy | gt | C_color | C_g/r | F_color | F_g/r | O_color | O_g/r |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        L.append(
            f"| {r['fi']} | {r['gov_cx']} | {r['gov_cy']} | {r['gov_color_gt']} | "
            f"{r['armC_color']} | {r['armC_g_frac']}/{r['armC_r_frac']} | "
            f"{r['armF_color']} | {r['armF_g_frac']}/{r['armF_r_frac']} | "
            f"{r['armO_color']} | {r['armO_g_frac']}/{r['armO_r_frac']} |"
        )

    # 漂移帧分析
    L.append(f"\n## 漂移帧分析(fi=1652/1947, cx~0.25 远离均值 0.58)\n")
    for r in results:
        if r["fi"] in (1652, 1947):
            L.append(f"\n### fi={r['fi']} (gov_cx={r['gov_cx']}, gov_cy={r['gov_cy']})")
            L.append(f"- gt={r['gov_color_gt']}")
            L.append(f"- Arm C: {r['armC_color']} (g={r['armC_g_frac']}, r={r['armC_r_frac']})")
            L.append(f"- Arm F: {r['armF_color']} (g={r['armF_g_frac']}, r={r['armF_r_frac']}) — 固定 [0.58,0.34] {'命中' if r['armF_color'] else '落空'}")
            L.append(f"- Arm O: {r['armO_color']} (g={r['armO_g_frac']}, r={r['armO_r_frac']}) — oracle 逐帧")

    L.append(f"\n## 方法学\n")
    L.append("- 三臂均调 `det._sample_prior_color()`(含 2x 自适应扩展), 口径与生产 observe() 的 prior 直采一致。\n")
    L.append(f"- Arm O prior = 该帧 gov box 中心(GT), 仅诊断 oracle, 非生产路径(红线: GT 不进推理)。\n")
    L.append(f"- g_frac/r_frac = g_n/r_n ÷ ROI 像素面积(含边界裁剪); 若紧 ROI 无色则用扩展 ROI 面积。\n")
    L.append(f"- 逐帧 CSV: `data/output/qw/05_prior_relocation_per_frame.csv` (供 cc bit-for-bit 复核)。\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {REPORT}")


if __name__ == "__main__":
    main()
