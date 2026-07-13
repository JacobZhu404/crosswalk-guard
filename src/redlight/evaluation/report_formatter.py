"""评测报告格式化器：薄层 CSV / 控制台输出，不引入新聚合逻辑。

输入均为已结构化的 metrics dict，与 evaluator.Evaluator 输出对齐。
"""

import csv
import os
from typing import Dict, List


def format_plate_per_video(result: dict) -> List[str]:
    """格式化单视频车牌评测控制台输出。

    Args:
        result: eval_plate.py 中 evaluate_video 返回的字典

    Returns:
        输出行列表（无换行符）
    """
    lines = []
    s = result["stats"]
    lines.append(f"  处理帧数: {s['processed_frames']}/{s['total_frames']}")
    lines.append(f"  识别到车牌帧数: {s['detected_plate_frames']}")
    lines.append(f"  识别到独立车牌: {s['unique_plates_detected']}")

    if result.get("matched"):
        lines.append("  ✅ 匹配结果:")
        for m in result["matched"]:
            flag = "精确" if m["edit_dist"] == 0 else f"近似(ED={m['edit_dist']})"
            lines.append(
                f"    {m['gt']} -> {m['detected']} [{flag}] "
                f"次数={m['count']} 平均conf={m['avg_conf']:.3f}"
            )

    if result.get("missed"):
        lines.append(f"  ❌ 未识别到: {result['missed']}")

    if result.get("false_positives"):
        lines.append(f"  ⚠️ 误检车牌: {result['false_positives']}")

    lines.append(f"  准确率: {s['accuracy']:.1%}")
    return lines


def format_plate_summary(total_stats: dict) -> List[str]:
    """格式化车牌评测合计控制台输出。

    Args:
        total_stats: 累计统计字典，键含 videos_tested, gt_plates,
                     exact_matches, near_matches, missed, false_positives

    Returns:
        输出行列表（无换行符）
    """
    lines = []
    sep = "=" * 80
    lines.append("")
    lines.append(sep)
    lines.append("综合评测结果")
    lines.append(sep)
    lines.append(f"测试视频数: {total_stats['videos_tested']}")
    lines.append(f"GT车牌总数: {total_stats['gt_plates']}")
    lines.append(f"精确匹配: {total_stats['exact_matches']}")
    lines.append(f"近似匹配(ED=1): {total_stats['near_matches']}")
    lines.append(f"未识别: {total_stats['missed']}")
    lines.append(f"误检车牌数: {total_stats['false_positives']}")
    gt = total_stats["gt_plates"]
    lines.append("")
    if gt > 0:
        acc = (total_stats["exact_matches"] + total_stats["near_matches"]) / gt
        lines.append(f"整体准确率: {acc:.1%}")
    else:
        lines.append("整体准确率: N/A")
    return lines


def write_plate_csv(results: List[dict], out_path: str) -> str:
    """写入车牌评测 CSV 报告。

    Args:
        results: evaluate_video 返回的字典列表
        out_path: 输出 CSV 路径

    Returns:
        实际写入的文件路径
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "video", "gt_plates", "detected_unique", "exact_matches",
            "near_matches", "missed", "false_positives", "accuracy",
        ])
        for r in results:
            s = r["stats"]
            w.writerow([
                r["video"],
                ",".join(r.get("gt_plates", [])),
                s["unique_plates_detected"],
                s["exact_matches"],
                s["near_matches"],
                ",".join(r.get("missed", [])),
                ",".join(r.get("false_positives", [])),
                f"{s['accuracy']:.2%}",
            ])
    return out_path
