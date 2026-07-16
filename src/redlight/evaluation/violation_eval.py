"""端到端违章事件评测 (eval-e2e): 把整条流水线产出的违章事件与 events.csv GT 对比。

区别于:
- eval-b (temporal_fusion): 只评②层灯态段准确率。
- eval-plate: 只评车牌 OCR。
本模块评**最终违章结论**(灯态×静止×压线×判定的综合结果), 是三项能力的乘积落地。

主指标(用户选定): **事件级重叠匹配** — 预测 confirmed 事件与 GT is_violation=1 段
按时间重叠(≥min_overlap_s)贪心 1:1 配对 -> TP/FP/FN -> 事件级 P/R/F1。
辅助诊断: **覆盖率** — 每个 GT 违章段被预测事件覆盖的时长占比(暴露"抓到但只抓一小段"的欠检)。

纯函数(便于单测); GT 解析读 datasets/gt/events.csv。
"""

import csv
import os


def load_violation_gt(events_csv):
    """从 events.csv 加载**违章段** GT (仅 is_violation==1 的段)。

    返回 {video: [{"start_s","end_s","plates":[...],"note"}, ...]} (按 start_s 升序)。
    plates 取 violating_plates 分号切分, 过滤空/?/无牌。
    """
    gt = {}
    if not os.path.exists(events_csv):
        return gt
    with open(events_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            video = (row.get("video") or "").strip()
            if not video:
                continue
            if (row.get("is_violation") or "").strip() != "1":
                continue
            try:
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
            except (KeyError, ValueError):
                continue
            plates = [t.strip() for t in (row.get("violating_plates") or "").split(";")
                      if t.strip() and t.strip() not in ("?", "无牌")]
            gt.setdefault(video, []).append({
                "start_s": start_s, "end_s": end_s,
                "plates": plates, "note": (row.get("note") or "").strip(),
            })
    for v in gt:
        gt[v].sort(key=lambda x: x["start_s"])
    return gt


def overlap_seconds(a, b):
    """两区间 [start,end] 的重叠秒数(无交=0)。a,b 均为 (start,end)。"""
    s, e = max(a[0], b[0]), min(a[1], b[1])
    return max(0.0, e - s)


def load_video_metadata(videos_csv):
    """读 datasets/gt/videos.csv -> {video: has_violation(bool)}。

    负例判定以 `has_violation==0` 为准, 调用方据此决定是否把该视频的 confirmed 全计真误报。
    不硬编码视频名。
    """
    meta = {}
    if not os.path.exists(videos_csv):
        return meta
    with open(videos_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = (row.get("video") or "").strip()
            if not v:
                continue
            hv = (row.get("has_violation") or "").strip()
            meta[v] = (hv == "1")
    return meta


def classify_false_positives(match_result, pred_events, gt_violations, is_negative, min_overlap_s=0.5):
    """把 match_violation_events 的 FP 事件(未 1:1 匹配)分类为 真误报 / 碎片。

    判据(已与 cc 裁定):
      - is_negative=True                 -> neg_true_fp  (负例视频任何 confirmed = 真误报)
      - 正例 & 与某 GT 窗重叠 >= min_overlap_s -> fragment (属同一 GT 窗但未被 1:1 认领, 同一窗被切片)
      - 正例 & 否则(含零重叠)           -> oow_true_fp (窗外真误报, precision 真敌人)

    返回 dict:
      neg_true_fp / oow_true_fp / fragment : 未匹配 pred 事件索引列表
      detail : 逐条 [{idx, category, span, overlap_gt_window_s, gt_window, light_state}]
      neg_count / oow_count / fragment_count : 计数(供 aggregate 汇总)
    """
    fp_idx = list(match_result.get("fp_events", []))
    neg, oow, frag = [], [], []
    detail = []
    for i in fp_idx:
        pe = pred_events[i]
        pspan = (pe["start_ts"], pe["end_ts"])
        best_ov, best_ge = 0.0, None
        for ge in gt_violations:
            ov = overlap_seconds(pspan, (ge["start_s"], ge["end_s"]))
            if ov > best_ov:
                best_ov, best_ge = ov, ge
        if is_negative:
            cat = "neg_true_fp"
        elif best_ov >= min_overlap_s:
            cat = "fragment"
        else:
            cat = "oow_true_fp"
        if cat == "neg_true_fp":
            neg.append(i)
        elif cat == "oow_true_fp":
            oow.append(i)
        else:
            frag.append(i)
        gw = None
        if best_ge is not None:
            gw = f"[{best_ge['start_s']:.0f}-{best_ge['end_s']:.0f}]"
        detail.append({
            "idx": i, "category": cat,
            "span": f"[{pe['start_ts']:.1f}-{pe['end_ts']:.1f}]",
            "overlap_gt_window_s": round(best_ov, 2),
            "gt_window": gw,
            "light_state": pe.get("light_state", ""),
        })
    return {
        "neg_true_fp": neg, "oow_true_fp": oow, "fragment": frag, "detail": detail,
        "neg_count": len(neg), "oow_count": len(oow), "fragment_count": len(frag),
    }


def _union_length(intervals):
    """一组 [start,end] 区间的并集总长度。"""
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def plate_match(pred_plate, gt_plates):
    """预测车牌是否精确命中 GT 违章车牌之一(空/无 GT 车牌 -> False)。"""
    if not pred_plate or not gt_plates:
        return False
    return pred_plate.strip() in set(gt_plates)


def match_violation_events(pred_events, gt_violations, min_overlap_s=0.5):
    """事件级重叠匹配。

    pred_events: [{"start_ts","end_ts","plate"?, ...}] — 调用方应只传 confirmed 事件。
    gt_violations: [{"start_s","end_s","plates":[...]}] — is_violation==1 段。
    min_overlap_s: 判为一对 TP 所需的最小时间重叠(秒), 滤掉相切/1帧擦碰。

    返回 dict:
      tp/fp/fn: int
      precision/recall/f1: float
      matches: [{"pred_idx","gt_idx","overlap_s","plate_hit"}] (贪心 1:1)
      fp_events: [pred_idx,...]  fn_gts: [gt_idx,...]
      gt_coverage: [每个 GT 违章段被**所有**预测事件覆盖的时长占比] (诊断欠检; 独立于1:1配对)
      mean_coverage: 命中(被覆盖>0)的 GT 段的平均覆盖率
      plate_hits/plate_total: TP 事件里车牌命中数 / 有GT车牌的TP数(次级指标)
    """
    preds = list(pred_events)
    gts = list(gt_violations)

    # --- 主指标: 贪心 1:1 重叠匹配 ---
    cand = []
    for pi, pe in enumerate(preds):
        pspan = (pe["start_ts"], pe["end_ts"])
        for gi, ge in enumerate(gts):
            ov = overlap_seconds(pspan, (ge["start_s"], ge["end_s"]))
            if ov >= min_overlap_s:
                cand.append((ov, pi, gi))
    cand.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for ov, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        ph = plate_match(preds[pi].get("plate", ""), gts[gi]["plates"])
        matches.append({"pred_idx": pi, "gt_idx": gi, "overlap_s": round(ov, 2),
                        "plate_hit": ph})
    tp = len(matches)
    fp = len(preds) - len(used_p)
    fn = len(gts) - len(used_g)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # --- 诊断: 每个 GT 违章段被所有预测事件覆盖的比例(union) ---
    gt_coverage = []
    for ge in gts:
        gdur = ge["end_s"] - ge["start_s"]
        overlaps = []
        for pe in preds:
            s, e = max(pe["start_ts"], ge["start_s"]), min(pe["end_ts"], ge["end_s"])
            if e > s:
                overlaps.append([s, e])
        cov = (_union_length(overlaps) / gdur) if gdur > 0 else 0.0
        gt_coverage.append(round(cov, 3))
    covered = [c for c in gt_coverage if c > 0]
    mean_coverage = round(sum(covered) / len(covered), 3) if covered else 0.0

    # --- 次级: 车牌命中(仅统计有 GT 车牌的 TP) ---
    plate_total = sum(1 for m in matches if gts[m["gt_idx"]]["plates"])
    plate_hits = sum(1 for m in matches if m["plate_hit"])

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "matches": matches,
        "fp_events": [pi for pi in range(len(preds)) if pi not in used_p],
        "fn_gts": [gi for gi in range(len(gts)) if gi not in used_g],
        "gt_coverage": gt_coverage, "mean_coverage": mean_coverage,
        "plate_hits": plate_hits, "plate_total": plate_total,
    }


def aggregate(per_video_results):
    """把多视频结果聚合为总体事件级 P/R/F1 + 覆盖/车牌汇总。

    per_video_results: [match_violation_events 的返回 dict, ...]
    """
    tp = sum(r["tp"] for r in per_video_results)
    fp = sum(r["fp"] for r in per_video_results)
    fn = sum(r["fn"] for r in per_video_results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    covered = [c for r in per_video_results for c in r["gt_coverage"] if c > 0]
    mean_coverage = round(sum(covered) / len(covered), 3) if covered else 0.0
    plate_hits = sum(r["plate_hits"] for r in per_video_results)
    plate_total = sum(r["plate_total"] for r in per_video_results)
    # 拆解汇总(可选键: 由调用方在逐视频结果上挂 neg_count/oow_count/fragment_count)
    true_fp = sum(r.get("neg_count", 0) + r.get("oow_count", 0) for r in per_video_results)
    fragment = sum(r.get("fragment_count", 0) for r in per_video_results)
    # 诊断: 仅看真误报的精度(不含碎片) —— "多少次冤枉好人"
    p_only_true_fp = round(tp / (tp + true_fp), 3) if (tp + true_fp) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "mean_coverage": mean_coverage,
        "plate_hits": plate_hits, "plate_total": plate_total,
        "true_fp_total": true_fp, "fragment_total": fragment,
        "p_only_true_fp": p_only_true_fp,
    }
