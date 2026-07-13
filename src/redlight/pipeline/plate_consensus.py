"""多帧车牌投票模块: 对同一车辆在不同帧的车牌识别结果做加权投票, 输出最可靠的车牌。

策略:
  - 按 track_id 累积识别结果
  - 加权投票: weight = frequency × avg_confidence
  - 去重: 编辑距离≤1的视为同一车牌, 合并置信度
  - 跨时间段跟踪: 保留全视频历史, 支持分段统计
  - 全局投票: 综合所有时间段的识别结果, 即使违章时刻未识别到车牌

用法:
    pc = PlateConsensus(keep_history=300)  # 保留300帧历史(约10秒)
    pc.update(track_id, plate_text, confidence, timestamp)
    
    # 获取最佳车牌(全局)
    best = pc.get_best(track_id)
    
    # 获取指定时间段内的车牌
    best_in_window = pc.get_best_in_window(track_id, start_ts, end_ts)
    
    # 获取跨时间段关联结果
    global_best = pc.get_global_best(track_id)
"""
import os
from collections import defaultdict

from ..evaluation.metrics import levenshtein


class PlateConsensus:
    def __init__(self, keep_history=300, dedup_dist=1):
        self.keep_history = keep_history
        self.dedup_dist = dedup_dist
        self.track_records = defaultdict(list)
        self.track_summary = defaultdict(dict)

    def update(self, track_id, plate_text, confidence, timestamp):
        if not plate_text:
            return
        self.track_records[track_id].append({
            "text": plate_text,
            "conf": confidence,
            "ts": timestamp,
        })
        if len(self.track_records[track_id]) > self.keep_history:
            self.track_records[track_id] = self.track_records[track_id][-self.keep_history:]
        self._recompute(track_id)

    def _recompute(self, track_id):
        records = self.track_records[track_id]
        if not records:
            self.track_summary[track_id] = {}
            return
        groups = {}
        for r in records:
            text = r["text"]
            matched = False
            for key in list(groups.keys()):
                if levenshtein(text, key) <= self.dedup_dist:
                    groups[key]["total_conf"] += r["conf"]
                    groups[key]["count"] += 1
                    groups[key]["latest_ts"] = r["ts"]
                    groups[key]["timestamps"].append(r["ts"])
                    matched = True
                    break
            if not matched:
                groups[text] = {
                    "total_conf": r["conf"],
                    "count": 1,
                    "latest_ts": r["ts"],
                    "timestamps": [r["ts"]],
                }
        for key in groups:
            groups[key]["avg_conf"] = groups[key]["total_conf"] / groups[key]["count"]
            groups[key]["weight"] = groups[key]["count"] * groups[key]["avg_conf"]
            groups[key]["timestamps"] = sorted(set(groups[key]["timestamps"]))
        self.track_summary[track_id] = groups

    def get_best(self, track_id):
        groups = self.track_summary.get(track_id, {})
        if not groups:
            return None
        best_key = max(groups.keys(), key=lambda k: groups[k]["weight"])
        best = groups[best_key]
        return {
            "text": best_key,
            "count": best["count"],
            "avg_conf": round(best["avg_conf"], 4),
            "weight": round(best["weight"], 4),
            "latest_ts": best["latest_ts"],
            "timestamps": best["timestamps"],
            "candidates": groups,
        }

    def get_best_in_window(self, track_id, start_ts, end_ts):
        records = self.track_records.get(track_id, [])
        window_records = [r for r in records if start_ts <= r["ts"] <= end_ts]
        if not window_records:
            return None
        
        groups = {}
        for r in window_records:
            text = r["text"]
            matched = False
            for key in list(groups.keys()):
                if levenshtein(text, key) <= self.dedup_dist:
                    groups[key]["total_conf"] += r["conf"]
                    groups[key]["count"] += 1
                    groups[key]["latest_ts"] = r["ts"]
                    matched = True
                    break
            if not matched:
                groups[text] = {
                    "total_conf": r["conf"],
                    "count": 1,
                    "latest_ts": r["ts"],
                }
        for key in groups:
            groups[key]["avg_conf"] = groups[key]["total_conf"] / groups[key]["count"]
            groups[key]["weight"] = groups[key]["count"] * groups[key]["avg_conf"]
        
        if not groups:
            return None
        best_key = max(groups.keys(), key=lambda k: groups[k]["weight"])
        best = groups[best_key]
        return {
            "text": best_key,
            "count": best["count"],
            "avg_conf": round(best["avg_conf"], 4),
            "weight": round(best["weight"], 4),
            "latest_ts": best["latest_ts"],
            "window": (start_ts, end_ts),
            "candidates": groups,
        }

    def get_global_best(self, track_id):
        records = self.track_records.get(track_id, [])
        if not records:
            return None
        
        groups = {}
        for r in records:
            text = r["text"]
            matched = False
            for key in list(groups.keys()):
                if levenshtein(text, key) <= self.dedup_dist:
                    groups[key]["total_conf"] += r["conf"]
                    groups[key]["count"] += 1
                    groups[key]["timestamps"].append(r["ts"])
                    matched = True
                    break
            if not matched:
                groups[text] = {
                    "total_conf": r["conf"],
                    "count": 1,
                    "timestamps": [r["ts"]],
                }
        
        for key in groups:
            groups[key]["avg_conf"] = groups[key]["total_conf"] / groups[key]["count"]
            groups[key]["weight"] = groups[key]["count"] * groups[key]["avg_conf"]
            groups[key]["timestamps"] = sorted(set(groups[key]["timestamps"]))
            groups[key]["time_spans"] = self._get_time_spans(groups[key]["timestamps"])
        
        if not groups:
            return None
        best_key = max(groups.keys(), key=lambda k: groups[k]["weight"])
        best = groups[best_key]
        return {
            "text": best_key,
            "count": best["count"],
            "avg_conf": round(best["avg_conf"], 4),
            "weight": round(best["weight"], 4),
            "timestamps": best["timestamps"],
            "time_spans": best["time_spans"],
            "candidates": groups,
        }

    def _get_time_spans(self, timestamps):
        if not timestamps:
            return []
        spans = []
        start = timestamps[0]
        prev = timestamps[0]
        for ts in timestamps[1:]:
            if ts - prev > 2.0:
                spans.append((round(start, 1), round(prev, 1)))
                start = ts
            prev = ts
        spans.append((round(start, 1), round(prev, 1)))
        return spans

    def get_time_segment_plates(self, track_id, segment_duration=10.0):
        records = self.track_records.get(track_id, [])
        if not records:
            return {}
        
        segments = {}
        for r in records:
            segment_start = int(r["ts"] // segment_duration) * segment_duration
            if segment_start not in segments:
                segments[segment_start] = []
            segments[segment_start].append(r)
        
        result = {}
        for seg_start, seg_records in segments.items():
            groups = {}
            for r in seg_records:
                text = r["text"]
                if text not in groups:
                    groups[text] = {"count": 0, "total_conf": 0, "timestamps": []}
                groups[text]["count"] += 1
                groups[text]["total_conf"] += r["conf"]
                groups[text]["timestamps"].append(r["ts"])
            
            for key in groups:
                groups[key]["avg_conf"] = round(groups[key]["total_conf"] / groups[key]["count"], 4)
            
            result[seg_start] = {
                "start_ts": seg_start,
                "end_ts": seg_start + segment_duration,
                "plates": groups,
            }
        return result

    def get_all(self):
        result = {}
        for tid in self.track_summary:
            best = self.get_best(tid)
            if best:
                result[tid] = best
        return result

    def get_global_all(self):
        result = {}
        for tid in self.track_records:
            best = self.get_global_best(tid)
            if best:
                result[tid] = best
        return result

    def clear(self):
        self.track_records.clear()
        self.track_summary.clear()
