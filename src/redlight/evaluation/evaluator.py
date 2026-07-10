"""L4c 评测聚合器: 把各类指标汇总成一份报告。

输入均为已结构化的 预测 / 真值, 不依赖任何模型推理。
"""
from .metrics import (
    detection_metrics, plate_accuracy, char_accuracy, province_accuracy,
    edit_similarity, event_metrics, compute_map,
)


class Evaluator:
    def __init__(self, iou_thr=0.5, tiou_thr=0.5):
        self.iou_thr = iou_thr
        self.tiou_thr = tiou_thr

    def evaluate_detection(self, pred_boxes, gt_boxes):
        return detection_metrics(pred_boxes, gt_boxes, self.iou_thr)

    def evaluate_plates(self, pred_texts, gt_texts):
        assert len(pred_texts) == len(gt_texts)
        n = len(gt_texts)
        if n == 0:
            return {"plate_accuracy": 0.0, "char_accuracy": 0.0,
                    "province_accuracy": 0.0, "mean_edit_similarity": 0.0, "n": 0}
        return {
            "plate_accuracy": plate_accuracy(pred_texts, gt_texts),
            "char_accuracy": sum(char_accuracy(p, g) for p, g in zip(pred_texts, gt_texts)) / n,
            "province_accuracy": province_accuracy(pred_texts, gt_texts),
            "mean_edit_similarity": sum(edit_similarity(p, g) for p, g in zip(pred_texts, gt_texts)) / n,
            "n": n,
        }

    def evaluate_events(self, pred_events, gt_events):
        return event_metrics(pred_events, gt_events, self.tiou_thr)

    def full_report(self, pred_boxes, gt_boxes, pred_texts, gt_texts,
                    pred_events, gt_events):
        return {
            "detection": self.evaluate_detection(pred_boxes, gt_boxes),
            "plates": self.evaluate_plates(pred_texts, gt_texts),
            "events": self.evaluate_events(pred_events, gt_events),
        }
