"""L5 任务编排: 车辆静止状态管理 (V2 滑动窗口速度)。

修复 v1 根因 (E7):
  v1 用"首尾质心距离 / 整窗时间"算速度, 窗口内任何早期位移都会把速度拉高,
  导致 speed_thres=15 对蠕行/抖动过严 -> 真停车也判不了静止 (违章01 全 stop=0)。
V2 改进:
  - 计算**相邻采样帧的瞬时速度序列**, 用"低速度占比 + 尾部连续低速度"双判据,
    对跟踪抖动鲁棒。
  - 提供三档灵敏度预设 (strict/balanced/loose), 由评测数据驱动选择最优。
"""
import numpy as np

from ..infrastructure.geometry import iou

SENSITIVITY_PRESETS = {
    # speed: 瞬时速度阈值(px/s, 回退位移法用); sustain: 需尾部连续静止帧数
    # duration: 三条件需持续帧数; overlap: 压线比例阈值
    # speed_window: 参与统计的近期采样帧数; stationary_ratio: 静止帧占比下限
    # stationary_iou: 相邻帧框 IoU 静止阈值(主判据, 抗YOLO框抖动; 越高越严)
    "strict":   {"speed": 15, "sustain": 8, "duration": 8, "overlap": 0.30, "box_overlap": 0.30, "speed_window": 8, "stationary_ratio": 1.0, "stationary_iou": 0.80},
    "balanced": {"speed": 30, "sustain": 5, "duration": 5, "overlap": 0.20, "box_overlap": 0.20, "speed_window": 6, "stationary_ratio": 0.7, "stationary_iou": 0.70},
    "loose":    {"speed": 50, "sustain": 3, "duration": 3, "overlap": 0.15, "box_overlap": 0.15, "speed_window": 4, "stationary_ratio": 0.5, "stationary_iou": 0.60},
    "very_loose": {"speed": 80, "sustain": 2, "duration": 2, "overlap": 0.10, "box_overlap": 0.10, "speed_window": 3, "stationary_ratio": 0.4, "stationary_iou": 0.50},
}


class TrackStateManagerV2:
    """基于 track id 维护每辆车位置历史, 用滑动窗口速度判定是否静止。"""

    def __init__(self, preset="balanced"):
        if preset not in SENSITIVITY_PRESETS:
            preset = "balanced"
        self.preset_name = preset
        p = SENSITIVITY_PRESETS[preset]
        self.speed_thres = p["speed"]
        self.sustain = p["sustain"]
        self.speed_window = p["speed_window"]
        self.stationary_ratio = p["stationary_ratio"]
        self.stationary_iou = p.get("stationary_iou", 0.70)
        self.states = {}

    def update(self, dets, timestamp):
        for st in self.states.values():
            st["active"] = False
        for d in dets:
            tid = d["id"]
            x1, y1, x2, y2 = d["xyxy"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            st = self.states.get(tid)
            if st is None:
                st = {"history": [], "stationary": False, "active": True,
                      "box": d["xyxy"], "cls": d.get("cls", ""), "conf": d.get("conf", 0.0)}
                self.states[tid] = st
            st["active"] = True
            st["box"] = d["xyxy"]
            st["cls"] = d.get("cls", "")
            st["conf"] = d.get("conf", 0.0)
            st["history"].append((timestamp, cx, cy, tuple(d["xyxy"])))
            keep = self.speed_window + 1
            if len(st["history"]) > keep:
                st["history"] = st["history"][-keep:]
            st["stationary"] = self._is_stationary(st)
        return self.states

    def _is_stationary(self, st):
        hist = st["history"]
        if len(hist) < 2:
            return False
        # 静止判定用相邻帧框 IoU(而非框中心位移): 车静止时即使 YOLO 框大小抖动
        # (违章02白车框宽±50%跳变), 位置仍大致重叠→IoU高; 位移法对框抖动敏感致真停车判非静止(实测仅3%)。
        # history 元素含 box(第4项)时用 IoU, 否则回退位移法(兼容)。
        if len(hist[-1]) >= 4:
            ious = []
            for i in range(1, len(hist)):
                b0 = hist[i - 1][3]
                b1 = hist[i][3]
                ious.append(iou(b0, b1))
            # IoU 高 = 位置稳 = 静止(阈值 stationary_iou)
            still = [v >= self.stationary_iou for v in ious]
            frac = sum(still) / len(still)
            trailing = 0
            for v in reversed(still):
                if v:
                    trailing += 1
                else:
                    break
            return (frac >= self.stationary_ratio) and (trailing >= self.sustain)
        # 回退: 框中心位移速度法
        speeds = []
        for i in range(1, len(hist)):
            t0, x0, y0 = hist[i - 1][:3]
            t1, x1_, y1_ = hist[i][:3]
            dt = max(t1 - t0, 1e-3)
            sp = float(np.hypot(x1_ - x0, y1_ - y0) / dt)
            speeds.append(sp)
        low = [s < self.speed_thres for s in speeds]
        frac = sum(low) / len(low)
        trailing = 0
        for v in reversed(low):
            if v:
                trailing += 1
            else:
                break
        return (frac >= self.stationary_ratio) and (trailing >= self.sustain)
