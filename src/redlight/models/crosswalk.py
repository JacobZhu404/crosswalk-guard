"""L3 斑马线检测 —— 经典 CV 兜底 (无需权重)。

版本演进:
  v7: 灰度阈值 → 误检草地(E15实证)
  v8: HSV白色+植被排除 → 整路被当候选→超面积漏检
  v9: HSV碎片合并 → 白车身/反光碎片段过多→全帧掩膜(更差)
  v10: 垂直梯度密度法(y0=h*0.50 固定ROI) → 01对齐但02/03掩膜错位(E17实证)
  v11 (当前默认): 多位置扫描 + 可选车辆锚定 — 解决 E17 泛化失败。
"""
import cv2
import numpy as np

from ..models.base_model import BaseModel, ModelInfo


class CrosswalkDetector(BaseModel):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        self.method = getattr(cfg.crosswalk, "method", "auto")
        self.cv_min_area = getattr(cfg.crosswalk, "cv_min_area", 3000)
        if self.method not in ("auto", "cv", "segment"):
            self.method = "cv"
        self._loaded = True
        self._vb = verbose

    def load(self, weights_path=None):
        self._loaded = True

    def get_info(self) -> ModelInfo:
        return ModelInfo(name="CrosswalkDetector", version="cv-v11",
                         classes=["crosswalk"], input_size=(0, 0))

    def infer(self, frame: np.ndarray):
        return self.detect(frame)

    def detect(self, frame, vehicle_boxes=None):
        """检测斑马线, 返回二值掩膜(h,w) uint8。

        Args:
            frame: BGR 图像
            vehicle_boxes: 可选, 车辆框列表 [x1,y1,x2,y2], 用于锚定搜索区域.
                          当提供时, 靠近静止车的条带会获得加分, 提高命中率。
        """
        return self._cv_v11(frame, vehicle_boxes)

    # ====================================================================== v11 (当前默认)
    def _cv_v11(self, frame, vehicle_boxes=None):
        """v11: 多位置垂直条带扫描 + 车辆锚定搜索 (修复 E17 泛化失败).

        核心改进 (相对 v10):
          - 不再硬编码 ROI=y0=h*0.50, 改为在全帧高度上扫描多个重叠条带
          - 每个条带独立做 v10 的梯度密度剖面分析
          - 综合评分: 条纹长度 × 梯度强度 × 亮度bonus
          - 当提供 vehicle_boxes 时, 靠近车辆中心 Y 坐标的条带获得加权提升
          - 选全局最高分条带作为最终斑马线掩膜

        性能: 全帧 Sobel 只算一次; 条带间共享梯度数据.
              典型 720p 帧 ~7 个条带, CPU 上比 v10 慢约 3-5x (可接受).
        """
        h, w = frame.shape[:2]
        if h < 120 or w < 160:
            return np.zeros((h, w), dtype=np.uint8)

        # ---- 全帧预处理 (只做一次) ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.abs(sobely)

        # ---- 扫描参数 ----
        BAND_FRAC = 0.30      # 每个条带占帧高的 30%
        STEP_FRAC = 0.12       # 步进 = 帧高 12% (条带间 60% 重叠)
        MIN_RUN_FRAC = 0.06     # 最小条纹运行长度 (相对于条带高度)
        band_h = max(40, int(h * BAND_FRAC))
        step = max(20, int(h * STEP_FRAC))
        min_run_px = max(5, int(band_h * MIN_RUN_FRAC))

        candidates = []  # (score, abs_y1, abs_y2, run_len, band_y0)

        for y0 in range(0, h - band_h + 1, step):
            y1 = min(y0 + band_h, h)
            actual_h = y1 - y0

            # 该条带的梯度密度剖面
            row_density = np.mean(grad_mag[y0:y1, :], axis=1)

            # 平滑
            ks = min(15, actual_h // 4)
            if ks % 2 == 0:
                ks += 1
            kernel = np.ones(ks) / ks
            smoothed = np.convolve(row_density, kernel, mode='same')

            # 自适应阈值
            mean_val = np.mean(smoothed)
            std_val = np.std(smoothed)
            if std_val < 1e-6:
                continue
            thresh = mean_val + 1.5 * std_val
            candidate_rows = smoothed > thresh

            # 找连续高密度行区间
            runs = []
            in_run = False
            start = 0
            for i, val in enumerate(candidate_rows):
                if val and not in_run:
                    start = i
                    in_run = True
                elif not val and in_run:
                    runs.append((start, i))
                    in_run = False
            if in_run:
                runs.append((start, len(candidate_rows)))

            if not runs:
                continue

            # 选该条带内最长/最强的运行
            best_run = max(runs, key=lambda r: (r[1] - r[0]))
            run_len = best_run[1] - best_run[0]

            if run_len < min_run_px:
                continue

            # 亮度交叉验证
            run_slice = frame[y0 + best_run[0]:y0 + best_run[1], :]
            run_gray = cv2.cvtColor(run_slice, cv2.COLOR_BGR2GRAY) \
                if len(run_slice.shape) == 3 else run_slice
            avg_brightness = np.mean(run_gray)
            if avg_brightness < 100:
                continue

            # 综合评分
            run_mean_grad = np.mean(smoothed[best_run[0]:best_run[1]])
            score = float(run_len) * (run_mean_grad - mean_val) / std_val + avg_brightness / 255.0

            # 车辆锚定加分: 斑马线通常在静止车附近
            if vehicle_boxes:
                cy_center_abs = y0 + (best_run[0] + best_run[1]) / 2.0
                for vb in vehicle_boxes:
                    if len(vb) >= 4:
                        veh_cy = (vb[1] + vb[3]) / 2.0
                        dist = abs(cy_center_abs - veh_cy) / float(h)
                        if dist < 0.25:
                            score *= (1.5 + 2.0 * (0.25 - dist))  # 近则加更多分

            abs_y1 = y0 + best_run[0]
            abs_y2 = y0 + best_run[1]
            candidates.append((score, abs_y1, abs_y2, run_len))

        if not candidates:
            return np.zeros((h, w), dtype=np.uint8)

        # 取最高分候选
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0]
        _, cy1, cy2, rl = best

        # 构建掩膜
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (0, cy1), (w, cy2), 255, cv2.FILLED)

        # 轻量膨胀覆盖条纹间隙
        k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 8))
        mask = cv2.dilate(mask, k_d, iterations=1)

        return mask
