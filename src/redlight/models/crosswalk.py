"""L3 斑马线检测 —— 经典 CV 兜底 (无需权重)。

版本演进:
  v7: 灰度阈值 → 误检草地(E15实证)
  v8: HSV白色+植被排除 → 整路被当候选→超面积漏检
  v9: HSV碎片合并 → 白车身/反光碎片段过多→全帧掩膜(更差)
  v10 (当前默认): 垂直梯度密度法 —— 利用斑马线"周期性条纹"的几何特征,
                    不依赖绝对亮度, 对白车/草地/路面均鲁棒。
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
        return ModelInfo(name="CrosswalkDetector", version="cv-v10",
                         classes=["crosswalk"], input_size=(0, 0))

    def infer(self, frame: np.ndarray):
        return self.detect(frame)

    def detect(self, frame):
        return self._cv_v10(frame)

    # ------------------------------------------------------------------ v10 (当前默认)
    def _cv_v10(self, frame):
        """v10: 垂直梯度密度法检测斑马线条纹 (2026-07-11)。

        原理:
          斑马线的本质特征是"横向周期性明暗交替条纹", 在垂直方向产生密集边缘.
          一般物体(车/草地/路面)要么颜色均匀(低梯度), 要么纹理随机(无规律).
          算法:
            1. ROI 取中下部 (y > h*0.45)
            2. 计算垂直 Sobel 梯度 → 强梯度=条纹边缘候选
            3. 沿 x 轴求和得到 1D "条纹密度剖面"(每行的梯度强度之和)
            4. 平滑 + 阈值 → 找出高密度行区间(斑马线位置)
            5. 可选: HSV 白色掩膜交叉验证(排除非白色区域)
            6. 输出矩形掩膜 + 轻量膨胀
        """
        h, w = frame.shape[:2]
        y0 = int(h * 0.50)
        roi = frame[y0:, :]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 40 or roi_w < 80:
            return np.zeros((h, w), dtype=np.uint8)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 垂直 Sobel 梯度: 条纹的水平边缘在垂直方向变化剧烈
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.abs(sobely)

        # 沿 x 轴求和: 得到每行的"条纹边缘密度"
        row_density = np.mean(grad_mag, axis=1)

        # 平滑去噪
        kernel_size = min(15, roi_h // 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(row_density, kernel, mode='same')

        # 自适应阈值: 密度 > mean + k*std 的区域为候选
        mean_val = np.mean(smoothed)
        std_val = np.std(smoothed)
        thresh = mean_val + 1.5 * std_val
        candidate_rows = smoothed > thresh

        # 找连续的高密度行区间
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
            return np.zeros((h, w), dtype=np.uint8)

        # 选最长/最强的高密度区间作为主候选
        best_run = max(runs, key=lambda r: r[1] - r[0])
        run_len = best_run[1] - best_run[0]

        # 斑马线至少应有一定高度(对应几条条纹宽度)
        min_crosswalk_height = int(roi_h * 0.05)
        if run_len < min_crosswalk_height:
            return np.zeros((h, w), dtype=np.uint8)

        # 交叉验证: 该区间的平均亮度应该偏亮(白色漆画)
        run_slice = roi[best_run[0]:best_run[1], :]
        run_gray = cv2.cvtColor(run_slice, cv2.COLOR_BGR2GRAY) if len(run_slice.shape) == 3 else run_slice
        avg_brightness = np.mean(run_gray)
        if avg_brightness < 100:   # 太暗 → 可能是阴影中的非斑马线区域
            return np.zeros((h, w), dtype=np.uint8)

        # 构建掩膜
        mask = np.zeros((h, w), dtype=np.uint8)
        cy1 = y0 + best_run[0]
        cy2 = y0 + best_run[1]
        cv2.rectangle(mask, (0, cy1), (w, cy2), 255, cv2.FILLED)

        # 轻量膨胀覆盖条纹间隙
        k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 8))
        mask = cv2.dilate(mask, k_d, iterations=1)

        return mask

    # ------------------------------------------------------------------ v7 (保留对比)
    def _cv_v7(self, frame):
        """v7: 灰度阈值 (已证实误检草地/E15)."""
        h, w = frame.shape[:2]
        y0 = int(h * 0.45)
        roi = frame[y0:, :]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 40 or roi_w < 80:
            return np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_h, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((h, w), dtype=np.uint8)
        valid_pts = []
        max_area = roi_w * roi_h * 0.35
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cv_min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            ar = bw / float(bh) if bh > 0 else 0
            if ar < 2.0:
                continue
            if bw < 40 or bh < 5:
                continue
            if y < int(roi_h * 0.30) and ar < 5.0:
                continue
            valid_pts.append(c + np.array([0, y0]))
        if valid_pts:
            cv2.drawContours(mask, valid_pts, -1, 255, cv2.FILLED)
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 10))
            mask = cv2.dilate(mask, k_d, iterations=1)
        return mask
