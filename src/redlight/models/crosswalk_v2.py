"""L3 斑马线检测 —— v2 有状态四边形检测器 (train-free 探针, Plan v6 Phase 1).

设计(改定 2: 时间聚合替首帧检测):
  - v11 产出全宽水平带 [cy1,cy2] -> 永远出不了透视四边形(根因)。
  - v2 改为: 对每帧算「条纹响应图」(斑马条边界=垂直边缘 ∧ 亮 ∧ 道路ROI), 跨帧
    **running-max 时间聚合** -> 移动的车被抵消、静止的斑马条纹留下 -> 看穿「部分帧
    被遮挡」, 恢复更完整斑马线。
  - 由聚合图分行求 x 延展(5–95 百分位去离群, 剔除孤立车道线) -> 直接输出**透视梯形
    掩膜**(非全宽水平带)。

状态: accum(聚合响应图) + frame_count。检测器每视频在 cli.run 内新建 -> 无跨视频污染。
红线: 只吃帧, 绝不 import/read datasets/gt、GtCrosswalkDetector、per-video 硬编码多边形。
"""
import cv2
import numpy as np

from ..models.base_model import BaseModel, ModelInfo


class CrosswalkDetectorV2(BaseModel):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        self.verbose = verbose
        # 时间聚合状态(每视频新建 -> 无跨视频污染)
        self._accum = None
        self._frame_count = 0

    def load(self, weights_path=None):
        pass

    def get_info(self) -> ModelInfo:
        return ModelInfo(name="CrosswalkDetectorV2", version="cv-v2-temporal",
                         classes=["crosswalk"], input_size=(0, 0))

    def infer(self, frame: np.ndarray):
        return self.detect(frame)

    def detect(self, frame, vehicle_boxes=None):
        """检测斑马线, 返回二值掩膜(h,w) uint8。

        Args:
            frame: BGR 图像
            vehicle_boxes: 预留(本探针不使用车辆锚定; 后续 seg 阶段可接)
        """
        h, w = frame.shape[:2]
        if h < 120 or w < 160:
            return np.zeros((h, w), dtype=np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 斑马条边界 = 垂直边缘(条纹间明暗交界)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge = np.abs(sobelx)
        edge = cv2.GaussianBlur(edge, (5, 5), 0)

        # 亮像素(漆线比沥青/天空亮), 排除平滑亮区(天空/白车车身)
        mean_v = float(np.mean(gray))
        std_v = float(np.std(gray)) + 1e-6
        bright_thr = max(160.0, mean_v + 0.8 * std_v)
        bright = gray > bright_thr

        emax = float(edge.max())
        edge_resp = edge > (0.10 * emax) if emax > 0 else np.zeros_like(edge, dtype=bool)

        # 道路 ROI: 仅下半帧(斑马线在路面, 排除天空/建筑)
        roi = np.zeros((h, w), dtype=bool)
        roi[int(0.35 * h):, :] = True

        resp = (edge_resp & bright & roi).astype(np.float32)

        # ---- 时间聚合(running-max, 抗部分帧遮挡, 改定 2) ----
        if self._accum is None:
            self._accum = resp
        else:
            self._accum = np.maximum(self._accum, resp)
        self._frame_count += 1

        return self._fit_trapezoid(self._accum, h, w)

    def _fit_trapezoid(self, resp, h, w):
        """由聚合响应图分行求 x 延展(5–95 百分位去离群) -> 透视梯形掩膜。

        y-band: 行响应数高于峰值的一定比例(梯形两端变窄, 低阈值保全长)。
        注: 这是 Phase 1 train-free 探针, mask-IoU 预期低于 0.5 bar(车底遮挡看不见,
        改定 1); 真正解在 Phase 2 seg。
        """
        row_count = (resp > 0).sum(axis=1)
        maxc = int(row_count.max())
        if maxc <= 0:
            return np.zeros((h, w), dtype=np.uint8)

        # y-band: 行响应数高于峰值的一定比例(梯形两端变窄, 低阈值保全长)
        band_thr = 0.10 * maxc
        band_rows = np.where(row_count > band_thr)[0]
        if len(band_rows) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        cy1, cy2 = int(band_rows.min()), int(band_rows.max())

        mask = np.zeros((h, w), dtype=np.uint8)
        for y in band_rows:
            xs = np.where(resp[y] > 0)[0]
            if len(xs) < 5:
                continue
            # 5–95 百分位去离群(剔除孤立车道线/路边), 得该行的斑马线左右边界
            xmin = int(np.percentile(xs, 5))
            xmax = int(np.percentile(xs, 95))
            mask[y, xmin:xmax + 1] = 255
        return mask
