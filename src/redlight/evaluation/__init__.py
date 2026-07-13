"""redlight.evaluation layer"""

from .evaluator import Evaluator
from .frame_dataset import FrameDataset
from .gallery_builder import BaseGalleryBuilder
from .gt_lookup import (
    expand_light_evidence,
    load_labeled_gt,
    load_light_segments,
    load_light_state_csv,
    load_plate_gt,
    state_at,
)
from .video_sampler import VideoSampler

# cv2 依赖的 gallery builder 不 eager import，避免无 cv2 环境报错
# 使用时直接 from redlight.evaluation.light_gallery import LightGalleryBuilder

__all__ = [
    "Evaluator",
    "FrameDataset",
    "BaseGalleryBuilder",
    "VideoSampler",
    "load_plate_gt",
    "load_light_segments",
    "load_light_state_csv",
    "load_labeled_gt",
    "state_at",
    "expand_light_evidence",
]
