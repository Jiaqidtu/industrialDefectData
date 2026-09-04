"""YOLO + SAM2(Hiera) hybrid defect detector for NEU-DET.

See ``README.md`` next to this file for the architecture and the ablation
protocol implemented here.
"""

from .adapter import Adapter
from .dataset import NEU_CLASSES, NEUDetDataset, build_dataloader
from .head import DecoupledHead, non_max_suppression
from .hiera import HIERA_PRESETS, HieraAdapterBackbone
from .loss import DetectionLoss, TaskAlignedAssigner
from .metrics import DetMetrics
from .neck import BiFPNNeck, FPNNeck, PANetNeck, build_neck
from .yolo_sam2 import DEFAULT_CFG, YOLOSAM2, build_model, load_config

__all__ = [
    "Adapter", "NEU_CLASSES", "NEUDetDataset", "build_dataloader",
    "DecoupledHead", "non_max_suppression", "HIERA_PRESETS",
    "HieraAdapterBackbone", "DetectionLoss", "TaskAlignedAssigner", "DetMetrics",
    "PANetNeck", "BiFPNNeck", "FPNNeck", "build_neck", "YOLOSAM2", "build_model",
    "load_config", "DEFAULT_CFG",
]
__version__ = "0.1.0"
