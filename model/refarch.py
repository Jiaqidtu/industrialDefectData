"""Bisection harness: ultralytics' YOLOv8 network inside OUR training loop.

ultralytics reaches 0.71 mAP from scratch on this data; our implementation
reaches 0.37 with a bit-identical assigner and decode.  The defect therefore
lives in the network (neck/head), the training loop, or the data pipeline.
This wrapper swaps in their network while keeping our loss, our dataloader and
our optimiser, which splits the remaining search space in half:

* reaches ~0.71  ->  the defect is in OUR neck/head
* stays at ~0.37 ->  the defect is in OUR loss / training loop / data
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .dataset import NEU_CLASSES
from .head import DFL, dist2bbox, make_anchors, non_max_suppression
from .loss import DetectionLoss


class RefHeadShim(nn.Module):
    """Presents ultralytics' Detect output through our head's interface."""

    def __init__(self, nc: int, reg_max: int, strides: List[int]):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.strides = list(strides)
        self.use_obj = False
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def flatten(self, raw: Dict):
        cls = raw["scores"].permute(0, 2, 1)            # (B, A, nc)
        reg = raw["boxes"]                              # (B, 4*reg_max, A)
        anchors, strides = make_anchors(raw["feats"], self.strides)
        return cls, reg, None, anchors, strides

    def decode(self, raw: Dict) -> torch.Tensor:
        cls, reg, _, anchors, strides = self.flatten(raw)
        dist = self.dfl(reg).permute(0, 2, 1) * strides
        return torch.cat([dist2bbox(dist, anchors.unsqueeze(0)), cls.sigmoid()], -1)


class RefArchModel(nn.Module):
    """ultralytics network + our DetectionLoss, exposing the YOLOSAM2 API."""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.cfg = cfg
        self.nc = int(cfg.get("nc", 6))
        self.names = list(cfg.get("names") or NEU_CLASSES)[: self.nc]
        self.imgsz = int(cfg.get("imgsz", 640))
        arch = cfg.get("arch", "yolov8n.yaml")

        try:
            from ultralytics.nn.tasks import DetectionModel
        except ImportError as e:
            raise SystemExit(f"--backbone ultra 需要 ultralytics: {e}")

        self.det = DetectionModel(cfg=arch, nc=self.nc, verbose=False)
        detect = self.det.model[-1]
        strides = [int(s) for s in detect.stride]
        self.head = RefHeadShim(self.nc, int(detect.reg_max), strides)
        self.criterion = DetectionLoss(self.head, nc=self.nc,
                                       **{k: v for k, v in cfg.get("loss", {}).items()})

    def _raw(self, x: torch.Tensor) -> Dict:
        out = self.det(x)
        return out if isinstance(out, dict) else out[1]

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        raw = self._raw(x)
        if self.training:
            if targets is None:
                return raw
            return self.criterion(raw, targets, self.head)
        return self.head.decode(raw)

    @torch.no_grad()
    def predict(self, x, conf_thres=0.25, iou_thres=0.65, max_det=300):
        was_training = self.training
        self.eval()
        out = non_max_suppression(self(x), conf_thres, iou_thres, max_det)
        if was_training:
            self.train()
        return out

    # ---- YOLOSAM2-compatible bookkeeping -------------------------------- #
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_counts(self) -> Dict:
        total = sum(p.numel() for p in self.parameters()) / 1e6
        return {"trunk_frozen_M": 0.0, "adapters_M": 0.0, "neck_M": 0.0,
                "head_M": 0.0, "total_M": round(total, 3),
                "trainable_M": round(total, 3), "adapter_stages": []}

    @torch.no_grad()
    def flops(self, imgsz=None):
        return 0.0

    @torch.no_grad()
    def benchmark_fps(self, imgsz=None, batch=1, warmup=5, iters=30):
        return 0.0

    def save(self, path: str, extra: Optional[Dict] = None):
        torch.save({"cfg": self.cfg, "state_dict": self.state_dict(),
                    "ref_arch": True, **(extra or {})}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu", strict: bool = True):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ck["cfg"])
        model.load_state_dict(ck["state_dict"], strict=strict)
        return model


def load_any(path: str, arch: str = "yolov8n.yaml", nc: int = 6,
             imgsz: int = 640, map_location: str = "cpu"):
    """Load one of the three checkpoint flavours we compare.

    Returns (model, needs_imagenet_norm).  ultralytics weights were trained on
    plain /255 inputs, so they must not get our ImageNet normalisation -- the
    flag is returned rather than assumed because getting it wrong is silent.
    """
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ck, dict) and ck.get("ref_arch"):
        return RefArchModel.load(path, map_location=map_location), True
    if isinstance(ck, dict) and "model" in ck and hasattr(ck["model"], "state_dict"):
        model = RefArchModel({"nc": nc, "imgsz": imgsz, "arch": arch})
        model.det.load_state_dict(
            {k: v.float() for k, v in ck["model"].state_dict().items()},
            strict=False)
        return model, False
    from .yolo_sam2 import YOLOSAM2
    return YOLOSAM2.load(path, map_location=map_location), True
