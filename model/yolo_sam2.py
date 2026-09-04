"""YOLO-SAM2: frozen SAM2 (Hiera) backbone + pluggable adapters + YOLO neck/head.

Architecture (see ``design.md``)::

    image -> [Hiera stage1..4]           frozen, 4 pluggable adapters
          -> C2 C3 C4 C5                 strides 4/8/16/32
          -> PANet / BiFPN / FPN neck    trainable
          -> P3 P4 P5
          -> decoupled head (cls/reg/obj) + NMS
          -> 6 NEU-DET defect classes

Only the adapters, the neck and the head are trained; the Hiera trunk stays
frozen, which is what makes a 40M+ parameter foundation-model encoder usable
on the 1440-image NEU-DET training split.
"""

import copy
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import yaml

from .backbone_cnn import CSPBackbone
from .dataset import NEU_CLASSES
from .head import DecoupledHead, non_max_suppression
from .hiera import HieraAdapterBackbone
from .loss import DetectionLoss
from .neck import build_neck

DEFAULT_CFG: Dict = {
    "nc": 6,
    "names": list(NEU_CLASSES),
    "imgsz": 640,
    "backbone": {
        "type": "hiera",               # hiera | cnn (cnn = control backbone)
        "width": 0.5,                  # cnn only
        "depth": 0.34,                 # cnn only
        "variant": "tiny",             # tiny | small | base_plus | large
        "adapter_stages": [1, 2, 3, 4],  # <- the branch add/remove ablation knob
        "adapter_ratio": 4,
        "adapter_scale": 1.0,
        "adapter_type": "mlp",         # mlp | conv (conv = spatial-aware)
        "adapter_kernel": 3,
        "freeze_trunk": True,
        "train_norm": False,
        "unfreeze_stages": [],         # fully fine-tune these Hiera stages
        "checkpoint": None,            # path to an official SAM2 .pt
    },
    "neck": {"name": "panet", "width": 192, "depth": 1, "use_sppf": True,
             "use_c2": True, "repeats": 2},
    # use_obj off: measured harmful, see train.py --use-obj
    "head": {"reg_max": 16, "use_obj": False, "n_conv": 2},
    "loss": {"box_w": 7.5, "cls_w": 0.5, "dfl_w": 1.5, "obj_w": 1.0,
             "topk": 10, "iou_type": "ciou",
             "size_aware_assign": False,   # measured: -0.10 mAP on NEU-DET
             "size_tol": 1.5},
}


def _deep_update(base: Dict, other: Optional[Dict]) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None, **overrides) -> Dict:
    cfg = copy.deepcopy(DEFAULT_CFG)
    if path:
        with open(path) as fh:
            cfg = _deep_update(cfg, yaml.safe_load(fh) or {})
    return _deep_update(cfg, overrides)


class YOLOSAM2(nn.Module):
    """The hybrid detector."""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()
        self.cfg = cfg = _deep_update(DEFAULT_CFG, cfg or {})
        self.nc = int(cfg["nc"])
        self.names = list(cfg.get("names") or NEU_CLASSES)[: self.nc]
        self.imgsz = int(cfg.get("imgsz", 640))

        bb = dict(cfg["backbone"])
        bb_type = bb.pop("type", "hiera")
        ckpt = bb.pop("checkpoint", None)
        cnn_kwargs = {k: bb.pop(k) for k in ("width", "depth") if k in bb}
        coco = bb.pop("coco_pretrained", None)
        # pop before branching: these live in the shared backbone dict, and
        # HieraAdapterBackbone(**bb) would choke on them
        freq_kwargs = {k: bb.pop(k) for k in
                       ("patch", "n_radial", "n_orient", "freq_width",
                        "aux_cls_weight", "aux_seg_weight", "gate_mode",
                        "use_fft", "hann", "phase", "band_routing",
                        "log_radial", "fusion", "mps_bond",
                        "imgsz") if k in bb}
        # Anything still in `bb` at this point would be dropped on the floor
        # for the cnn/freqdual trunks, which do not take **bb.  That is how a
        # whole --mps-bond experiment ran three seeds without ever building
        # the module: the flag reached the config, the config reached here,
        # and nothing downstream ever asked for it.  Fail loudly instead.
        _HIERA_ONLY = {"variant", "freeze_trunk", "train_norm", "adapter_type",
                       "adapter_stages", "adapter_ratio", "adapter_scale",
                       "adapter_kernel", "unfreeze_stages"}
        unused = sorted(set(bb) - _HIERA_ONLY)
        if bb_type in ("cnn", "freqdual") and unused:
            raise ValueError(
                f"backbone kwargs {unused} are not consumed by the {bb_type} "
                f"trunk -- add them to freq_kwargs/cnn_kwargs rather than "
                f"letting them be silently ignored")
        if bb_type == "cnn":
            self.backbone = CSPBackbone(**cnn_kwargs)
            if coco:
                self.backbone.load_coco_checkpoint(coco)
        elif bb_type in ("freqdual", "freqhiera"):
            # spatial CSP trunk plus a spectral path, mixed by a class gate.
            # Kept behind the same interface so neck/head/loss are untouched.
            from model_newage.freqdual import FreqDualBackbone
            if coco and bb_type == "freqhiera":
                raise ValueError("--coco-pretrained applies to the CSP trunk; "
                                 "freqhiera uses Hiera, pass --sam2-ckpt")
            if bb_type == "freqhiera":
                self.backbone = FreqDualBackbone(nc=self.nc, trunk="hiera",
                                                 checkpoint=ckpt, **bb,
                                                 **freq_kwargs)
            else:
                self.backbone = FreqDualBackbone(nc=self.nc, trunk="cnn",
                                                 **cnn_kwargs, **freq_kwargs)
                # the spectral path has no COCO counterpart and stays random;
                # only the shared CSP trunk is initialised, which is what keeps
                # the freq-vs-cnn ablation comparable under pretraining.
                if coco:
                    self.backbone.spatial.load_coco_checkpoint(coco)
        elif bb_type == "hiera":
            self.backbone = HieraAdapterBackbone(**bb)
            if ckpt:
                if not os.path.exists(ckpt):
                    raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")
                self.backbone.load_sam2_checkpoint(ckpt)
        else:
            raise ValueError(f"unknown backbone type {bb_type!r}; use "
                             f"hiera, cnn, freqdual, freqhiera or ultra")

        self.neck = build_neck(cfg["neck"]["name"], self.backbone.out_channels,
                               **{k: v for k, v in cfg["neck"].items() if k != "name"})
        self.head = DecoupledHead(nc=self.nc, in_channels=self.neck.out_channels,
                                  strides=self.neck.out_strides, imgsz=self.imgsz,
                                  **cfg["head"])
        self.criterion = DetectionLoss(self.head, nc=self.nc, **cfg["loss"])

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Training: returns ``(loss, loss_items)``.  Inference: raw predictions."""
        feats = self.backbone(x)
        feats = self.neck(feats)
        if self.training:
            raw = self.head(feats)
            if targets is None:
                return raw
            return self.criterion(raw, targets, self.head)
        preds, _ = self.head(feats)
        return preds

    @torch.no_grad()
    def predict(self, x: torch.Tensor, conf_thres: float = 0.25,
                iou_thres: float = 0.65, max_det: int = 300) -> List[torch.Tensor]:
        """Convenience wrapper: forward + NMS -> list of (n, 6) detections."""
        was_training = self.training
        self.eval()
        preds = self(x)
        out = non_max_suppression(preds, conf_thres=conf_thres,
                                  iou_thres=iou_thres, max_det=max_det)
        if was_training:
            self.train()
        return out

    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_counts(self) -> Dict[str, float]:
        """Parameter budget in millions, split by component."""
        def s(module, only_trainable=False):
            return sum(p.numel() for p in module.parameters()
                       if p.requires_grad or not only_trainable) / 1e6

        adapters = sum(p.numel() for p in self.backbone.adapters.parameters()) / 1e6 \
            if hasattr(self.backbone, "adapters") else 0.0
        trunk = s(self.backbone) - adapters
        return {
            "trunk_frozen_M": round(trunk, 3),
            "adapters_M": round(adapters, 3),
            "neck_M": round(s(self.neck), 3),
            "head_M": round(s(self.head), 3),
            "total_M": round(s(self), 3),
            "trainable_M": round(
                sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6, 3),
            "adapter_stages": list(getattr(self.backbone, "adapter_stages", ())),
        }

    @torch.no_grad()
    def flops(self, imgsz: Optional[int] = None) -> float:
        """Approximate GFLOPs (conv + linear MACs x2) at ``imgsz`` resolution."""
        imgsz = imgsz or self.imgsz
        total = [0]
        hooks = []

        def conv_hook(m, i, o):
            out_elems = o.shape[-1] * o.shape[-2]
            total[0] += 2 * out_elems * m.in_channels // m.groups \
                * m.out_channels * m.kernel_size[0] * m.kernel_size[1]

        def lin_hook(m, i, o):
            total[0] += 2 * m.in_features * m.out_features * (o.numel() // o.shape[-1])

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                hooks.append(mod.register_forward_hook(conv_hook))
            elif isinstance(mod, nn.Linear):
                hooks.append(mod.register_forward_hook(lin_hook))

        mode = self.training
        self.eval()
        device = next(self.parameters()).device
        self(torch.zeros(1, 3, imgsz, imgsz, device=device))
        self.train(mode)
        for h in hooks:
            h.remove()
        return round(total[0] / 1e9, 3)

    @torch.no_grad()
    def benchmark_fps(self, imgsz: Optional[int] = None, batch: int = 1,
                      warmup: int = 5, iters: int = 30) -> float:
        """Images per second of a pure forward pass (no NMS)."""
        import time
        imgsz = imgsz or self.imgsz
        device = next(self.parameters()).device
        x = torch.zeros(batch, 3, imgsz, imgsz, device=device)
        mode = self.training
        self.eval()
        for _ in range(warmup):
            self(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            self(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.train(mode)
        return round(batch * iters / dt, 2)

    # ------------------------------------------------------------------ #
    def save(self, path: str, extra: Optional[Dict] = None):
        """Save the trainable part (adapters + neck + head) plus the cfg.

        The frozen trunk is normally *not* stored -- it is rebuilt from the
        SAM2 checkpoint named in the config, which keeps an ablation sweep of
        dozens of runs down to a few MB each.  When no SAM2 checkpoint was
        given the trunk is randomly initialised and therefore not reproducible,
        so in that case it is saved as well.
        """
        keep = ("backbone.adapters.", "neck.", "head.")
        if (self.cfg["backbone"].get("type", "hiera") in ("cnn", "freqdual",
                                                          "freqhiera")
                or not self.cfg["backbone"].get("checkpoint")):
            keep = ("backbone.", "neck.", "head.")
        # anything that received gradients must be stored too, otherwise a
        # partially unfrozen trunk (--unfreeze-stages / --train-norm) would be
        # silently restored from the pretrained checkpoint instead
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        state = {k: v for k, v in self.state_dict().items()
                 if k.startswith(keep) or k in trainable}
        torch.save({"cfg": self.cfg, "state_dict": state, **(extra or {})}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu", strict: bool = False) -> "YOLOSAM2":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ck["cfg"])
        report = model.load_state_dict(ck["state_dict"], strict=strict)
        missing = [k for k in report.missing_keys
                   if k.startswith(("neck.", "head.", "backbone.adapters."))]
        if missing:
            print(f"[load] warning: {len(missing)} trainable tensors missing")
        return model


def build_model(cfg_path: Optional[str] = None, **overrides):
    cfg = load_config(cfg_path, **overrides)
    if cfg["backbone"].get("type") == "ultra":
        from .refarch import RefArchModel
        return RefArchModel(cfg)
    return YOLOSAM2(cfg)
