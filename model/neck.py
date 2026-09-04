"""Necks that turn the Hiera C2..C5 pyramid into the P3/P4/P5 detection maps.

Three interchangeable variants (the "third level" ablation of the design doc):
``panet`` (default), ``bifpn`` (learned cross-scale weights) and ``fpn``
(top-down only).  All of them take the four Hiera stage outputs -- C2 (stride
4) is not detected on directly, it is injected into P3 so that hairline
defects (crazing / scratches) keep their high-frequency detail.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import C2f, Conv, SPPF


def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


class _NeckBase(nn.Module):
    """Shared lateral 1x1 projections C2..C5 -> common width + SPPF on C5."""

    def __init__(self, in_channels: Sequence[int], width: int = 256,
                 use_sppf: bool = True, use_c2: bool = True):
        super().__init__()
        assert len(in_channels) == 4, "expects [C2, C3, C4, C5]"
        self.width = width
        self.use_c2 = use_c2
        self.lateral = nn.ModuleList(Conv(c, width, 1, 1) for c in in_channels)
        self.sppf = SPPF(width, width) if use_sppf else nn.Identity()
        # C2 (stride 4) -> stride 8, folded into the P3 branch
        self.c2_down = Conv(width, width, 3, 2) if use_c2 else None
        self.c2_fuse = Conv(2 * width, width, 1, 1) if use_c2 else None
        self.out_channels = [width, width, width]
        self.out_strides = [8, 16, 32]

    def project(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        c2, c3, c4, c5 = (lat(f) for lat, f in zip(self.lateral, feats))
        c5 = self.sppf(c5)
        if self.use_c2:
            c3 = self.c2_fuse(torch.cat([c3, _upsample_to(self.c2_down(c2), c3)], 1))
        return [c3, c4, c5]


class PANetNeck(_NeckBase):
    """Top-down (semantics) then bottom-up (localisation) path aggregation."""

    def __init__(self, in_channels, width=256, depth=1, use_sppf=True, use_c2=True):
        super().__init__(in_channels, width, use_sppf, use_c2)
        self.td4 = C2f(2 * width, width, n=depth)
        self.td3 = C2f(2 * width, width, n=depth)
        self.down3 = Conv(width, width, 3, 2)
        self.bu4 = C2f(2 * width, width, n=depth)
        self.down4 = Conv(width, width, 3, 2)
        self.bu5 = C2f(2 * width, width, n=depth)

    def forward(self, feats):
        c3, c4, c5 = self.project(feats)
        t4 = self.td4(torch.cat([c4, _upsample_to(c5, c4)], 1))
        p3 = self.td3(torch.cat([c3, _upsample_to(t4, c3)], 1))
        p4 = self.bu4(torch.cat([t4, self.down3(p3)], 1))
        p5 = self.bu5(torch.cat([c5, self.down4(p4)], 1))
        return [p3, p4, p5]


class FPNNeck(_NeckBase):
    """Plain top-down FPN, the cheapest baseline of the neck ablation."""

    def __init__(self, in_channels, width=256, depth=1, use_sppf=True, use_c2=True):
        super().__init__(in_channels, width, use_sppf, use_c2)
        self.smooth4 = C2f(2 * width, width, n=depth)
        self.smooth3 = C2f(2 * width, width, n=depth)
        self.out5 = Conv(width, width, 3, 1)

    def forward(self, feats):
        c3, c4, c5 = self.project(feats)
        p4 = self.smooth4(torch.cat([c4, _upsample_to(c5, c4)], 1))
        p3 = self.smooth3(torch.cat([c3, _upsample_to(p4, c3)], 1))
        return [p3, p4, self.out5(c5)]


class _WeightedFuse(nn.Module):
    """Fast normalised fusion of BiFPN: sum(w_i * x_i) / (sum(w_i) + eps)."""

    def __init__(self, n_inputs: int, channels: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = eps
        self.conv = C2f(channels, channels, n=1)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = sum(wi * xi for wi, xi in zip(w, xs))
        return self.conv(out)


class BiFPNNeck(_NeckBase):
    """Single BiFPN stage with learnable cross-scale weights (repeatable)."""

    def __init__(self, in_channels, width=256, depth=1, use_sppf=True, use_c2=True,
                 repeats: int = 2):
        super().__init__(in_channels, width, use_sppf, use_c2)
        self.repeats = repeats
        self.td4 = nn.ModuleList(_WeightedFuse(2, width) for _ in range(repeats))
        self.td3 = nn.ModuleList(_WeightedFuse(2, width) for _ in range(repeats))
        self.bu4 = nn.ModuleList(_WeightedFuse(3, width) for _ in range(repeats))
        self.bu5 = nn.ModuleList(_WeightedFuse(2, width) for _ in range(repeats))
        self.down3 = nn.ModuleList(Conv(width, width, 3, 2) for _ in range(repeats))
        self.down4 = nn.ModuleList(Conv(width, width, 3, 2) for _ in range(repeats))

    def forward(self, feats):
        p3, p4, p5 = self.project(feats)
        for i in range(self.repeats):
            t4 = self.td4[i]([p4, _upsample_to(p5, p4)])
            o3 = self.td3[i]([p3, _upsample_to(t4, p3)])
            o4 = self.bu4[i]([p4, t4, self.down3[i](o3)])
            o5 = self.bu5[i]([p5, self.down4[i](o4)])
            p3, p4, p5 = o3, o4, o5
        return [p3, p4, p5]


NECKS = {"panet": PANetNeck, "bifpn": BiFPNNeck, "fpn": FPNNeck}


def build_neck(name: str, in_channels: Sequence[int], **kwargs) -> nn.Module:
    if name not in NECKS:
        raise ValueError(f"unknown neck {name!r}; choose from {sorted(NECKS)}")
    cls = NECKS[name]
    if name != "bifpn":
        kwargs.pop("repeats", None)
    return cls(in_channels, **kwargs)
