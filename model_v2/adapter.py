"""Pluggable adapters used to steer the frozen SAM2 Hiera features.

Two variants, selected by ``adapter_type``:

``mlp``
    The SAM2-UNet recipe: ``LN -> down -> GELU -> up -> GELU``, applied per
    token.  Cheapest, but purely channel-wise -- it cannot see a token's
    neighbours.

``conv``
    Same bottleneck with a depthwise 3x3 convolution inside it, so the adapter
    can model *local spatial patterns*.  This matters on NEU-DET: the frozen
    SAM2 features are trained for boundary-defined objects and transfer poorly
    to texture-defined defects (crazing, pitted_surface, rolled-in_scale),
    which are exactly the classes a purely point-wise adapter cannot help.

Both are zero-initialised at the output projection, so inserting an adapter is
an exact identity at step 0 and the "how many stages carry an adapter"
ablation stays a fair comparison.
"""

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Bottleneck adapter over channel-last tokens ``(B, H, W, C)``.

    Args:
        dim: token dimension of the stage the adapter sits in front of.
        ratio: bottleneck ratio; hidden width is ``max(dim // ratio, 8)``.
        scale: residual scaling factor of the adapter output.
        pre_norm: LayerNorm before the bottleneck (stabilises training when
            the trunk statistics are frozen).
        adapter_type: ``"mlp"`` (point-wise) or ``"conv"`` (adds a depthwise
            spatial mixer inside the bottleneck).
        kernel_size: depthwise kernel of the ``conv`` variant.
    """

    def __init__(self, dim: int, ratio: int = 4, scale: float = 1.0,
                 pre_norm: bool = True, adapter_type: str = "mlp",
                 kernel_size: int = 3):
        super().__init__()
        if adapter_type not in ("mlp", "conv"):
            raise ValueError(f"adapter_type must be 'mlp' or 'conv', got {adapter_type!r}")
        hidden = max(dim // ratio, 8)
        self.dim = dim
        self.hidden = hidden
        self.scale = scale
        self.adapter_type = adapter_type
        self.norm = nn.LayerNorm(dim, eps=1e-6) if pre_norm else nn.Identity()
        self.down = nn.Linear(dim, hidden)
        self.act1 = nn.GELU()
        self.dw = (nn.Conv2d(hidden, hidden, kernel_size, padding=kernel_size // 2,
                             groups=hidden) if adapter_type == "conv" else None)
        self.act_dw = nn.GELU() if adapter_type == "conv" else None
        self.up = nn.Linear(hidden, dim)
        self.act2 = nn.GELU()
        self._reset_parameters()

    def _reset_parameters(self):
        # near-identity at init so plugging an adapter in never hurts the
        # pretrained features before it has learned anything
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        if self.dw is not None:
            nn.init.trunc_normal_(self.dw.weight, std=0.02)
            nn.init.zeros_(self.dw.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.down(self.norm(x)))
        if self.dw is not None:
            # (B, H, W, hidden) -> depthwise spatial mixing -> back
            h = h.permute(0, 3, 1, 2)
            h = self.act_dw(self.dw(h))
            h = h.permute(0, 2, 3, 1)
        h = self.act2(self.up(h))
        return x + self.scale * h

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, hidden={self.hidden}, type={self.adapter_type}, "
                f"scale={self.scale}")
