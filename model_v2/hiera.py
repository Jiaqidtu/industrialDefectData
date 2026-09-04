"""SAM2 Hiera trunk with per-stage pluggable adapters.

Self-contained re-implementation of ``sam2.modeling.backbones.hieradet.Hiera``
(Meta Platforms, Apache-2.0) so that this package depends only on ``torch``.
Parameter names are kept identical to the official implementation, therefore an
official SAM2 checkpoint (``sam2_hiera_t.pt`` / ``sam2.1_hiera_s.pt`` / ...)
loads directly into this module -- see :func:`load_sam2_checkpoint`.

The only architectural addition is the pluggable adapter: before the first
block of each stage an optional lightweight bottleneck
(down-project -> GELU -> up-project -> GELU, residual) may be inserted.  The
four on/off switches are the "branch add/remove" ablation variable of the
design document.
"""

from functools import partial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import Adapter


# --------------------------------------------------------------------------- #
# windowing utilities (verbatim behaviour of sam2.modeling.backbones.utils)
# --------------------------------------------------------------------------- #
def window_partition(x: torch.Tensor, window_size: int):
    """(B, H, W, C) -> (B*nW, ws, ws, C), zero padded on the right/bottom."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw, hw):
    """Inverse of :func:`window_partition`, dropping the padding again."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def do_pool(x: torch.Tensor, pool: Optional[nn.Module]) -> torch.Tensor:
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    return x.permute(0, 2, 3, 1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MLP(nn.Module):
    """Same layout as ``sam2.modeling.sam2_utils.MLP`` (``layers.0/1``)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 activation=nn.GELU):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3),
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_pool=None):
        super().__init__()
        self.dim, self.dim_out, self.num_heads = dim, dim_out, num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        return self.proj(x)


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4.0, drop_path=0.0,
                 norm_layer="LayerNorm", q_stride=None, act_layer=nn.GELU,
                 window_size=0):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)
        self.dim, self.dim_out = dim, dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride,
                                     ceil_mode=False)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads,
                                        q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2,
                       activation=act_layer)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        shortcut = x  # B, H, W, C
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        window_size = self.window_size
        pad_hw = None
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        x = self.attn(x)
        if self.q_stride:
            # shapes changed because of Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --------------------------------------------------------------------------- #
# Hiera trunk with pluggable adapters
# --------------------------------------------------------------------------- #
HIERA_PRESETS = {
    # name: kwargs matching sam2/configs/sam2.1/sam2.1_hiera_*.yaml
    "tiny": dict(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                 global_att_blocks=(5, 7, 9),
                 window_pos_embed_bkg_spatial_size=(7, 7)),
    "small": dict(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2),
                  global_att_blocks=(7, 10, 13),
                  window_pos_embed_bkg_spatial_size=(7, 7)),
    "base_plus": dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3),
                      global_att_blocks=(12, 16, 20),
                      window_pos_embed_bkg_spatial_size=(14, 14)),
    "large": dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4),
                  global_att_blocks=(23, 33, 43),
                  window_pos_embed_bkg_spatial_size=(7, 7),
                  window_spec=(8, 4, 16, 8)),
}


class HieraAdapterBackbone(nn.Module):
    """Hiera trunk returning C2/C3/C4/C5 (strides 4/8/16/32).

    Args:
        variant: one of :data:`HIERA_PRESETS`.
        adapter_stages: which stages get a trainable adapter, 1-indexed.
            ``(1, 2, 3, 4)`` = all on, ``()`` = pure frozen features.
            This is the ablation variable of the design document.
        adapter_ratio: bottleneck ratio of the adapter (dim // ratio hidden).
        freeze_trunk: freeze every original Hiera parameter (SAM2-UNet recipe).
        train_norm: keep LayerNorms trainable even when the trunk is frozen
            (cheap partial unfreeze, used by the "fourth level" ablation).
    """

    def __init__(
        self,
        variant: str = "tiny",
        adapter_stages: Sequence[int] = (1, 2, 3, 4),
        adapter_ratio: int = 4,
        adapter_scale: float = 1.0,
        adapter_type: str = "mlp",
        adapter_kernel: int = 3,
        freeze_trunk: bool = True,
        train_norm: bool = False,
        unfreeze_stages: Sequence[int] = (),
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_spec: Sequence[int] = (8, 4, 14, 7),
        in_chans: int = 3,
        **overrides,
    ):
        super().__init__()
        if variant not in HIERA_PRESETS:
            raise ValueError(f"unknown Hiera variant {variant!r}; "
                             f"choose from {sorted(HIERA_PRESETS)}")
        cfg = dict(HIERA_PRESETS[variant])
        cfg.setdefault("window_spec", tuple(window_spec))
        cfg.update(overrides)

        embed_dim = cfg["embed_dim"]
        num_heads = cfg["num_heads"]
        stages = tuple(cfg["stages"])
        window_spec = tuple(cfg["window_spec"])
        global_att_blocks = tuple(cfg["global_att_blocks"])
        wpe_size = tuple(cfg["window_pos_embed_bkg_spatial_size"])
        assert len(stages) == len(window_spec) == 4

        self.variant = variant
        self.stages = stages
        self.window_spec = window_spec
        self.q_stride = q_stride
        self.window_pos_embed_bkg_spatial_size = wpe_size

        depth = sum(stages)
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.stage_starts = [0] + [e + 1 for e in self.stage_ends[:-1]]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *wpe_size))
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, window_spec[0], window_spec[0])
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = window_spec[cur_stage - 1]
            if global_att_blocks is not None:
                window_size = 0 if i in global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            self.blocks.append(
                MultiScaleBlock(
                    dim=embed_dim, dim_out=dim_out, num_heads=num_heads,
                    drop_path=dpr[i],
                    q_stride=q_stride if i in self.q_pool_blocks else None,
                    window_size=window_size,
                )
            )
            embed_dim = dim_out

        # channels of the C2..C5 outputs (ascending stride)
        self.out_channels: List[int] = [self.blocks[i].dim_out for i in self.stage_ends]
        self.out_strides: List[int] = [4, 8, 16, 32]

        # ---- pluggable adapters (the trainable "branches") ------------------
        self.adapter_stages = tuple(sorted(set(int(s) for s in adapter_stages)))
        if any(s < 1 or s > 4 for s in self.adapter_stages):
            raise ValueError(f"adapter_stages must be within 1..4, got {adapter_stages}")
        self.adapters = nn.ModuleDict()
        for s in self.adapter_stages:
            in_dim = self.blocks[self.stage_starts[s - 1]].dim
            self.adapters[str(s)] = Adapter(in_dim, ratio=adapter_ratio,
                                            scale=adapter_scale,
                                            adapter_type=adapter_type,
                                            kernel_size=adapter_kernel)

        self.freeze_trunk_flag = freeze_trunk
        self.train_norm = train_norm
        self.unfreeze_stages = tuple(sorted(set(int(s) for s in unfreeze_stages)))
        if freeze_trunk:
            self.freeze_trunk(train_norm=train_norm,
                              unfreeze_stages=self.unfreeze_stages)

    # ------------------------------------------------------------------ #
    def freeze_trunk(self, train_norm: bool = False,
                     unfreeze_stages: Sequence[int] = ()):
        """Freeze everything except the adapters (SAM2-UNet training recipe).

        ``train_norm`` additionally trains every LayerNorm, and
        ``unfreeze_stages`` fully unfreezes the blocks of the given stages --
        both are partial-fine-tuning knobs for the domain-gap ablation.
        """
        for name, p in self.named_parameters():
            if name.startswith("adapters."):
                continue
            p.requires_grad = False
        if train_norm:
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad = True
        for s in unfreeze_stages:
            if not 1 <= s <= 4:
                raise ValueError(f"unfreeze_stages must be within 1..4, got {s}")
            for i in range(self.stage_starts[s - 1], self.stage_ends[s - 1] + 1):
                for p in self.blocks[i].parameters():
                    p.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the frozen trunk in eval mode (no drop-path on frozen blocks)."""
        super().train(mode)
        if mode and self.freeze_trunk_flag:
            unfrozen = set()
            for st in self.unfreeze_stages:
                unfrozen.update(range(self.stage_starts[st - 1],
                                      self.stage_ends[st - 1] + 1))
            for i, block in enumerate(self.blocks):
                block.train(i in unfrozen)
            self.patch_embed.eval()
            for a in self.adapters.values():
                a.train(True)
        return self

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        return pos_embed.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns [C2, C3, C4, C5] as NCHW tensors (strides 4/8/16/32)."""
        x = self.patch_embed(x)                       # B H W C
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            if i in self.stage_starts:
                stage = self.stage_starts.index(i) + 1
                if str(stage) in self.adapters:
                    x = self.adapters[str(stage)](x)
            x = block(x)
            if i in self.stage_ends:
                outputs.append(x.permute(0, 3, 1, 2).contiguous())
        return outputs

    # ------------------------------------------------------------------ #
    def load_sam2_checkpoint(self, path: str, strict_shapes: bool = True,
                             verbose: bool = True) -> dict:
        """Load an official SAM2 ``.pt`` checkpoint into the trunk.

        Handles the ``model.image_encoder.trunk.`` / ``image_encoder.trunk.``
        prefixes used by the released checkpoints.  Adapters (which do not
        exist in the checkpoint) are left at their initialisation.
        """
        obj = torch.load(path, map_location="cpu", weights_only=False)
        sd = obj.get("model", obj) if isinstance(obj, dict) else obj
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        trunk = {}
        for k, v in sd.items():
            for prefix in ("model.image_encoder.trunk.", "image_encoder.trunk.",
                           "trunk."):
                if k.startswith(prefix):
                    trunk[k[len(prefix):]] = v
                    break
            else:
                if k.startswith(("patch_embed.", "blocks.", "pos_embed")):
                    trunk[k] = v
        if not trunk:
            raise RuntimeError(f"no Hiera trunk weights found in {path}")

        own = self.state_dict()
        skipped = []
        if strict_shapes:
            for k in list(trunk):
                if k in own and own[k].shape != trunk[k].shape:
                    skipped.append(k)
                    trunk.pop(k)
        report = self.load_state_dict(trunk, strict=False)
        missing = [k for k in report.missing_keys if not k.startswith("adapters.")]
        if verbose:
            print(f"[hiera] loaded {len(trunk)} tensors from {path}")
            if missing:
                print(f"[hiera] missing (kept random): {len(missing)} -> {missing[:6]}")
            if skipped:
                print(f"[hiera] shape mismatch, skipped: {skipped[:6]}")
            if report.unexpected_keys:
                print(f"[hiera] unexpected: {len(report.unexpected_keys)}")
        return {"missing": missing, "unexpected": list(report.unexpected_keys),
                "skipped": skipped}
