"""CSPDarknet-style CNN backbone -- the control group for the SAM2 trunk.

Without this, nothing in the results attributes the accuracy to SAM2: a run
with the frozen Hiera trunk only says "the whole pipeline reaches X", not
"the foundation-model features contributed Y".  This backbone plugs into the
exact same neck and head, is trained from scratch, and has the same C2..C5
interface, so it isolates the backbone as the single changed variable.
"""

from typing import List, Sequence

import torch
import torch.nn as nn

from .admsb import C2f_ADMSB
from .layers import C2f, Conv


class CSPBackbone(nn.Module):
    """Four-stage CSP backbone emitting C2/C3/C4/C5 at strides 4/8/16/32.

    Args:
        width: channel multiplier (0.5 ~ YOLOv8s, 0.25 ~ YOLOv8n).
        depth: repeat multiplier for the CSP blocks.
        in_chans: input channels.
        use_admsb: put ADMSB blocks in the P3/P4 C2f (False = stock baseline).
        n_admsb: how many leading Bottleneck slots per stage ADMSB replaces.
    """

    def __init__(self, width: float = 0.5, depth: float = 0.34, in_chans: int = 3,
                 use_admsb: bool = True, n_admsb: int = 2):
        super().__init__()
        def c(ch):
            return max(int(round(ch * width)), 16)

        def n(rep):
            return max(int(round(rep * depth)), 1)

        self.stem = Conv(in_chans, c(64), 3, 2)                 # /2
        self.stage1 = nn.Sequential(Conv(c(64), c(128), 3, 2),   # /4  -> C2
                                    C2f(c(128), c(128), n(3), shortcut=True))
        # P3 / P4 carry the small and mid-size defects, so their C2f blocks
        # use ADMSB instead of the stock Bottleneck (first `n_admsb` slots).
        # use_admsb=False gives the untouched baseline for the A/B comparison.
        def p34(ch, rep):
            if use_admsb:
                return C2f_ADMSB(c(ch), c(ch), n(rep), shortcut=True,
                                 n_admsb=n_admsb)
            return C2f(c(ch), c(ch), n(rep), shortcut=True)

        self.stage2 = nn.Sequential(Conv(c(128), c(256), 3, 2),  # /8  -> C3
                                    p34(256, 6))
        self.stage3 = nn.Sequential(Conv(c(256), c(512), 3, 2),  # /16 -> C4
                                    p34(512, 6))
        # P5 keeps the stock C2f -- ADMSB there would cost compute on the
        # stride-32 map where the large-defect gain is smallest.
        self.stage4 = nn.Sequential(Conv(c(512), c(1024), 3, 2),  # /32 -> C5
                                    C2f(c(1024), c(1024), n(3), shortcut=True))

        self.out_channels: List[int] = [c(128), c(256), c(512), c(1024)]
        self.out_strides: List[int] = [4, 8, 16, 32]
        # kept for API compatibility with HieraAdapterBackbone
        self.adapters = nn.ModuleDict()
        self.adapter_stages: Sequence[int] = ()
        self.use_admsb = use_admsb
        self.freeze_trunk_flag = False
        self.unfreeze_stages: Sequence[int] = ()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return [c2, c3, c4, c5]

    def load_sam2_checkpoint(self, *args, **kwargs):
        raise RuntimeError("CSPBackbone has no SAM2 weights; drop --sam2-ckpt")
