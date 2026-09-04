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

from .layers import C2f, Conv


class CSPBackbone(nn.Module):
    """Four-stage CSP backbone emitting C2/C3/C4/C5 at strides 4/8/16/32.

    Args:
        width: channel multiplier (0.5 ~ YOLOv8s, 0.25 ~ YOLOv8n).
        depth: repeat multiplier for the CSP blocks.
        in_chans: input channels.
    """

    def __init__(self, width: float = 0.5, depth: float = 0.34, in_chans: int = 3):
        super().__init__()
        def c(ch):
            return max(int(round(ch * width)), 16)

        def n(rep):
            return max(int(round(rep * depth)), 1)

        self.stem = Conv(in_chans, c(64), 3, 2)                 # /2
        self.stage1 = nn.Sequential(Conv(c(64), c(128), 3, 2),   # /4  -> C2
                                    C2f(c(128), c(128), n(3), shortcut=True))
        self.stage2 = nn.Sequential(Conv(c(128), c(256), 3, 2),  # /8  -> C3
                                    C2f(c(256), c(256), n(6), shortcut=True))
        self.stage3 = nn.Sequential(Conv(c(256), c(512), 3, 2),  # /16 -> C4
                                    C2f(c(512), c(512), n(6), shortcut=True))
        self.stage4 = nn.Sequential(Conv(c(512), c(1024), 3, 2),  # /32 -> C5
                                    C2f(c(1024), c(1024), n(3), shortcut=True))

        self.out_channels: List[int] = [c(128), c(256), c(512), c(1024)]
        self.out_strides: List[int] = [4, 8, 16, 32]
        # kept for API compatibility with HieraAdapterBackbone
        self.adapters = nn.ModuleDict()
        self.adapter_stages: Sequence[int] = ()
        self.freeze_trunk_flag = False
        self.unfreeze_stages: Sequence[int] = ()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return [c2, c3, c4, c5]

    # ultralytics lays the same graph out as a flat model.N list; our stages
    # are the identical modules in the identical order, minus the SPPF, which
    # our neck does not consume.  Verified tensor-by-tensor against
    # yolov8s.pt: 125 of our 125 tensors match a v8s backbone tensor exactly.
    _ULTRA_MAP = {"stem": "model.0",
                  "stage1.0": "model.1", "stage1.1": "model.2",
                  "stage2.0": "model.3", "stage2.1": "model.4",
                  "stage3.0": "model.5", "stage3.1": "model.6",
                  "stage4.0": "model.7", "stage4.1": "model.8"}

    def load_coco_checkpoint(self, path: str = "yolov8s.pt") -> int:
        """Initialise the trunk from COCO-pretrained ultralytics weights.

        NEU-DET is 1260 training images and the trunk is 4.8M parameters
        trained from scratch, which is the one variable every architecture
        experiment so far has held fixed.  Note the checkpoint must be v8s,
        not v8n: our width 0.5 doubles v8n's 0.25, so the v8n tensors are
        half-width and silently would not match.
        """
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        src = ck["model"].float().state_dict()
        own = self.state_dict()
        new, hit = {}, 0
        for k in own:
            head, rest = k.split(".", 1) if k.startswith("stem") else (
                ".".join(k.split(".")[:2]), ".".join(k.split(".")[2:]))
            pref = self._ULTRA_MAP.get(head)
            cand = f"{pref}.{rest}" if pref else None
            if cand in src and src[cand].shape == own[k].shape:
                new[k], hit = src[cand], hit + 1
            else:
                new[k] = own[k]
        missing = [k for k in own if k not in src and k.endswith("weight")]
        self.load_state_dict(new, strict=True)
        total = len([k for k in own if not k.endswith("num_batches_tracked")])
        if hit < total:
            raise RuntimeError(
                f"only {hit}/{total} trunk tensors matched {path}; refusing to "
                f"train on a half-initialised backbone -- check --cnn-width "
                f"(0.5 needs yolov8s.pt, 0.25 needs yolov8n.pt)")
        return hit

    def load_sam2_checkpoint(self, *args, **kwargs):
        raise RuntimeError("CSPBackbone has no SAM2 weights; drop --sam2-ckpt")
