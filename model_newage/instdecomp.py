"""Design 2: predict how many boxes a defect region becomes, instead of NMS.

The measurements that motivate this.  On crazing our detector localises 86% of
the ground truth at IoU>=0.5 and classifies it perfectly, yet AP50 is 0.46 --
because it emits ~5x more confident boxes than there are annotations, and NMS
cannot merge them (their box IoU is too low).  The reference implementation
over-predicts at exactly the same rate, so this is a property of the task, not
of one model.  Meanwhile the annotation itself is ambiguous in *count*: 43
crazing images carry one box, 142 carry two, 94 carry three, for visually
similar surfaces.

But the count is not random.  A fixed "split the frame top and bottom" layout
scores 0.2806 on crazing with oracle class presence -- far above every other
fixed layout -- which says the annotators followed a convention.  NMS cannot
learn a convention; a head can.

So: predict a per-class defect region, predict how many instances it should be
cut into, and predict the cut.  Slots are supervised in a canonical order (top
to bottom, then left to right), which is the convention the trivial-layout
result points at.

This head replaces DecoupledHead and needs its own loss, so it does not drop
into YOLOSAM2 unchanged -- build it with `InstanceDecompModel`.

    python -m model_newage.instdecomp        # shape and loss self-test
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Conv

MAX_SLOTS = 5          # NEU-DET never annotates more than 5 boxes of one class


def canonical_order(boxes: torch.Tensor) -> torch.Tensor:
    """Sort boxes top-to-bottom then left-to-right.

    Slot k is trained to predict the k-th box in this order.  A canonical order
    is what makes slot supervision well defined without Hungarian matching, and
    this particular order is the one the fixed-layout probe pointed at.
    """
    if boxes.numel() == 0:
        return boxes
    key = boxes[:, 1] * 10000 + boxes[:, 0]              # y1 major, x1 minor
    return boxes[key.argsort()]


def rasterise(boxes: torch.Tensor, h: int, w: int, imgsz: int) -> torch.Tensor:
    """(K,4) xyxy pixel boxes -> (K,h,w) binary masks on the feature grid."""
    if boxes.numel() == 0:
        return torch.zeros(0, h, w, device=boxes.device)
    ys = torch.arange(h, device=boxes.device).view(1, h, 1) * (imgsz / h)
    xs = torch.arange(w, device=boxes.device).view(1, 1, w) * (imgsz / w)
    x1, y1, x2, y2 = (boxes[:, i].view(-1, 1, 1) for i in range(4))
    return ((xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)).float()


class InstanceDecompHead(nn.Module):
    """Region map + instance count + slot assignment, all at one stride.

    One stride only: the diffuse classes this targets have boxes of 216-489 px,
    so a pyramid buys nothing here and a single coarse map keeps the count and
    split heads looking at the whole defect at once.
    """

    def __init__(self, in_ch: int, nc: int = 6, hidden: int = 128,
                 max_slots: int = MAX_SLOTS, imgsz: int = 640, stride: int = 16):
        super().__init__()
        self.nc, self.max_slots = nc, max_slots
        self.imgsz, self.stride = imgsz, stride
        self.stem = nn.Sequential(Conv(in_ch, hidden, 3, 1),
                                  Conv(hidden, hidden, 3, 1))
        self.region = nn.Conv2d(hidden, nc, 1)                 # per-class region
        self.slots = nn.Conv2d(hidden, nc * max_slots, 1)      # per-class split
        self.count = nn.Linear(hidden, nc * max_slots)         # per-class count
        nn.init.constant_(self.region.bias, -4.0)              # sparse prior

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.stem(feat)
        b, _, h, w = f.shape
        return {
            "region": self.region(f),                                  # (B,nc,h,w)
            "slots": self.slots(f).view(b, self.nc, self.max_slots, h, w),
            "count": self.count(f.mean((2, 3))).view(b, self.nc, self.max_slots),
        }

    @torch.no_grad()
    def decode(self, out: Dict[str, torch.Tensor],
               region_thresh: float = 0.5) -> List[torch.Tensor]:
        """-> list of (n,6) [x1,y1,x2,y2,conf,cls], no NMS anywhere."""
        region = out["region"].sigmoid()
        slots = out["slots"].softmax(2)
        count = out["count"].softmax(-1)
        b, nc, h, w = region.shape
        scale = self.imgsz / h
        results = []
        for i in range(b):
            rows = []
            for c in range(nc):
                m = region[i, c] > region_thresh
                if not m.any():
                    continue
                k = int(count[i, c].argmax()) + 1
                # each slot claims the pixels it wins among the first k
                claim = slots[i, c, :k].argmax(0)
                for s in range(k):
                    sel = m & (claim == s)
                    if not sel.any():
                        continue
                    ys, xs = torch.nonzero(sel, as_tuple=True)
                    conf = float(region[i, c][sel].mean() * count[i, c, k - 1])
                    rows.append([float(xs.min()) * scale, float(ys.min()) * scale,
                                 float(xs.max() + 1) * scale,
                                 float(ys.max() + 1) * scale, conf, float(c)])
            results.append(torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
                           if rows else torch.zeros(0, 6))
        return results


class InstanceDecompLoss(nn.Module):
    """Region BCE + count cross-entropy + slot cross-entropy on box pixels."""

    def __init__(self, head: InstanceDecompHead, w_region: float = 1.0,
                 w_count: float = 0.5, w_slot: float = 1.0):
        super().__init__()
        self.head = head
        self.w = (w_region, w_count, w_slot)

    def forward(self, out: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        region, slots, count = out["region"], out["slots"], out["count"]
        b, nc, h, w = region.shape
        dev = region.device
        S = self.head.imgsz

        t_region = torch.zeros_like(region)
        t_count = torch.zeros(b, nc, dtype=torch.long, device=dev)
        slot_loss = region.new_zeros(())
        n_slot = 0

        for i in range(b):
            rows = targets[targets[:, 0] == i]
            for c in range(nc):
                sel = rows[rows[:, 1].long() == c]
                if sel.numel() == 0:
                    continue
                cx, cy, bw, bh = (sel[:, 2] * S, sel[:, 3] * S,
                                  sel[:, 4] * S, sel[:, 5] * S)
                boxes = torch.stack([cx - bw / 2, cy - bh / 2,
                                     cx + bw / 2, cy + bh / 2], 1).to(dev)
                boxes = canonical_order(boxes)[: self.head.max_slots]
                masks = rasterise(boxes, h, w, S)                  # (K,h,w)
                t_region[i, c] = masks.amax(0)
                t_count[i, c] = boxes.shape[0] - 1
                # a pixel's slot label is the box it belongs to; overlaps go to
                # the earlier box, matching the canonical order
                lab = torch.full((h, w), -100, dtype=torch.long, device=dev)
                for k in range(masks.shape[0] - 1, -1, -1):
                    lab[masks[k] > 0] = k
                if (lab >= 0).any():
                    slot_loss = slot_loss + F.cross_entropy(
                        slots[i, c].unsqueeze(0), lab.unsqueeze(0),
                        ignore_index=-100)
                    n_slot += 1

        l_region = F.binary_cross_entropy_with_logits(region, t_region)
        l_count = F.cross_entropy(count.reshape(-1, self.head.max_slots),
                                  t_count.reshape(-1))
        l_slot = slot_loss / max(n_slot, 1)
        wr, wc, ws = self.w
        total = wr * l_region + wc * l_count + ws * l_slot
        return total, {"region": float(l_region), "count": float(l_count),
                       "slot": float(l_slot), "total": float(total)}


class InstanceDecompModel(nn.Module):
    """Backbone + neck + the decomposition head, with the usual API."""

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()
        from model.backbone_cnn import CSPBackbone
        from model.dataset import NEU_CLASSES
        from model.neck import build_neck

        cfg = dict(cfg or {})
        self.cfg = cfg
        self.nc = int(cfg.get("nc", 6))
        self.names = list(cfg.get("names") or NEU_CLASSES)[: self.nc]
        self.imgsz = int(cfg.get("imgsz", 640))
        self.backbone = CSPBackbone(**cfg.get("backbone", {}))
        neck_cfg = dict(cfg.get("neck", {}))
        neck_name = neck_cfg.pop("name", "panet")
        self.neck = build_neck(neck_name, self.backbone.out_channels,
                               **neck_cfg)
        # the level nearest stride 16: coarse enough to see a whole defect
        lvl = min(range(len(self.neck.out_strides)),
                  key=lambda i: abs(self.neck.out_strides[i] - 16))
        self.level = lvl
        self.head = InstanceDecompHead(self.neck.out_channels[lvl], nc=self.nc,
                                       imgsz=self.imgsz,
                                       stride=self.neck.out_strides[lvl])
        self.criterion = InstanceDecompLoss(self.head)

    def forward(self, x, targets=None):
        out = self.head(self.neck(self.backbone(x))[self.level])
        if self.training and targets is not None:
            return self.criterion(out, targets)
        return out

    @torch.no_grad()
    def predict(self, x, region_thresh: float = 0.5):
        was = self.training
        self.eval()
        r = self.head.decode(self.head(self.neck(self.backbone(x))[self.level]),
                             region_thresh)
        if was:
            self.train()
        return r

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_counts(self) -> Dict:
        t = sum(p.numel() for p in self.parameters()) / 1e6
        return {"trunk_frozen_M": 0.0, "adapters_M": 0.0,
                "neck_M": sum(p.numel() for p in self.neck.parameters()) / 1e6,
                "head_M": sum(p.numel() for p in self.head.parameters()) / 1e6,
                "total_M": round(t, 3), "trainable_M": round(t, 3),
                "adapter_stages": []}


def _selftest():
    torch.manual_seed(0)
    m = InstanceDecompModel({"imgsz": 640, "backbone": {"width": 0.5,
                                                        "depth": 0.34}})
    x = torch.randn(2, 3, 640, 640)
    # image 0: two crazing boxes stacked vertically; image 1: one pitted box
    tgt = torch.tensor([
        [0., 0., .50, .25, .90, .45],
        [0., 0., .50, .75, .90, .45],
        [1., 3., .50, .50, .95, .95],
    ])
    m.train()
    loss, items = m(x, tgt)
    loss.backward()
    print(f"[ok] 训练前向 loss={items['total']:.4f}  "
          f"region={items['region']:.4f} count={items['count']:.4f} "
          f"slot={items['slot']:.4f}")
    dets = m.predict(x, region_thresh=0.05)
    print(f"[ok] 解码 (无 NMS): 每图框数 {[int(d.shape[0]) for d in dets]}")
    print(f"[ok] 参数 {m.param_counts()['total_M']}M   "
          f"检测头输出层级 stride={m.neck.out_strides[m.level]}")
    b = torch.tensor([[0., 300., 100., 400.], [0., 0., 100., 100.]])
    print(f"[ok] 规范顺序 (先上后下): {canonical_order(b)[:, 1].tolist()}")


if __name__ == "__main__":
    _selftest()
