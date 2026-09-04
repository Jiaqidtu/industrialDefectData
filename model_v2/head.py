"""Anchor-free decoupled detection head (cls / reg-DFL / obj) + decoding.

The three branches match the design diagram: a classification branch over the
six NEU-DET defect classes, a regression branch predicting the four distances
to the box sides through a Distribution-Focal-Loss discretisation, and an
objectness branch whose score is multiplied into the class score at inference.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .layers import Conv


def make_anchors(feats: Sequence[torch.Tensor], strides: Sequence[int],
                 grid_cell_offset: float = 0.5):
    """Anchor points (in input-image pixels) and their stride, per level."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2) * stride)
        stride_tensor.append(
            torch.full((h * w, 1), float(stride), dtype=dtype, device=device)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor,
              xywh: bool = False) -> torch.Tensor:
    """ltrb distances (already in pixels) -> xyxy (or xywh) boxes."""
    lt, rb = distance.chunk(2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), -1)
    return torch.cat((x1y1, x2y2), -1)


def bbox2dist(bbox: torch.Tensor, anchor_points: torch.Tensor,
              reg_max: int) -> torch.Tensor:
    """xyxy boxes (in stride units) -> ltrb targets clamped to the DFL range."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points),
                     -1).clamp_(0, reg_max - 1 - 0.01)


class DFL(nn.Module):
    """Integral of the discrete distribution over ``reg_max`` bins."""

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False).requires_grad_(False)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.arange(reg_max, dtype=torch.float).view(1, reg_max, 1, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4*reg_max, A)
        b, _, a = x.shape
        x = x.view(b, 4, self.reg_max, a).transpose(2, 1).softmax(1)
        return self.conv(x).view(b, 4, a)


class DecoupledHead(nn.Module):
    """Per-level decoupled head sharing nothing but the neck features.

    Outputs during training: raw ``(cls, reg, obj)`` maps per level.
    Outputs during inference: ``(B, 4 + nc, A)`` with xyxy boxes in pixels and
    class scores already multiplied by the objectness score.
    """

    def __init__(self, nc: int = 6, in_channels: Sequence[int] = (256, 256, 256),
                 strides: Sequence[int] = (8, 16, 32), reg_max: int = 16,
                 hidden: int = None, use_obj: bool = False, n_conv: int = 2,
                 imgsz: int = 640):
        super().__init__()
        self.nc = nc
        self.nl = len(in_channels)
        self.reg_max = reg_max
        self.strides = list(strides)
        self.use_obj = use_obj
        self.no = nc + 4 * reg_max + (1 if use_obj else 0)

        c_cls = hidden or max(in_channels[0] // 2, min(nc * 4, 128))
        c_reg = hidden or max(in_channels[0] // 2, 4 * reg_max)

        def stem(c_in, c_out):
            layers = [Conv(c_in, c_out, 3, 1)]
            layers += [Conv(c_out, c_out, 3, 1) for _ in range(n_conv - 1)]
            return nn.Sequential(*layers)

        self.cls_stem = nn.ModuleList(stem(c, c_cls) for c in in_channels)
        self.reg_stem = nn.ModuleList(stem(c, c_reg) for c in in_channels)
        self.cls_pred = nn.ModuleList(nn.Conv2d(c_cls, nc, 1) for _ in in_channels)
        self.reg_pred = nn.ModuleList(
            nn.Conv2d(c_reg, 4 * reg_max, 1) for _ in in_channels
        )
        self.obj_pred = nn.ModuleList(
            nn.Conv2d(c_reg, 1, 1) for _ in in_channels
        ) if use_obj else None

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        self._init_biases(imgsz=imgsz)

    def _init_biases(self, imgsz: int = 640):
        """Stride-aware classification prior (the YOLOv8 `bias_init` recipe).

        A single constant prior across levels is wrong here: the stride-8 map
        holds 16x more anchors than stride-32, so an identical per-anchor
        false-positive prior gives the fine level 16x the negative BCE mass.
        The optimiser's cheapest response is to push *every* logit down, which
        leaves the model permanently under-confident and starves whichever
        level loses the race.  Scaling the prior by the number of anchors per
        level, log(5 / nc / (imgsz/stride)^2), removes that imbalance.
        """
        import math
        for m, s in zip(self.cls_pred, self.strides):
            nn.init.constant_(m.bias, math.log(5 / self.nc / (imgsz / s) ** 2))
            nn.init.normal_(m.weight, std=0.01)
        if self.obj_pred is not None:
            for m, s in zip(self.obj_pred, self.strides):
                nn.init.constant_(m.bias, math.log(5 / (imgsz / s) ** 2))
                nn.init.normal_(m.weight, std=0.01)
        for m in self.reg_pred:
            nn.init.constant_(m.bias, 1.0)

    def forward(self, feats: List[torch.Tensor]):
        cls_out, reg_out, obj_out = [], [], []
        for i, f in enumerate(feats):
            cf = self.cls_stem[i](f)
            rf = self.reg_stem[i](f)
            cls_out.append(self.cls_pred[i](cf))
            reg_out.append(self.reg_pred[i](rf))
            if self.obj_pred is not None:
                obj_out.append(self.obj_pred[i](rf))
        raw = {"cls": cls_out, "reg": reg_out, "obj": obj_out, "feats": feats}
        if self.training:
            return raw
        return self.decode(raw), raw

    # ------------------------------------------------------------------ #
    def flatten(self, raw) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor]:
        """Concatenate the per-level maps into (B, A, *) tensors."""
        b = raw["cls"][0].shape[0]
        cls = torch.cat([c.view(b, self.nc, -1) for c in raw["cls"]], 2).permute(0, 2, 1)
        reg = torch.cat([r.view(b, 4 * self.reg_max, -1) for r in raw["reg"]], 2)
        obj = (torch.cat([o.view(b, 1, -1) for o in raw["obj"]], 2).permute(0, 2, 1)
               if raw["obj"] else None)
        anchors, strides = make_anchors(raw["feats"], self.strides)
        return cls, reg, obj, anchors, strides

    def decode(self, raw) -> torch.Tensor:
        """-> (B, A, 4 + nc): xyxy pixel boxes and final per-class scores."""
        cls, reg, obj, anchors, strides = self.flatten(raw)
        dist = self.dfl(reg).permute(0, 2, 1) * strides            # (B, A, 4)
        boxes = dist2bbox(dist, anchors.unsqueeze(0))
        scores = cls.sigmoid()
        if obj is not None:
            scores = scores * obj.sigmoid()
        return torch.cat([boxes, scores], -1)


# --------------------------------------------------------------------------- #
# post-processing (pure torch, no torchvision dependency)
# --------------------------------------------------------------------------- #
def box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU matrix between (N,4) and (M,4) xyxy boxes."""
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + eps)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    """Greedy NMS, returns kept indices ordered by decreasing score."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_thres]
    return torch.stack(keep)


def non_max_suppression(pred: torch.Tensor, conf_thres: float = 0.001,
                        iou_thres: float = 0.65, max_det: int = 300,
                        max_nms: int = 30000, multi_label: bool = False,
                        agnostic: bool = False) -> List[torch.Tensor]:
    """(B, A, 4+nc) -> list of (n, 6) tensors ``[x1, y1, x2, y2, conf, cls]``."""
    outputs = []
    for x in pred:
        boxes, scores = x[:, :4], x[:, 4:]
        if multi_label:
            i, j = (scores > conf_thres).nonzero(as_tuple=True)
            det = torch.cat([boxes[i], scores[i, j, None], j[:, None].float()], 1)
        else:
            conf, j = scores.max(1, keepdim=True)
            det = torch.cat([boxes, conf, j.float()], 1)[conf.view(-1) > conf_thres]
        if det.shape[0] == 0:
            outputs.append(torch.zeros((0, 6), device=pred.device))
            continue
        if det.shape[0] > max_nms:
            det = det[det[:, 4].argsort(descending=True)[:max_nms]]
        # offset boxes per class so that NMS is class-aware in one pass
        offset = 0 if agnostic else det[:, 5:6] * 8192.0
        keep = nms(det[:, :4] + offset, det[:, 4], iou_thres)[:max_det]
        outputs.append(det[keep])
    return outputs
