"""Task-aligned assignment + detection losses (BCE / CIoU / DFL / objectness)."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import bbox2dist, dist2bbox


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = False,
             CIoU: bool = False, SIoU: bool = False, eps: float = 1e-7):
    """Pairwise (broadcasting) IoU / CIoU / SIoU between xyxy boxes."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1 = (x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2)
        b2 = (x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2)
    else:
        b1 = box1.chunk(4, -1)
        b2 = box2.chunk(4, -1)
    b1_x1, b1_y1, b1_x2, b1_y2 = b1
    b2_x1, b2_y1, b2_x2, b2_y2 = b2

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    w1_, h1_ = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2_, h2_ = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1_ * h1_ + w2_ * h2_ - inter + eps
    iou = inter / union

    if not (CIoU or SIoU):
        return iou

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    if CIoU:
        c2 = cw.pow(2) + ch.pow(2) + eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
        import math
        v = (4 / math.pi ** 2) * (torch.atan(w2_ / (h2_ + eps)) -
                                  torch.atan(w1_ / (h1_ + eps))).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)
    # SIoU (angle + distance + shape cost), useful for elongated scratches
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
    sigma = (s_cw.pow(2) + s_ch.pow(2)).sqrt() + eps
    sin_alpha = (torch.min(s_cw.abs(), s_ch.abs()) / sigma).clamp(-1, 1)
    import math
    angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
    rho_x = (s_cw / (cw + eps)).pow(2)
    rho_y = (s_ch / (ch + eps)).pow(2)
    gamma = 2 - angle_cost
    dist_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)
    omiga_w = (w1_ - w2_).abs() / (w1_.maximum(w2_) + eps)
    omiga_h = (h1_ - h2_).abs() / (h1_.maximum(h2_) + eps)
    shape_cost = (1 - torch.exp(-omiga_w)).pow(4) + (1 - torch.exp(-omiga_h)).pow(4)
    return iou - 0.5 * (dist_cost + shape_cost)


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """Mask of anchor centres that fall inside a ground-truth box."""
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), 2)
    return deltas.view(bs, n_boxes, n_anchors, 4).amin(3).gt_(eps)


class TaskAlignedAssigner(nn.Module):
    """TOOD/YOLOv8 assigner: align(score^alpha * iou^beta) top-k per GT."""

    def __init__(self, topk: int = 10, num_classes: int = 6, alpha: float = 0.5,
                 beta: float = 6.0, eps: float = 1e-9, size_tol: float = 1.5):
        super().__init__()
        self.topk, self.num_classes = topk, num_classes
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.size_tol = size_tol

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt,
                anc_strides=None, reg_max=None):
        """
        Args (boxes in image pixels):
            pd_scores: (B, A, nc) sigmoid scores
            pd_bboxes: (B, A, 4) xyxy
            anc_points: (A, 2)
            gt_labels: (B, M, 1); gt_bboxes: (B, M, 4); mask_gt: (B, M, 1)
            anc_strides: (A,) stride of each anchor; when given together with
                ``reg_max`` an anchor may only be assigned to a ground-truth
                box whose longest side fits inside ``tol`` times the DFL range
                ``2 * (reg_max - 1) * stride``.

                MEASURED ON NEU-DET: switching this on with ``tol=1.0`` costs
                ~0.10 mAP.  Clamping a slightly-too-large target is far less
                harmful than banishing the box to a coarser level -- a 448 px
                scratch predicted at its 368 px P3 limit still reaches IoU
                ~0.8, while moving it to P4/P5 destroys its 77 px width.  Keep
                it off unless you are reproducing that ablation.
        Returns target_labels, target_bboxes, target_scores, fg_mask
        """
        bs, n_max = gt_bboxes.shape[:2]
        device = gt_bboxes.device
        if n_max == 0:
            return (torch.full_like(pd_scores[..., 0], self.num_classes).long(),
                    torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0]).bool())

        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        if anc_strides is not None and reg_max is not None:
            gt_side = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).amax(-1)   # (B, M)
            reach = self.size_tol * 2.0 * (reg_max - 1) * anc_strides.view(1, 1, -1)
            keep = gt_side.unsqueeze(-1) <= reach
            # never leave a ground-truth box without any candidate anchor
            keep = keep | (~keep.any(-1, keepdim=True))
            mask_in_gts = mask_in_gts * keep
        # alignment metric
        ind = torch.zeros([2, bs, n_max], dtype=torch.long, device=device)
        ind[0] = torch.arange(bs, device=device).view(-1, 1).expand(-1, n_max)
        ind[1] = gt_labels.squeeze(-1).long()
        bbox_scores = pd_scores[ind[0], :, ind[1]]                    # (B, M, A)
        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1),
                            CIoU=True).squeeze(-1).clamp_(0)          # (B, M, A)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        align_metric = align_metric * mask_in_gts * mask_gt

        # top-k candidates per GT
        topk = min(self.topk, align_metric.shape[-1])
        topk_metrics, topk_idxs = align_metric.topk(topk, dim=-1, largest=True)
        mask_topk = torch.zeros_like(align_metric, dtype=torch.bool)
        mask_topk.scatter_(-1, topk_idxs, topk_metrics > 0)
        mask_pos = mask_topk * mask_in_gts * mask_gt.bool()

        # resolve anchors matched to several GTs -> keep the highest IoU
        fg_mask = mask_pos.sum(1)
        if fg_mask.max() > 1:
            multi = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max, -1)
            max_idx = overlaps.argmax(1)
            is_max = torch.zeros_like(mask_pos)
            is_max.scatter_(1, max_idx.unsqueeze(1), True)
            mask_pos = torch.where(multi, is_max, mask_pos)
            fg_mask = mask_pos.sum(1)
        target_gt_idx = mask_pos.float().argmax(1)                    # (B, A)

        # gather targets
        batch_ind = torch.arange(bs, device=device, dtype=torch.long).unsqueeze(-1)
        flat_idx = target_gt_idx + batch_ind * n_max
        target_labels = gt_labels.long().flatten()[flat_idx]          # (B, A)
        target_bboxes = gt_bboxes.view(-1, 4)[flat_idx]               # (B, A, 4)
        target_labels.clamp_(0, self.num_classes - 1)
        target_scores = F.one_hot(target_labels, self.num_classes).float()
        fg = fg_mask.bool()
        target_scores = target_scores * fg.unsqueeze(-1)

        # normalise the soft label by the alignment metric (TOOD)
        align_metric = align_metric * mask_pos
        pos_align = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm = (align_metric * pos_overlaps / (pos_align + self.eps)).amax(-2)
        target_scores = target_scores * norm.unsqueeze(-1)
        return target_labels, target_bboxes, target_scores, fg


class DetectionLoss(nn.Module):
    """Total loss = box(CIoU) + cls(BCE, soft TAL labels) + dfl + obj."""

    def __init__(self, head, nc: int = 6, box_w: float = 7.5, cls_w: float = 0.5,
                 dfl_w: float = 1.5, obj_w: float = 1.0, topk: int = 10,
                 iou_type: str = "ciou", size_aware_assign: bool = False,
                 size_tol: float = 1.5):
        super().__init__()
        self.nc = nc
        self.reg_max = head.reg_max
        self.strides = head.strides
        self.use_obj = head.use_obj
        self.dfl = head.dfl
        self.weights = dict(box=box_w, cls=cls_w, dfl=dfl_w, obj=obj_w)
        self.iou_type = iou_type
        self.size_aware_assign = size_aware_assign
        self.assigner = TaskAlignedAssigner(topk=topk, num_classes=nc,
                                            size_tol=size_tol)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.proj = torch.arange(self.reg_max, dtype=torch.float)

    # -------------------------------------------------------------- #
    def _decode_reg(self, reg: torch.Tensor, anchors: torch.Tensor):
        """(B, 4*reg_max, A) -> xyxy in stride units."""
        b, _, a = reg.shape
        d = reg.view(b, 4, self.reg_max, a).permute(0, 3, 1, 2).softmax(-1)
        d = d.matmul(self.proj.to(reg.device).type(reg.dtype))        # (B, A, 4)
        return dist2bbox(d, anchors.unsqueeze(0))

    def _dfl_loss(self, pred_dist, target_ltrb):
        tl = target_ltrb.long()
        tr = tl + 1
        wl = tr - target_ltrb
        wr = 1 - wl
        loss = (F.cross_entropy(pred_dist, tl.view(-1), reduction="none") * wl.view(-1)
                + F.cross_entropy(pred_dist, tr.clamp(max=self.reg_max - 1).view(-1),
                                  reduction="none") * wr.view(-1))
        return loss.view(target_ltrb.shape[0], 4).mean(-1, keepdim=True)

    @staticmethod
    def strides_seen(s_flat) -> list:
        return [int(v) for v in torch.unique(s_flat).tolist()]

    def forward(self, raw: Dict, targets: torch.Tensor, head) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            raw: dict returned by :class:`DecoupledHead` in training mode
            targets: (N, 6) ``[batch_idx, cls, cx, cy, w, h]`` normalised xywh
        """
        cls, reg, obj, anchors, strides = head.flatten(raw)
        device = cls.device
        bs = cls.shape[0]
        anchors_s = anchors / strides                                  # stride units

        # ---- build (B, M, 5) ground truth ------------------------------ #
        gt_labels, gt_bboxes, mask_gt = self._prepare_targets(
            targets, bs, raw["feats"], device, head.strides[0]
        )

        pred_bboxes = self._decode_reg(reg, anchors_s)                  # grid units
        # NOTE: the alignment metric uses the class score ALONE, as in TOOD /
        # YOLOv8.  Folding the objectness score in here creates a feedback
        # loop: obj is trained against 8400 anchors of which ~40 are positive,
        # so it collapses towards zero on whichever pyramid level falls behind
        # first, that level's alignment metric becomes 0, it stops receiving
        # positives, and it never recovers.  Measured on NEU-DET: all confident
        # detections came from stride 16 while strides 8 and 32 were dead.
        pred_scores = cls.detach().sigmoid()

        # Assignment happens in image pixels, the box/dfl losses in grid units.
        # The assigner MUST run in fp32.  Under autocast the predicted boxes
        # arrive as fp16, and CIoU squares the enclosing-box diagonal: at 640px
        # that is 640^2 = 409600, well past the fp16 ceiling of 65504.  The
        # penalty term overflows to inf/nan, the alignment metric fails the
        # `> 0` test, and the ground truth gets no positives at all -- but only
        # for boxes big enough to overflow.  Measured on NEU-DET: stride 32
        # received exactly zero positives from the first epoch, and every
        # large-box class scored ~0.00 while the small-box classes were fine.
        # The reference casts with `.type(gt_bboxes.dtype)` for this reason.
        t_labels, t_bboxes, t_scores, fg = self.assigner(
            pred_scores.float(), (pred_bboxes.detach() * strides).float(),
            anchors.float(), gt_labels.float(), gt_bboxes.float(), mask_gt,
            anc_strides=strides.squeeze(-1) if self.size_aware_assign else None,
            reg_max=self.reg_max,
        )
        t_bboxes = (t_bboxes / strides).to(cls.dtype)
        t_scores = t_scores.to(cls.dtype)
        target_sum = max(t_scores.sum(), 1.0)

        # ---- classification -------------------------------------------- #
        loss_cls = self.bce(cls, t_scores).sum() / target_sum

        # ---- box + dfl -------------------------------------------------- #
        loss_box = torch.zeros(1, device=device)
        loss_dfl = torch.zeros(1, device=device)
        iou_full = torch.zeros_like(fg, dtype=cls.dtype)
        if fg.any():
            weight = t_scores.sum(-1)[fg].unsqueeze(-1)
            iou = bbox_iou(pred_bboxes[fg], t_bboxes[fg],
                           CIoU=self.iou_type == "ciou",
                           SIoU=self.iou_type == "siou")
            loss_box = ((1.0 - iou) * weight).sum() / target_sum
            iou_full[fg] = iou.detach().squeeze(-1).clamp(0, 1).to(iou_full.dtype)

            reg_flat = reg.permute(0, 2, 1).reshape(bs, -1, 4 * self.reg_max)
            target_ltrb = bbox2dist(t_bboxes, anchors_s.unsqueeze(0), self.reg_max)
            loss_dfl = (self._dfl_loss(
                reg_flat[fg].view(-1, self.reg_max), target_ltrb[fg]
            ) * weight).sum() / target_sum

        # ---- objectness (IoU-aware) ------------------------------------- #
        loss_obj = torch.zeros(1, device=device)
        if obj is not None:
            # normalised like the other terms, otherwise the ~40 positives are
            # diluted by 8400 anchors and the branch barely trains at all
            loss_obj = self.bce(obj.squeeze(-1), iou_full).sum() / target_sum

        w = self.weights
        total = (w["box"] * loss_box + w["cls"] * loss_cls +
                 w["dfl"] * loss_dfl + w["obj"] * loss_obj)
        items = {"box": float(loss_box.detach()), "cls": float(loss_cls.detach()),
                 "dfl": float(loss_dfl.detach()), "obj": float(loss_obj.detach()),
                 "total": float(total.detach()), "n_pos": int(fg.sum())}
        # positives per pyramid level.  A level that never receives any is
        # never trained, which is what "stride 32 emits 0 predictions" looks
        # like from the outside -- and it is invisible in the total loss.
        s_flat = strides.view(-1)
        for s in self.strides_seen(s_flat):
            items[f"p{s}"] = int(fg[:, s_flat == s].sum())
        return total.squeeze() * bs, items

    # -------------------------------------------------------------- #
    @staticmethod
    def _prepare_targets(targets, bs, feats, device, stride0):
        """(N, 6) normalised targets -> padded (B, M, 1) labels / (B, M, 4) xyxy
        boxes in image pixels."""
        _, _, h, w = feats[0].shape
        img_h = h * float(stride0)
        img_w = w * float(stride0)
        if targets.numel() == 0:
            return (torch.zeros(bs, 0, 1, device=device),
                    torch.zeros(bs, 0, 4, device=device),
                    torch.zeros(bs, 0, 1, device=device))
        targets = targets.to(device)
        bi = targets[:, 0].long()
        counts = torch.bincount(bi, minlength=bs)
        m = int(counts.max())
        gt_labels = torch.zeros(bs, m, 1, device=device)
        gt_bboxes = torch.zeros(bs, m, 4, device=device)
        mask_gt = torch.zeros(bs, m, 1, device=device)
        for b in range(bs):
            sel = targets[bi == b]
            n = sel.shape[0]
            if n == 0:
                continue
            gt_labels[b, :n, 0] = sel[:, 1]
            cx, cy = sel[:, 2] * img_w, sel[:, 3] * img_h
            bw, bh = sel[:, 4] * img_w, sel[:, 5] * img_h
            gt_bboxes[b, :n] = torch.stack(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1
            )
            mask_gt[b, :n, 0] = 1.0
        return gt_labels, gt_bboxes, mask_gt
