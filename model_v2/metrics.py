"""COCO-style detection metrics: per-class AP, mAP@0.5 and mAP@0.5:0.95."""

from typing import Dict, List, Sequence

import numpy as np
import torch

from .head import box_iou

# numpy 2 renamed trapz -> trapezoid; the cluster has 1.23 (login node) and
# 2.x (conda env), so bind whichever exists
_trapz = getattr(np, "trapezoid", None) or np.trapz


def match_predictions(detections: torch.Tensor, labels: torch.Tensor,
                      iou_thresholds: torch.Tensor) -> torch.Tensor:
    """(n_det, n_thr) bool matrix of true positives.

    Args:
        detections: (n, 6) ``[x1, y1, x2, y2, conf, cls]`` sorted by conf desc.
        labels: (m, 5) ``[cls, x1, y1, x2, y2]``.
    """
    correct = torch.zeros(detections.shape[0], iou_thresholds.numel(),
                          dtype=torch.bool, device=detections.device)
    if labels.shape[0] == 0 or detections.shape[0] == 0:
        return correct
    iou = box_iou(labels[:, 1:], detections[:, :4])
    same_cls = labels[:, 0:1] == detections[:, 5].unsqueeze(0)
    iou = iou * same_cls
    for k, thr in enumerate(iou_thresholds):
        gt_i, det_i = torch.where(iou >= thr)
        if gt_i.numel() == 0:
            continue
        matches = torch.stack([gt_i, det_i, iou[gt_i, det_i]], 1).cpu().numpy()
        matches = matches[matches[:, 2].argsort()[::-1]]
        # After this de-duplication the rows are ordered by DETECTION index,
        # and detections arrive sorted by confidence -- so de-duplicating by
        # ground truth next keeps the most confident detection for each box.
        # Re-sorting by IoU here instead (which reads like an improvement)
        # awards the hit to the best-overlapping detection, which is often a
        # low-confidence one: the total number of true positives is unchanged
        # but they move down the ranking, the confident detection becomes a
        # false positive, and AP halves.  Measured at 0.359 against 0.672 on
        # reference weights.  COCO assigns greedily in confidence order.
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), k] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (COCO convention)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


class DetMetrics:
    """Accumulates per-image results and reduces them to AP statistics."""

    def __init__(self, names: Sequence[str], iou_thresholds: Sequence[float] = None):
        self.names = list(names)
        self.iouv = torch.tensor(
            iou_thresholds if iou_thresholds is not None
            else np.linspace(0.5, 0.95, 10).tolist(), dtype=torch.float32
        )
        self.stats: List[tuple] = []

    def update(self, detections: torch.Tensor, labels: torch.Tensor):
        """detections (n,6) xyxy/conf/cls; labels (m,5) cls/xyxy, same scale."""
        if detections.shape[0]:
            detections = detections[detections[:, 4].argsort(descending=True)]
        tp = match_predictions(detections, labels, self.iouv.to(detections.device))
        self.stats.append((
            tp.cpu().numpy(),
            detections[:, 4].cpu().numpy() if detections.shape[0] else np.zeros(0),
            detections[:, 5].cpu().numpy() if detections.shape[0] else np.zeros(0),
            labels[:, 0].cpu().numpy() if labels.shape[0] else np.zeros(0),
        ))

    def compute(self, eps: float = 1e-16) -> Dict:
        if not self.stats:
            return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0,
                    "per_class": {}}
        tp = np.concatenate([s[0] for s in self.stats], 0)
        conf = np.concatenate([s[1] for s in self.stats], 0)
        pred_cls = np.concatenate([s[2] for s in self.stats], 0)
        target_cls = np.concatenate([s[3] for s in self.stats], 0)

        order = (-conf).argsort()
        tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
        n_thr = tp.shape[1] if tp.size else self.iouv.numel()

        per_class, ap_all, p_all, r_all = {}, [], [], []
        for ci, name in enumerate(self.names):
            n_gt = int((target_cls == ci).sum())
            mask = pred_cls == ci
            n_p = int(mask.sum())
            if n_gt == 0:
                continue
            if n_p == 0:
                per_class[name] = {"AP50": 0.0, "AP50-95": 0.0, "P": 0.0, "R": 0.0,
                                   "n_gt": n_gt}
                ap_all.append([0.0] * n_thr)
                p_all.append(0.0)
                r_all.append(0.0)
                continue
            tpc = tp[mask].cumsum(0)
            fpc = (1 - tp[mask]).cumsum(0)
            recall = tpc / (n_gt + eps)
            precision = tpc / (tpc + fpc + eps)
            aps = [compute_ap(recall[:, k], precision[:, k]) for k in range(n_thr)]
            # P/R at the max-F1 operating point of the IoU=0.5 curve
            f1 = 2 * precision[:, 0] * recall[:, 0] / (precision[:, 0] + recall[:, 0] + eps)
            i = int(f1.argmax())
            per_class[name] = {"AP50": aps[0], "AP50-95": float(np.mean(aps)),
                               "P": float(precision[i, 0]), "R": float(recall[i, 0]),
                               "n_gt": n_gt}
            ap_all.append(aps)
            p_all.append(float(precision[i, 0]))
            r_all.append(float(recall[i, 0]))

        if not ap_all:
            return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0,
                    "per_class": {}}
        ap_arr = np.array(ap_all)
        return {
            "mAP50": float(ap_arr[:, 0].mean()),
            "mAP50-95": float(ap_arr.mean()),
            "precision": float(np.mean(p_all)),
            "recall": float(np.mean(r_all)),
            "per_class": per_class,
        }

    def reset(self):
        self.stats.clear()
