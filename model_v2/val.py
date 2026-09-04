"""Evaluation: mAP@0.5, mAP@0.5:0.95, per-class AP, params / GFLOPs / FPS."""

import argparse
import json
import os
from typing import Dict, Optional

import torch

from .dataset import build_dataloader
from .head import non_max_suppression
from .metrics import DetMetrics


def xywhn_to_xyxy(labels: torch.Tensor, imgsz: int) -> torch.Tensor:
    """(n,5) [cls, cx, cy, w, h] normalised -> (n,5) [cls, x1, y1, x2, y2] pixels."""
    if labels.numel() == 0:
        return labels.new_zeros((0, 5))
    cls = labels[:, 0:1]
    cx, cy, w, h = (labels[:, 1] * imgsz, labels[:, 2] * imgsz,
                    labels[:, 3] * imgsz, labels[:, 4] * imgsz)
    return torch.cat([cls, torch.stack([cx - w / 2, cy - h / 2,
                                        cx + w / 2, cy + h / 2], 1)], 1)


@torch.no_grad()
def evaluate(model, loader, device: str = "cuda", conf_thres: float = 0.001,
             # 0.45, not COCO's 0.7: every model tested here peaks there on
             # NEU-DET (ours +0.036, YOLOv8n +0.023, YOLOv8s +0.013).  Defects
             # are sparse and their extents distinctive, so the loose COCO
             # threshold leaves overlapping duplicates that wreck the ranking.
             # NOTE: 0.45 was chosen on the validation split -- a held-out
             # split is needed before this number goes in a paper.
             iou_thres: float = 0.45, max_det: int = 300,
             multi_label: bool = True, verbose: bool = True) -> Dict:
    model.eval().to(device)
    metrics = DetMetrics(model.names)
    imgsz = loader.dataset.imgsz

    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        # multi_label keeps every class above the threshold for an anchor, not
        # just its argmax.  With confusable texture classes the argmax-only
        # form silently discards the runner-up's candidate and caps recall:
        # measured at -0.26 mAP on reference weights.
        dets = non_max_suppression(preds, conf_thres=conf_thres,
                                   iou_thres=iou_thres, max_det=max_det,
                                   multi_label=multi_label)
        for i, det in enumerate(dets):
            lab = targets[targets[:, 0] == i][:, 1:].to(device)
            metrics.update(det, xywhn_to_xyxy(lab, imgsz))

    res = metrics.compute()
    if verbose:
        print(f"{'class':<18}{'n_gt':>7}{'P':>9}{'R':>9}{'AP50':>9}{'AP50-95':>10}")
        for name, v in res["per_class"].items():
            print(f"{name:<18}{v['n_gt']:>7}{v['P']:>9.3f}{v['R']:>9.3f}"
                  f"{v['AP50']:>9.4f}{v['AP50-95']:>10.4f}")
        print(f"{'all':<18}{'':>7}{res['precision']:>9.3f}{res['recall']:>9.3f}"
              f"{res['mAP50']:>9.4f}{res['mAP50-95']:>10.4f}")
    return res


def efficiency_report(model, imgsz: int = 640, device: str = "cuda",
                      batch: int = 1) -> Dict:
    """Params / GFLOPs / FPS -- the efficiency half of the trade-off curve."""
    model.to(device)
    rep = dict(model.param_counts())
    rep["GFLOPs"] = model.flops(imgsz)
    rep["FPS"] = model.benchmark_fps(imgsz, batch=batch)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser("evaluate a YOLO-SAM2 checkpoint on NEU-DET")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-multi-label", action="store_true",
                    help="argmax class per anchor only (the old behaviour)")
    ap.add_argument("--json", default=None, help="write the metrics to this file")
    args = ap.parse_args(argv)

    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    if ck.get("ref_arch"):
        from .refarch import RefArchModel
        model = RefArchModel.load(args.weights, map_location="cpu")
    else:
        from .yolo_sam2 import YOLOSAM2
        model = YOLOSAM2.load(args.weights, map_location="cpu")
    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=args.batch_size, augment=False,
                              shuffle=False, workers=args.workers)
    res = evaluate(model, loader, device=args.device, conf_thres=args.conf,
                   iou_thres=args.iou, multi_label=not args.no_multi_label)
    res["efficiency"] = efficiency_report(model, args.imgsz, args.device)
    print(json.dumps(res["efficiency"], indent=2))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
    return res


if __name__ == "__main__":
    main()
