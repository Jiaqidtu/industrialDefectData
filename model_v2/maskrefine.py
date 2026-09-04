"""Use SAM2 masks to merge duplicate detections, at inference time only.

The measured bottleneck is not localisation (IoU>=.5 on 85% of ground truth,
classification near-perfect) but duplicates: on crazing both our model and the
reference emit ~5x more confident boxes than there are objects, and NMS cannot
tell a duplicate from a neighbour because it only sees box overlap.  A mask
sees the actual extent, so two boxes covering the same defect give nearly the
same mask even when their box IoU is low.

Nothing is retrained -- this measures whether the mask signal is worth building
on before any of it is built.

    python -m model.maskrefine --weights runs/u_nomosaic/best.pt
"""

import argparse
import os

import numpy as np
import torch

from .dataset import build_dataloader
from .head import non_max_suppression
from .metrics import DetMetrics
from .refarch import load_any
from .val import xywhn_to_xyxy


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/u_nomosaic/best.pt")
    ap.add_argument("--arch", default="yolov8n.yaml")
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.45, help="box NMS, as usual")
    ap.add_argument("--mask-iou", type=float, default=0.7,
                    help="merge two detections whose masks overlap this much")
    ap.add_argument("--topk", type=int, default=20,
                    help="only the most confident k boxes per image get a mask")
    ap.add_argument("--replace-box", action="store_true",
                    help="also replace each box with its mask's bounding box")
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model, norm = load_any(args.weights, arch=args.arch, imgsz=args.imgsz)
    model = model.to(args.device).eval()
    predictor = SAM2ImagePredictor(build_sam2(args.cfg, args.ckpt, device=args.device))

    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=4, augment=False, shuffle=False,
                              workers=2, normalize=norm)

    base = DetMetrics(model.names)
    refined = DetMetrics(model.names)
    n_before = n_after = 0

    for imgs, targets, paths in loader:
        with torch.no_grad():
            preds = model(imgs.to(args.device))
        dets = non_max_suppression(preds, args.conf, args.iou, 300)
        for i, det in enumerate(dets):
            lab = xywhn_to_xyxy(targets[targets[:, 0] == i][:, 1:], args.imgsz)
            base.update(det.cpu(), lab)
            if det.shape[0] == 0:
                refined.update(det.cpu(), lab)
                continue

            det = det.cpu()
            k = min(args.topk, det.shape[0])
            arr = np.asarray(Image.open(paths[i]).convert("RGB")
                             .resize((args.imgsz, args.imgsz))).copy()
            predictor.set_image(arr)
            with torch.inference_mode():
                masks, _, _ = predictor.predict(
                    box=det[:k, :4].numpy(), multimask_output=False)
            masks = np.asarray(masks).reshape(k, args.imgsz, args.imgsz) > 0

            keep, used = [], np.zeros(k, dtype=bool)
            for a in range(k):                      # det is confidence-sorted
                if used[a]:
                    continue
                keep.append(a)
                for b in range(a + 1, k):
                    if used[b] or det[a, 5] != det[b, 5]:
                        continue
                    if mask_iou(masks[a], masks[b]) >= args.mask_iou:
                        used[b] = True
            out = det[keep + list(range(k, det.shape[0]))].clone()
            if args.replace_box:
                for j, a in enumerate(keep):
                    ys, xs = np.nonzero(masks[a])
                    if xs.size:
                        out[j, :4] = torch.tensor(
                            [xs.min(), ys.min(), xs.max(), ys.max()],
                            dtype=out.dtype)
            n_before += k
            n_after += len(keep)
            refined.update(out, lab)

    rb, rr = base.compute(), refined.compute()
    print(f"\n前 {args.topk} 个框中，掩码合并后剩 {n_after}/{n_before} "
          f"({n_after / max(n_before, 1):.1%})")
    print(f"\n{'类别':<18}{'仅框NMS':>10}{'+掩码合并':>12}{'差':>9}")
    for n in model.names:
        a = rb["per_class"].get(n, {}).get("AP50", 0.0)
        b = rr["per_class"].get(n, {}).get("AP50", 0.0)
        print(f"{n:<18}{a:>10.4f}{b:>12.4f}{b - a:>+9.4f}")
    print(f"{'mAP50':<18}{rb['mAP50']:>10.4f}{rr['mAP50']:>12.4f}"
          f"{rr['mAP50'] - rb['mAP50']:>+9.4f}")
    print(f"{'mAP50-95':<18}{rb['mAP50-95']:>10.4f}{rr['mAP50-95']:>12.4f}"
          f"{rr['mAP50-95'] - rb['mAP50-95']:>+9.4f}")


if __name__ == "__main__":
    main()
