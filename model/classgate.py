"""How much AP is lost to detections of classes the image does not contain?

93.2% of NEU-DET images hold exactly one defect class, yet the detector scores
every class on every image and AP is pooled across all 360.  A confident
inclusion box on a crazing image cannot be right, but it still lands in
inclusion's precision curve.

This measures the ceiling of removing them, using the ground-truth image-level
labels as an oracle gate.  It is an upper bound, not a method: a real system
would need a classifier, though image-level classification on NEU-DET is close
to solved.

    python -m model.classgate --weights runs/u_nomosaic/best.pt
"""

import argparse

import torch

from .dataset import build_dataloader
from .head import non_max_suppression
from .metrics import DetMetrics
from .refarch import load_any
from .val import xywhn_to_xyxy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/u_nomosaic/best.pt")
    ap.add_argument("--arch", default="yolov8n.yaml")
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    model, norm = load_any(args.weights, arch=args.arch, imgsz=args.imgsz)
    model = model.to(args.device).eval()
    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=8, augment=False, shuffle=False,
                              workers=2, normalize=norm)

    plain, gated = DetMetrics(model.names), DetMetrics(model.names)
    kept = dropped = 0
    for imgs, targets, _ in loader:
        with torch.no_grad():
            preds = model(imgs.to(args.device))
        dets = non_max_suppression(preds, args.conf, args.iou, 300)
        for i, det in enumerate(dets):
            lab = targets[targets[:, 0] == i][:, 1:]
            gt = xywhn_to_xyxy(lab, args.imgsz)
            det = det.cpu()
            plain.update(det, gt)
            present = {int(c) for c in lab[:, 0].tolist()}
            if det.shape[0]:
                keep = torch.tensor([int(c) in present for c in det[:, 5].tolist()])
                kept += int(keep.sum())
                dropped += int((~keep).sum())
                det = det[keep]
            gated.update(det, gt)

    rp, rg = plain.compute(), gated.compute()
    print(f"\n丢弃 {dropped} 个跨类检测，保留 {kept} 个 "
          f"({dropped / max(kept + dropped, 1):.1%} 被丢弃)")
    print(f"\n{'类别':<18}{'原始':>10}{'神谕类别门控':>14}{'差':>9}")
    for n in model.names:
        a = rp["per_class"].get(n, {}).get("AP50", 0.0)
        b = rg["per_class"].get(n, {}).get("AP50", 0.0)
        print(f"{n:<18}{a:>10.4f}{b:>14.4f}{b - a:>+9.4f}")
    print(f"{'mAP50':<18}{rp['mAP50']:>10.4f}{rg['mAP50']:>14.4f}"
          f"{rg['mAP50'] - rp['mAP50']:>+9.4f}")
    print(f"{'mAP50-95':<18}{rp['mAP50-95']:>10.4f}{rg['mAP50-95']:>14.4f}"
          f"{rg['mAP50-95'] - rp['mAP50-95']:>+9.4f}")
    print("\n这是上界：真实系统要用分类器代替神谕，且 6.8% 的图含多个类别")


if __name__ == "__main__":
    main()
