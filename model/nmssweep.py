"""How much AP is recoverable by suppressing duplicates alone?

Diagnosis: on crazing both our model and the reference emit ~5x more confident
boxes than there are ground truths, while localising them well (IoU>=.5 on 86%)
and classifying them perfectly.  Before building anything to fix that, this
measures the ceiling of the cheapest possible fix -- a stricter NMS, and a cap
on detections per image -- per class, on weights that already exist.  If no
setting moves crazing, duplicate suppression is not the lever.

    python -m model.nmssweep --weights runs/u_nomosaic/best.pt
"""

import argparse

import torch

from .dataset import build_dataloader
from .head import non_max_suppression
from .metrics import DetMetrics
from .val import xywhn_to_xyxy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/u_nomosaic/best.pt")
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--arch", default="yolov8n.yaml",
                    help="network yaml, only for raw ultralytics checkpoints")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--ious", default="0.2,0.3,0.45,0.6,0.7")
    ap.add_argument("--max-dets", default="300,20,10,5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    normalize = True
    if ck.get("ref_arch"):
        from .refarch import RefArchModel
        model = RefArchModel.load(args.weights, map_location="cpu")
    elif "model" in ck and hasattr(ck["model"], "state_dict"):
        # raw ultralytics checkpoint; those weights expect plain /255 inputs
        from .refarch import RefArchModel
        model = RefArchModel({"nc": 6, "imgsz": args.imgsz, "arch": args.arch})
        model.det.load_state_dict(
            {k: v.float() for k, v in ck["model"].state_dict().items()},
            strict=False)
        normalize = False
        print(f"[weights] ultralytics {args.arch}，已关闭 ImageNet 归一化")
    else:
        from .yolo_sam2 import YOLOSAM2
        model = YOLOSAM2.load(args.weights, map_location="cpu")
    model = model.to(args.device).eval()

    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=args.batch_size, augment=False,
                              shuffle=False, workers=2, normalize=normalize)

    cache = []                      # raw predictions; the sweep is post-processing
    with torch.no_grad():
        for imgs, targets, _ in loader:
            cache.append((model(imgs.to(args.device)).cpu(), targets))

    names = model.names

    def run(iou, cap):
        m = DetMetrics(names)
        for preds, targets in cache:
            dets = non_max_suppression(preds, args.conf, iou, cap)
            for i in range(preds.shape[0]):
                lab = targets[targets[:, 0] == i][:, 1:]
                m.update(dets[i], xywhn_to_xyxy(lab, args.imgsz))
        return m.compute()

    print(f"{'IoU':>6}{'max_det':>8}{'mAP50':>9}" +
          "".join(f"{n[:10]:>11}" for n in names))
    for cap in (int(x) for x in args.max_dets.split(",")):
        for iou in (float(x) for x in args.ious.split(",")):
            r = run(iou, cap)
            print(f"{iou:>6.2f}{cap:>8}{r['mAP50']:>9.4f}" +
                  "".join(f"{r['per_class'][n]['AP50']:>11.4f}" for n in names))
    print("\n若 crazing / rolled-in 在任何组合下都不上升 -> 重复抑制不是杠杆")


if __name__ == "__main__":
    main()
