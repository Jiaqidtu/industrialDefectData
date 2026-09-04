"""Score an ultralytics checkpoint with OUR evaluation pipeline.

The mirror image of the network bisection.  ultralytics reports 0.6725 for
runs/ultra_nomosaic; if our decode / NMS / mAP code reports something far
lower for the very same weights, the defect is in the evaluation path rather
than in training -- which would explain why our training losses now match the
reference while our validation mAP does not.

    python -m model.evalcheck --weights runs/ultra_nomosaic/weights/best.pt
"""

import argparse
import os

import torch

from .dataset import build_dataloader
from .refarch import RefArchModel
from .val import evaluate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/ultra_nomosaic/weights/best.pt")
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--arch", default="yolov8n.yaml")
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--no-multi-label", action="store_true")
    ap.add_argument("--json", default=None, help="write the metrics here")
    ap.add_argument("--normalize", action="store_true", default=False,
                    help="ultralytics trains on plain /255 images, so our "
                         "ImageNet normalisation must be OFF for its weights")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    src = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = src.state_dict() if hasattr(src, "state_dict") else src
    sd = {k: v.float() for k, v in sd.items()}

    model = RefArchModel({"nc": args.nc, "imgsz": args.imgsz, "arch": args.arch})
    missing, unexpected = model.det.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "num_batches_tracked" not in k]
    unexpected = [k for k in unexpected if "num_batches_tracked" not in k]
    print(f"[weights] 未匹配 {len(missing)}  多余 {len(unexpected)}")
    if missing[:3] or unexpected[:3]:
        print(f"          示例 missing={missing[:3]} unexpected={unexpected[:3]}")

    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=args.batch_size, augment=False,
                              shuffle=False, workers=2, normalize=args.normalize)
    print(f"[data] {len(loader.dataset)} 张, ImageNet 归一化 = {args.normalize}")
    res = evaluate(model, loader, device=args.device, conf_thres=args.conf,
                   iou_thres=args.iou, multi_label=not args.no_multi_label)
    if args.json:
        import json as _json
        d = os.path.dirname(os.path.abspath(args.json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.json, "w") as fh:
            _json.dump(res, fh, indent=2)
    print("\nultralytics 自己报告 0.6725。若这里接近 -> 我们的评估没问题；"
          "若远低 -> bug 在解码/NMS/指标")


if __name__ == "__main__":
    main()
