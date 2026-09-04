"""Feed one set of statistics to both AP implementations.

predcheck showed that ultralytics' own detections, scored by our DetMetrics,
give 0.359 where ultralytics' validator gives 0.672 on the same weights and
the same 854 ground-truth boxes.  Detections are therefore correct and the
defect is in our metric.  This splits the metric in half: the true-positive
matching, and the AP computation on top of it.

    python -m model.metriccheck --weights runs/ultra_nomosaic/weights/best.pt
"""

import argparse

import numpy as np
import torch

from .dataset import build_dataloader
from .metrics import DetMetrics
from .refarch import RefArchModel
from .val import xywhn_to_xyxy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/ultra_nomosaic/weights/best.pt")
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    from ultralytics.utils.metrics import ap_per_class
    try:
        from ultralytics.utils.nms import non_max_suppression as ref_nms
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression as ref_nms

    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    src = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = RefArchModel({"nc": args.nc, "imgsz": args.imgsz}).to(args.device)
    model.det.load_state_dict({k: v.float() for k, v in src.state_dict().items()},
                              strict=False)
    model.eval()

    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=args.batch_size, augment=False,
                              shuffle=False, workers=2, normalize=False)

    metrics = DetMetrics(model.names)
    with torch.no_grad():
        for imgs, targets, _ in loader:
            imgs = imgs.to(args.device)
            y, _ = model.det(imgs)
            dets = ref_nms(y, args.conf, args.iou, nc=0, multi_label=True, max_det=300)
            for i in range(imgs.shape[0]):
                lab = xywhn_to_xyxy(targets[targets[:, 0] == i][:, 1:].to(args.device),
                                    args.imgsz)
                metrics.update(dets[i], lab)

    tp = np.concatenate([s[0] for s in metrics.stats], 0)
    conf = np.concatenate([s[1] for s in metrics.stats], 0)
    pred_cls = np.concatenate([s[2] for s in metrics.stats], 0)
    target_cls = np.concatenate([s[3] for s in metrics.stats], 0)
    print(f"[stats] 检测框 {tp.shape[0]}  真值 {target_cls.shape[0]}  "
          f"命中率 {tp[:, 0].mean():.4f}")

    ours = metrics.compute()

    out = ap_per_class(tp, conf, pred_cls, target_cls)
    ref_ap = next(v for v in out
                  if isinstance(v, np.ndarray) and v.ndim == 2
                  and v.shape[1] == tp.shape[1])

    print(f"\n{'AP 实现':<24}{'mAP50':>10}{'mAP50-95':>12}")
    print(f"{'我们的 compute()':<24}{ours['mAP50']:>10.4f}{ours['mAP50-95']:>12.4f}")
    print(f"{'ultralytics ap_per_class':<24}{ref_ap[:, 0].mean():>10.4f}"
          f"{ref_ap.mean():>12.4f}")
    print("\n两者相同 -> bug 在 match_predictions（tp 矩阵算错）;"
          "\n两者不同 -> bug 在 compute()（AP 汇总算错）")


if __name__ == "__main__":
    main()
