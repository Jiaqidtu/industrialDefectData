"""Is a SAM2 mask's *quality* a usable signal for rescoring detections?

maskrefine failed because it used the mask's extent, which is a function of the
prompt box -- circular.  The mask's quality is not: a box that really contains a
defect gives SAM2 something to trace (measured fill ratio ~0.62, rectangularity
~0.69), while a box on clean plate has no structure to find and the mask should
either fill the box or collapse.

This does not build a rescorer.  It measures whether true and false positives
separate at all, by AUC, before anything is built.

    python -m model.maskscore --weights runs/u_nomosaic/best.pt
"""

import argparse
import os

import numpy as np
import torch

from .dataset import build_dataloader
from .head import box_iou, non_max_suppression
from .metrics import DetMetrics
from .refarch import load_any
from .val import xywhn_to_xyxy


def auc(pos, neg):
    """Probability a random positive scores above a random negative."""
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    r = ranks[:len(pos)].sum()
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/u_nomosaic/best.pt")
    ap.add_argument("--arch", default="yolov8n.yaml")
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05,
                    help="only boxes a user would see are worth rescoring")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--topk", type=int, default=20)
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

    feats = {k: {"tp": [], "fp": []} for k in
             ("检测器置信度", "SAM2质量分", "掩码/框面积", "矩形度", "掩码碎片度")}
    S = args.imgsz

    for imgs, targets, paths in loader:
        with torch.no_grad():
            preds = model(imgs.to(args.device))
        dets = non_max_suppression(preds, args.conf, args.iou, 300)
        for i, det in enumerate(dets):
            if det.shape[0] == 0:
                continue
            det = det.cpu()[:args.topk]
            lab = xywhn_to_xyxy(targets[targets[:, 0] == i][:, 1:], args.imgsz)
            # a detection is a true positive if it matches a same-class GT
            is_tp = np.zeros(det.shape[0], dtype=bool)
            if lab.shape[0]:
                iou = box_iou(lab[:, 1:], det[:, :4]).numpy()
                same = (lab[:, :1].numpy() == det[:, 5].numpy()[None, :])
                is_tp = ((iou * same) >= 0.5).any(0)

            arr = np.asarray(Image.open(paths[i]).convert("RGB")
                             .resize((S, S))).copy()
            predictor.set_image(arr)
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    box=det[:, :4].numpy(), multimask_output=False)
            masks = np.asarray(masks).reshape(det.shape[0], S, S) > 0
            scores = np.asarray(scores).reshape(-1)

            for j in range(det.shape[0]):
                m = masks[j]
                a = int(m.sum())
                x1, y1, x2, y2 = det[j, :4].tolist()
                ba = max((x2 - x1) * (y2 - y1), 1.0)
                if a:
                    ys, xs = np.nonzero(m)
                    bb = max((xs.max()-xs.min()+1) * (ys.max()-ys.min()+1), 1)
                    rect, frag = a / bb, 1.0 - a / bb
                else:
                    rect, frag = 0.0, 1.0
                key = "tp" if is_tp[j] else "fp"
                feats["检测器置信度"][key].append(float(det[j, 4]))
                feats["SAM2质量分"][key].append(float(scores[j]))
                feats["掩码/框面积"][key].append(a / ba)
                feats["矩形度"][key].append(rect)
                feats["掩码碎片度"][key].append(frag)

    n_tp = len(feats["检测器置信度"]["tp"])
    n_fp = len(feats["检测器置信度"]["fp"])
    print(f"\n真阳性 {n_tp}   假阳性 {n_fp}   (IoU>=0.5 且类别正确记为真阳性)")
    print(f"\n{'特征':<16}{'真阳性均值':>12}{'假阳性均值':>12}{'AUC':>9}")
    for k, v in feats.items():
        pos, neg = np.array(v["tp"]), np.array(v["fp"])
        a = auc(pos, neg)
        flag = "  <- 有区分力" if not np.isnan(a) and abs(a - 0.5) > 0.08 else ""
        print(f"{k:<16}{pos.mean():>12.3f}{neg.mean():>12.3f}{a:>9.3f}{flag}")
    print("\nAUC 0.5 = 毫无区分力；检测器置信度那一行是参照，"
          "掩码特征要明显超过它才值得做重打分")


if __name__ == "__main__":
    main()
