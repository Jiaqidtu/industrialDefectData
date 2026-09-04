"""Draw ground truth and predictions, and dump augmented training samples.

Two modes:

``--weights W``  draw GT (green) vs predictions (red) on validation images
``--aug``        draw the *training* samples exactly as the model sees them,
                 mosaic included -- the fastest way to catch a broken label
                 transform, which no loss curve will ever tell you about

    python3 -m model.visualize --aug --out runs/viz_aug
    python3 -m model.visualize --weights runs/f_cnn_baseline/best.pt --out runs/viz
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from .dataset import SAM2_MEAN, SAM2_STD, NEUDetDataset
from .head import non_max_suppression

COLORS = [(0, 200, 0), (220, 40, 40)]     # gt, prediction


def denormalize(img: torch.Tensor) -> Image.Image:
    mean = torch.tensor(SAM2_MEAN).view(3, 1, 1)
    std = torch.tensor(SAM2_STD).view(3, 1, 1)
    arr = (img.cpu() * std + mean).clamp(0, 1).mul(255).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy())


def draw(im: Image.Image, boxes, names, color, tag=""):
    d = ImageDraw.Draw(im)
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b[:4]]
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = names[int(b[4])] if len(b) > 4 else ""
        if len(b) > 5:
            label += f" {float(b[5]):.2f}"
        d.text((x1 + 3, max(y1 - 12, 2)), f"{tag}{label}", fill=color)
    return im


def main(argv=None):
    ap = argparse.ArgumentParser("visualise labels and predictions")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--filter", default=None,
                    help="only images whose filename contains this")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--aug", action="store_true",
                    help="dump augmented training samples (mosaic included)")
    ap.add_argument("--out", default="runs/viz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    split = args.split or ("train" if args.aug else "validation")
    ds = NEUDetDataset(args.data, split, imgsz=args.imgsz, augment=args.aug,
                       mosaic=1.0 if args.aug else 0.0,
                       scale=0.5 if args.aug else 0.0,
                       translate=0.1 if args.aug else 0.0)
    if args.filter:
        keep = [j for j, f in enumerate(ds.img_files) if args.filter in f]
        if not keep:
            raise SystemExit(f"没有文件名包含 {args.filter!r} 的图片")
        ds.img_files = [ds.img_files[j] for j in keep]
        ds.lbl_files = [ds.lbl_files[j] for j in keep]
        if ds.labels is not None:
            ds.labels = [ds.labels[j] for j in keep]
        print(f"[filter] {args.filter}: {len(keep)} 张")
    os.makedirs(args.out, exist_ok=True)

    model = None
    if args.weights:
        from .yolo_sam2 import YOLOSAM2
        model = YOLOSAM2.load(args.weights, map_location="cpu").to(args.device).eval()
    names = model.names if model else ds.classes

    idxs = np.linspace(0, len(ds) - 1, args.n).astype(int)
    for k, i in enumerate(idxs):
        img, lab, path = ds[int(i)]
        im = denormalize(img)
        S = args.imgsz
        gt = []
        for l in lab.numpy():
            cx, cy, w, h = l[1] * S, l[2] * S, l[3] * S, l[4] * S
            gt.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, l[0]])
        draw(im, gt, names, COLORS[0], tag="GT:")

        if model is not None:
            with torch.no_grad():
                pred = model(img.unsqueeze(0).to(args.device))
            det = non_max_suppression(pred, conf_thres=args.conf, iou_thres=0.65)[0]
            boxes = [[*d[:4].tolist(), d[5].item(), d[4].item()] for d in det.cpu()]
            draw(im, boxes, names, COLORS[1])

        base = os.path.splitext(os.path.basename(path))[0]
        im.save(os.path.join(args.out, f"{k:02d}_{base}.png"))
    print(f"[viz] {len(idxs)} 张图写到 {args.out}/  "
          f"(绿=真值{'，红=预测' if model else ''})")
    if args.aug:
        print("     重点看：mosaic 拼接后框有没有对准缺陷、有没有错位或越界")


if __name__ == "__main__":
    main()
