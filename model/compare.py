"""Side-by-side prediction sheets for eyeballing several models at once.

One panel per model on the same image, green = ground truth, red = prediction.
Written because the numbers have stopped separating the models (everything sits
at 0.72-0.74 overall, and crazing/rolled-in_scale are stuck for all of them) --
the remaining questions are about *what kind* of mistake each model makes, and
that is faster to see than to measure.

    python -m model.compare --filter rolled-in --n 12
    python -m model.compare --runs "ours=runs/u_nomosaic/best.pt" --n 8
"""

import argparse
import os

import torch
from PIL import Image, ImageDraw

from .dataset import NEU_CLASSES, NEUDetDataset
from .head import non_max_suppression
from .refarch import load_any

GT_COLOR = (60, 200, 90)
PRED_COLOR = (235, 70, 70)
PANEL_PAD = 28

DEFAULT_RUNS = [
    ("ours-CNN", "runs/u_nomosaic/best.pt", "yolov8n.yaml"),
    ("ours-SAM2", "runs/v2_ada_none/best.pt", "yolov8n.yaml"),
    ("YOLOv8n", "runs/ultra_scratch/weights/best.pt", "yolov8n.yaml"),
    ("YOLOv8s", "runs/ref_v8s/weights/best.pt", "yolov8s.yaml"),
]


def draw_boxes(im, boxes, names, color, with_conf=False):
    d = ImageDraw.Draw(im)
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if with_conf:
            label = f"{names[int(b[5])]} {float(b[4]):.2f}"
        else:
            label = names[int(b[4])] if len(b) > 4 else ""
        if label:
            d.text((x1 + 4, max(y1 - 12, 2)), label, fill=color)
    return im


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--filter", default=None,
                    help="only images whose filename contains this")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="display threshold; 0.001 is right for AP but "
                         "unreadable on a sheet")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", default=None,
                    help="name=path[:arch] pairs, comma separated; "
                         "default compares our two against YOLOv8n/s")
    ap.add_argument("--out", default="runs/compare")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    if args.runs:
        runs = []
        for item in args.runs.split(","):
            name, _, rest = item.partition("=")
            path, _, arch = rest.partition(":")
            runs.append((name, path, arch or "yolov8n.yaml"))
    else:
        runs = DEFAULT_RUNS
    runs = [(n, p, a) for n, p, a in runs if os.path.exists(p)]
    if not runs:
        raise SystemExit("没有可用的权重文件")
    print("[models] " + ", ".join(n for n, _, _ in runs))

    models = []
    for name, path, arch in runs:
        m, norm = load_any(path, arch=arch, imgsz=args.imgsz)
        models.append((name, m.to(args.device).eval(), norm))
        print(f"  {name:<12}{path}  ImageNet归一化={norm}")

    # two datasets: one normalised, one not, so each model gets its own input
    ds_norm = NEUDetDataset(args.data, args.split, imgsz=args.imgsz,
                            augment=False, normalize=True)
    ds_raw = NEUDetDataset(args.data, args.split, imgsz=args.imgsz,
                           augment=False, normalize=False)
    idxs = list(range(len(ds_norm)))
    if args.filter:
        idxs = [i for i in idxs if args.filter in ds_norm.img_files[i]]
        if not idxs:
            raise SystemExit(f"没有文件名包含 {args.filter!r} 的图片")
    idxs = idxs[:args.n]

    out_dir = os.path.join(args.out, args.filter or "all")
    os.makedirs(out_dir, exist_ok=True)

    for i in idxs:
        img_norm, lab, path = ds_norm[i]
        img_raw, _, _ = ds_raw[i]
        base = Image.open(path).convert("RGB").resize((args.imgsz, args.imgsz))

        gt = []
        for row in lab.tolist():
            c, cx, cy, w, h = row
            gt.append([(cx - w / 2) * args.imgsz, (cy - h / 2) * args.imgsz,
                       (cx + w / 2) * args.imgsz, (cy + h / 2) * args.imgsz, c])

        panels = []
        for name, model, norm in models:
            x = (img_norm if norm else img_raw).unsqueeze(0).to(args.device)
            with torch.no_grad():
                det = non_max_suppression(model(x), args.conf, args.iou,
                                          args.max_det)[0].cpu()
            p = base.copy()
            draw_boxes(p, gt, NEU_CLASSES, GT_COLOR)
            draw_boxes(p, det.tolist(), NEU_CLASSES, PRED_COLOR, with_conf=True)
            canvas = Image.new("RGB", (args.imgsz, args.imgsz + PANEL_PAD),
                               (255, 255, 255))
            canvas.paste(p, (0, PANEL_PAD))
            ImageDraw.Draw(canvas).text(
                (6, 8), f"{name}   预测 {det.shape[0]} / 真值 {len(gt)}",
                fill=(20, 20, 20))
            panels.append(canvas)

        w = sum(p.width for p in panels)
        sheet = Image.new("RGB", (w, panels[0].height), (255, 255, 255))
        for k, p in enumerate(panels):
            sheet.paste(p, (k * args.imgsz, 0))
        name = os.path.splitext(os.path.basename(path))[0]
        sheet.save(os.path.join(out_dir, f"{name}.jpg"), quality=92)

    print(f"\n{len(idxs)} 张已写入 {out_dir}/")
    print("绿框 = 真值，红框 = 预测（带置信度）。每张图横向并排各模型。")


if __name__ == "__main__":
    main()
