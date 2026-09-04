"""Two ways of producing the final boxes, compared side by side.

Method 1 (fine + coarse): SAM2 segments the whole image; each of our detections
    then claims the segments that lie mostly inside it, and the box becomes the
    bounding box of their union.  SAM2 supplies the extent, our detector
    supplies the grouping and the class.
Method 2 (ours alone): the detector's boxes as they are.

Both panels carry the same green ground truth.  --eval scores both over the
whole split instead of drawing.

    python -m model.twomethods --filter rolled-in --n 8
    python -m model.twomethods --eval
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from .dataset import NEU_CLASSES, NEUDetDataset
from .head import non_max_suppression
from .metrics import DetMetrics
from .refarch import load_any
from .val import xywhn_to_xyxy

GT_COLOR = (60, 220, 90)
M1_COLOR = (70, 140, 255)
M2_COLOR = (235, 70, 70)
PAD = 30


def merge_boxes(det, anns, S, seg_in_box, min_segments):
    """Method 1: rebuild each detection's box from the segments it contains."""
    out = det.clone()
    if not anns:
        return out, 0
    segs = [a["segmentation"].astype(bool) for a in anns]
    areas = [max(int(s.sum()), 1) for s in segs]
    n_used = 0
    for j in range(det.shape[0]):
        x1, y1, x2, y2 = (int(round(v)) for v in det[j, :4].tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, S), min(y2, S)
        if x2 <= x1 or y2 <= y1:
            continue
        union = np.zeros((S, S), dtype=bool)
        hit = 0
        for s, a in zip(segs, areas):
            inside = int(s[y1:y2, x1:x2].sum())
            if inside / a >= seg_in_box:
                union |= s
                hit += 1
        if hit < min_segments or not union.any():
            continue                      # keep the detector's box
        ys, xs = np.nonzero(union)
        out[j, :4] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()],
                                  dtype=out.dtype)
        n_used += 1
    return out, n_used


def panel(img, title, S):
    c = Image.new("RGB", (S, S + PAD), (255, 255, 255))
    c.paste(img, (0, PAD))
    ImageDraw.Draw(c).text((6, 9), title, fill=(20, 20, 20))
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--eval", action="store_true",
                    help="score both methods over the whole split, no images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weights", default="runs/u_nomosaic/best.pt")
    ap.add_argument("--arch", default="yolov8n.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--seg-in-box", type=float, default=0.5,
                    help="a segment joins a box if this much of it is inside")
    ap.add_argument("--min-segments", type=int, default=1,
                    help="below this many segments, keep the detector's box")
    # permissive: at the defaults SAM2 returns one whole-image mask, which
    # would give method 1 nothing to merge
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.5)
    ap.add_argument("--stability-thresh", type=float, default=0.7)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--out", default="runs/2methods")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    gen = SAM2AutomaticMaskGenerator(
        build_sam2(args.cfg, args.ckpt, device=args.device),
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_mask_region_area=args.min_area)

    model, norm = load_any(args.weights, arch=args.arch, imgsz=args.imgsz)
    model = model.to(args.device).eval()

    ds = NEUDetDataset(args.data, args.split, imgsz=args.imgsz,
                       augment=False, normalize=norm)
    idxs = list(range(len(ds)))
    if args.filter:
        idxs = [i for i in idxs if args.filter in ds.img_files[i]]
        if not idxs:
            raise SystemExit(f"没有文件名包含 {args.filter!r} 的图片")
    if not args.eval:
        idxs = idxs[:args.n]

    S = args.imgsz
    out_dir = os.path.join(args.out, args.filter or "all")
    if not args.eval:
        os.makedirs(out_dir, exist_ok=True)
    m1 = DetMetrics(model.names)
    m2 = DetMetrics(model.names)
    n_seg, n_rebuilt, n_det = [], 0, []

    # conf 0.001 for a fair AP; the drawing threshold is applied only to images
    conf = 0.001 if args.eval else args.conf

    for i in idxs:
        img_t, lab, path = ds[i]
        base = Image.open(path).convert("RGB").resize((S, S))
        arr = np.asarray(base).copy()

        with torch.no_grad():
            det = non_max_suppression(model(img_t.unsqueeze(0).to(args.device)),
                                      conf, args.iou, args.max_det)[0].cpu()
        with torch.inference_mode():
            anns = gen.generate(arr)
        merged, used = merge_boxes(det, anns, S, args.seg_in_box,
                                   args.min_segments)
        n_seg.append(len(anns))
        n_det.append(det.shape[0])
        n_rebuilt += used

        gt = xywhn_to_xyxy(lab, S)
        m1.update(merged, gt)
        m2.update(det, gt)
        if args.eval:
            continue

        gt_rows = [(b[1], b[2], b[3], b[4], int(b[0])) for b in gt.tolist()]
        ims = []
        for boxes, col, title in (
                (merged, M1_COLOR, f"方法1 SAM2分割+合并: {merged.shape[0]} 框"
                                   f" (重建 {used})"),
                (det, M2_COLOR, f"方法2 仅检测器: {det.shape[0]} 框")):
            im = base.copy()
            d = ImageDraw.Draw(im)
            for b in boxes.tolist():
                d.rectangle(b[:4], outline=col, width=3)
                d.text((b[0] + 4, max(b[1] - 12, 2)),
                       f"{NEU_CLASSES[int(b[5])]} {b[4]:.2f}", fill=col)
            for x1, y1, x2, y2, _ in gt_rows:
                d.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=3)
            ims.append(panel(im, f"{title} / 真值 {len(gt_rows)}", S))
        sheet = Image.new("RGB", (2 * S, ims[0].height), (255, 255, 255))
        sheet.paste(ims[0], (0, 0))
        sheet.paste(ims[1], (S, 0))
        sheet.save(os.path.join(out_dir, os.path.basename(path)), quality=92)

    print(f"\n平均每图  SAM2 掩码 {np.mean(n_seg):.1f}   "
          f"检测框 {np.mean(n_det):.1f}   其中被重建 {n_rebuilt}")
    r1, r2 = m1.compute(), m2.compute()
    print(f"\n{'类别':<18}{'方法1 分割+合并':>16}{'方法2 仅检测器':>16}{'差':>9}")
    for n in model.names:
        a = r1["per_class"].get(n, {}).get("AP50", 0.0)
        b = r2["per_class"].get(n, {}).get("AP50", 0.0)
        print(f"{n:<18}{a:>16.4f}{b:>16.4f}{a - b:>+9.4f}")
    print(f"{'mAP50':<18}{r1['mAP50']:>16.4f}{r2['mAP50']:>16.4f}"
          f"{r1['mAP50'] - r2['mAP50']:>+9.4f}")
    if not args.eval:
        print(f"\n{len(idxs)} 张写入 {out_dir}/  "
              f"(上面的 AP 只基于这几张，看趋势用 --eval 跑全集)")
        print("左=方法1(蓝框)  右=方法2(红框)  绿框=真值，两边相同")


if __name__ == "__main__":
    main()
