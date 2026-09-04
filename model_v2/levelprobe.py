"""Why does one pyramid level never win a positive?

TAL picks the top-k anchors by score^alpha * iou^beta, so a starved level is
losing on one of those two factors.  With the reference network at random init
stride 32 takes 50 of 90 positives on pitted_surface; with ours it takes 0 from
the very first epoch.  This prints both factors per level so the loser is
visible rather than inferred.

    python -m model.levelprobe --filter pitted_surface
"""

import argparse

import torch

from .dataset import build_dataloader
from .head import box_iou
from .yolo_sam2 import build_model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--backbone", default="cnn")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    if args.weights:
        from .yolo_sam2 import YOLOSAM2
        model = YOLOSAM2.load(args.weights, map_location="cpu")
    else:
        model = build_model(None, imgsz=args.imgsz,
                            backbone={"type": args.backbone})
    model = model.to(args.device).train()

    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=args.batch_size, augment=False,
                              shuffle=False, workers=0)
    ds = loader.dataset
    if args.filter:
        keep = [j for j, f in enumerate(ds.img_files) if args.filter in f]
        ds.img_files = [ds.img_files[j] for j in keep]
        ds.lbl_files = [ds.lbl_files[j] for j in keep]
        if ds.labels is not None:
            ds.labels = [ds.labels[j] for j in keep]
        print(f"[filter] {args.filter}: {len(keep)} 张")

    imgs, targets, _ = next(iter(loader))
    imgs = imgs.to(args.device)
    head = model.head
    crit = model.criterion

    with torch.no_grad():
        raw = model(imgs)
        cls, reg, obj, anchors, strides = head.flatten(raw)
        anchors_s = anchors / strides
        gt_labels, gt_bboxes, mask_gt = crit._prepare_targets(
            targets, imgs.shape[0], raw["feats"], args.device, head.strides[0])
        pred_boxes = crit._decode_reg(reg, anchors_s) * strides      # pixels
        scores = cls.sigmoid()

        s_flat = strides.view(-1)
        print(f"\n真值框 {int(mask_gt.sum())} 个, 尺寸 (px):")
        for b in range(gt_bboxes.shape[0]):
            for g in range(gt_bboxes.shape[1]):
                if mask_gt[b, g, 0] > 0:
                    x1, y1, x2, y2 = gt_bboxes[b, g].tolist()
                    print(f"  图{b} 类{int(gt_labels[b, g, 0])} "
                          f"{x2 - x1:.0f} x {y2 - y1:.0f}")

        print(f"\n{'层级':<11}{'anchor数':>9}{'预测框中位边长':>15}"
              f"{'框内anchor':>11}{'最大IoU':>9}{'最大分数':>9}{'最大对齐度':>12}")
        for s in sorted({int(v) for v in s_flat.tolist()}):
            m = s_flat == s
            pb = pred_boxes[0][m]
            side = ((pb[:, 2] - pb[:, 0]) + (pb[:, 3] - pb[:, 1])) / 2
            best_iou = best_align = 0.0
            n_in = 0
            for b in range(gt_bboxes.shape[0]):
                for g in range(gt_bboxes.shape[1]):
                    if mask_gt[b, g, 0] <= 0:
                        continue
                    gt = gt_bboxes[b, g:g + 1]
                    a = anchors[m]
                    inside = ((a[:, 0] > gt[0, 0]) & (a[:, 0] < gt[0, 2]) &
                              (a[:, 1] > gt[0, 1]) & (a[:, 1] < gt[0, 3]))
                    n_in = max(n_in, int(inside.sum()))
                    if not inside.any():
                        continue
                    iou = box_iou(gt, pred_boxes[b][m][inside]).squeeze(0)
                    sc = scores[b][m][inside][:, int(gt_labels[b, g, 0])]
                    align = sc.pow(0.5) * iou.clamp(0).pow(6.0)
                    best_iou = max(best_iou, float(iou.max()))
                    best_align = max(best_align, float(align.max()))
            sc_max = float(scores[0][m].max())
            print(f"stride {s:<4}{int(m.sum()):>9}{float(side.median()):>15.0f}"
                  f"{n_in:>11}{best_iou:>9.3f}{sc_max:>9.4f}{best_align:>12.3e}")
    print("\n对齐度量 = 分数^0.5 * IoU^6，最大对齐度最高的层级拿走 top-k 正样本")


if __name__ == "__main__":
    main()
