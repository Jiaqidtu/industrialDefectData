"""Does SAM2 produce usable masks on these defects at all?

The proposal is to segment the defect and derive one box from the segmentation,
which would make the extent deterministic and remove the "how many boxes"
ambiguity.  That rests on an assumption we have never tested: SAM2 is trained
on objects with boundaries, while crazing and rolled-in_scale are boundaryless
textures.  If its masks on them are fragments -- or the whole steel plate --
the idea fails before any of it is built.

Renders, per image: the ground-truth box (green), the mask SAM2 returns when
prompted with that box, and the box that mask implies (blue).  A mask that
tracks the defect gives a blue box close to the green one.

    python -m model.sam2probe --filter rolled-in --n 8
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from .dataset import NEU_CLASSES, NEUDetDataset

GT_COLOR = (60, 200, 90)
MASK_BOX_COLOR = (70, 130, 240)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--prompt", default="auto", choices=["auto", "box"],
                    help="auto = no prompt at all (does SAM2 find the defect on "
                         "its own); box = prompt with the GT box, which mostly "
                         "asks whether SAM2 obeys the prompt")
    ap.add_argument("--points-per-side", type=int, default=16)
    # The generator's defaults (pred_iou 0.8, stability 0.95, min area 100) are
    # tuned for natural images and discard weak masks.  scratches produced zero
    # masks, which could be filtering rather than blindness -- these knobs tell
    # the two apart.
    ap.add_argument("--pred-iou-thresh", type=float, default=0.8)
    ap.add_argument("--stability-thresh", type=float, default=0.95)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--out", default="runs/viz_sam2",
                    help="one subdirectory per class filter")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    try:
        from sam2.build_sam import build_sam2
        if args.prompt == "auto":
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        else:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise SystemExit(
            f"需要完整的 sam2 包（我们只实现了 Hiera 主干）：{e}\n"
            f"安装：pip install --no-cache-dir -e ~/sam2")

    sam = build_sam2(args.cfg, args.ckpt, device=args.device)
    if args.prompt == "auto":
        engine = SAM2AutomaticMaskGenerator(
            sam, points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_thresh,
            min_mask_region_area=args.min_area)
    else:
        engine = SAM2ImagePredictor(sam)

    ds = NEUDetDataset(args.data, args.split, imgsz=args.imgsz,
                       augment=False, normalize=False)
    idxs = list(range(len(ds)))
    if args.filter:
        idxs = [i for i in idxs if args.filter in ds.img_files[i]]
        if not idxs:
            raise SystemExit(f"没有文件名包含 {args.filter!r} 的图片")
    idxs = idxs[:args.n]

    out_dir = os.path.join(args.out, args.filter or "all")
    os.makedirs(out_dir, exist_ok=True)
    S = args.imgsz
    stats, fills, rects = [], [], []

    for i in idxs:
        _, lab, path = ds[i]
        im = Image.open(path).convert("RGB").resize((S, S))
        arr = np.asarray(im)
        if args.prompt == "box":
            engine.set_image(arr)      # the auto generator does this itself

        overlay = arr.astype(np.float32).copy()
        boxes_gt = []
        gt_union = np.zeros((S, S), dtype=bool)
        for row in lab.tolist():
            c, cx, cy, w, h = row
            box = np.array([(cx - w / 2) * S, (cy - h / 2) * S,
                            (cx + w / 2) * S, (cy + h / 2) * S])
            boxes_gt.append((box, int(c)))
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            gt_union[max(y1, 0):y2, max(x1, 0):x2] = True

        if args.prompt == "auto":
            with torch.inference_mode():
                anns = engine.generate(arr)
            masks = [a["segmentation"].astype(bool) for a in anns]
        else:
            masks = []
            for box, _ in boxes_gt:
                with torch.inference_mode():
                    m, _, _ = engine.predict(box=box[None, :],
                                             multimask_output=False)
                m = m[0].astype(bool)
                masks.append(m)
                # A box-prompted mask that merely fills the prompt looks like a
                # perfect segmentation but carries no information the box did
                # not already have.  fill = mask/box area, rect = how close the
                # mask is to being a rectangle.  Both near 1.0 => it copied the
                # prompt; fill well under 1 with low rect => it traced a shape.
                ba = max((box[2]-box[0]) * (box[3]-box[1]), 1.0)
                a = int(m.sum())
                if a:
                    ys, xs = np.nonzero(m)
                    bb = max((xs.max()-xs.min()+1) * (ys.max()-ys.min()+1), 1)
                    fills.append(a / ba)
                    rects.append(a / bb)

        # containment = 掩码落在真值框内的比例
        contain = []
        palette = [(255, 170, 60), (80, 200, 255), (255, 110, 200),
                   (160, 255, 120), (255, 240, 100)]
        for j, m in enumerate(masks):
            a = int(m.sum())
            if a == 0:
                continue
            contain.append(float((m & gt_union).sum()) / a)
            col = np.array(palette[j % len(palette)], dtype=np.float32)
            overlay[m] = overlay[m] * 0.6 + col * 0.4
        gt_cov = (float((gt_union & np.any(masks, 0)).sum()) / max(int(gt_union.sum()), 1)
                  if masks else 0.0)
        med = float(np.median(contain)) if contain else 0.0
        # A mask dropped at random lands inside the boxes with probability equal
        # to their area fraction -- 0.89 of the frame for pitted_surface, 0.52
        # for crazing.  Without dividing that out, "85% of the mask is inside
        # the box" can be worse than chance.
        chance = float(gt_union.sum()) / (S * S)
        lift = med / chance if chance > 0 else 0.0
        good = sum(c > 0.8 for c in contain)
        stats.append((med, gt_cov, len(contain), good, chance, lift))
        print(f"  {os.path.basename(path):<24} 掩码{len(contain):>3}个  "
              f"落框内 {med:.2f}  随机基线 {chance:.2f}  提升 {lift:>4.2f}x  "
              f"真值被覆盖 {gt_cov:.2f}")

        vis = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(vis)
        for box, c in boxes_gt:
            d.rectangle(list(box), outline=GT_COLOR, width=3)
            d.text((box[0] + 4, max(box[1] - 12, 2)), NEU_CLASSES[c], fill=GT_COLOR)
        vis.save(os.path.join(out_dir, os.path.basename(path)), quality=92)

    if stats:
        med = np.median([x[0] for x in stats])
        cov = np.median([x[1] for x in stats])
        chance = np.median([x[4] for x in stats])
        lift = np.median([x[5] for x in stats])
        if fills:
            print(f"\n[提示框模式] 掩码面积/提示框面积 中位 {np.median(fills):.3f}"
                  f"   矩形度 中位 {np.median(rects):.3f}")
            print("  两者都接近 1.0 -> SAM2 只是把框填满，没有带来新信息")
            print("  面积比明显 <1 且矩形度低 -> 它确实沿缺陷形状勾了边")
        print(f"\n提示方式 = {args.prompt}   共 {len(stats)} 张")
        print(f"  掩码落在真值框内   中位 {med:.3f}")
        print(f"  随机基线(框面积占比) {chance:.3f}")
        print(f"  提升               {lift:.2f}x      <- 这个才是准确率的意义所在")
        print(f"  真值区域被掩码覆盖  中位 {cov:.3f}")
        print(f"  平均每图掩码数      {np.mean([x[2] for x in stats]):.1f}")
        print(f"\n提升 ~1.0 = 和随机撒掩码没区别；>1.5 = SAM2 确实偏好缺陷区域")
    print(f"{len(idxs)} 张写入 {out_dir}/")
    print("绿框=真值，彩色区域=SAM2 掩码（每个掩码一种颜色）")
    print("落在框内比例高且掩码数量少 -> SAM2 自己就能圈出缺陷；"
          "比例低或掩码碎成几十片 -> 它切的是别的东西")


if __name__ == "__main__":
    main()
