"""What does a predictor that ignores the image score?

For crazing a typical annotation is 2 boxes covering 52% of the frame, and the
count varies from 1 to 5 across otherwise similar images.  When the boxes cover
most of the image their coordinates carry little information, and detection AP
starts measuring "is this class present" rather than "where is it".  This puts
a floor under that intuition: predict a fixed layout -- one full-frame box, or
a k-way tiling -- for every image of the class, with no model at all.

    python -m model.trivial
"""

import argparse

import torch

from .dataset import build_dataloader
from .metrics import DetMetrics
from .val import xywhn_to_xyxy


def layouts(S: int):
    """Fixed box sets, in xyxy pixels."""
    full = [(0, 0, S, S)]
    h2 = [(0, 0, S, S / 2), (0, S / 2, S, S)]
    v2 = [(0, 0, S / 2, S), (S / 2, 0, S, S)]
    q4 = [(0, 0, S / 2, S / 2), (S / 2, 0, S, S / 2),
          (0, S / 2, S / 2, S), (S / 2, S / 2, S, S)]
    inset = [(S * 0.1, S * 0.1, S * 0.9, S * 0.9)]
    return {"整幅图 1框": full, "上下 2框": h2, "左右 2框": v2,
            "四宫格 4框": q4, "内缩 1框": inset}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--oracle-presence", action="store_true", default=True,
                    help="only predict a class on images that contain it. "
                         "Without this the floor is set by class presence "
                         "(each class appears in 60 of 360 images, capping "
                         "precision at 0.167) rather than by localisation.")
    ap.add_argument("--all-classes", dest="oracle_presence",
                    action="store_false")
    args = ap.parse_args(argv)

    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=16, augment=False, shuffle=False,
                              workers=0)
    names = loader.dataset.classes if hasattr(loader.dataset, "classes") else None
    from .dataset import NEU_CLASSES
    names = list(names or NEU_CLASSES)

    batches = [(t.clone(), i.shape[0]) for i, t, _ in loader]

    print(f"神谕类别存在性 = {args.oracle_presence}"
          f"（关闭则精度上限被 60/360 的类别先验钉死）\n")
    print(f"{'固定布局':<14}{'mAP50':>9}" + "".join(f"{n[:10]:>11}" for n in names))
    for label, boxes in layouts(args.imgsz).items():
        m = DetMetrics(names)
        for targets, bs in batches:
            for i in range(bs):
                lab = xywhn_to_xyxy(targets[targets[:, 0] == i][:, 1:], args.imgsz)
                if lab.shape[0] == 0:
                    continue
                # every class predicted at every fixed location, score 1.0;
                # AP is computed per class so the classes do not interfere
                present = (sorted({int(v) for v in lab[:, 0].tolist()})
                           if args.oracle_presence else range(len(names)))
                det = torch.tensor(
                    [[x1, y1, x2, y2, 1.0, c]
                     for (x1, y1, x2, y2) in boxes for c in present],
                    dtype=torch.float32)
                m.update(det, lab)
        r = m.compute()
        print(f"{label:<14}{r['mAP50']:>9.4f}" +
              "".join(f"{r['per_class'][n]['AP50']:>11.4f}"
                      if n in r["per_class"] else f"{'-':>11}" for n in names))
    print("\n某类在此处的分数越接近训练出来的模型，说明该类的框坐标越不携带信息")


if __name__ == "__main__":
    main()
