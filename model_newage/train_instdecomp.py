"""Train the instance-decomposition head as a crazing specialist.

crazing loses its AP to decomposition, not to features: classification is
perfect, 86% of ground truth is localised at IoU>=0.5, yet visually similar
surfaces are annotated as anywhere from 1 to 5 boxes and a fixed top/bottom
split already scores 0.28 AP.  Anchor+NMS has no way to express "how many";
this head predicts the count and the cut explicitly.

The decisive early metric is COUNT ACCURACY on val: if the head cannot beat
"always predict the majority count", the convention is not predictable from
pixels and the idea is dead regardless of AP.

    python -m model_newage.train_instdecomp --imgsz 256 --epochs 100
"""

import argparse
import os

import torch

from model.dataset import NEU_CLASSES, build_dataloader
from model.metrics import DetMetrics
from model.val import xywhn_to_xyxy
from model_newage.instdecomp import MAX_SLOTS, InstanceDecompModel, canonical_order


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="neu-det-3way")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--cls", type=int, default=0, help="0 = crazing")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--region-thresh", type=float, default=0.4)
    ap.add_argument("--name", default="instdecomp_crazing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    out = os.path.join("runs", args.name)
    os.makedirs(out, exist_ok=True)

    model = InstanceDecompModel({"imgsz": args.imgsz,
                                 "backbone": {"width": 0.5, "depth": 0.34}}
                                ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    tl = build_dataloader(args.data, args.train_split, imgsz=args.imgsz,
                          batch_size=args.batch_size, augment=True,
                          shuffle=True, workers=args.workers, mosaic=0.0)
    vl = build_dataloader(args.data, args.val_split, imgsz=args.imgsz,
                          batch_size=args.batch_size, augment=False,
                          shuffle=False, workers=args.workers)
    steps = len(tl) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, args.lr * 0.01)
    print(f"[data] train={len(tl.dataset)} val={len(vl.dataset)}  "
          f"目标类别: {NEU_CLASSES[args.cls]}")

    def filt(t):                      # keep only the specialist's class
        return t[t[:, 1].long() == args.cls]

    log = open(os.path.join(out, "train.log"), "a")
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        agg = {}
        for imgs, targets, _ in tl:
            loss, items = model(imgs.to(args.device), filt(targets))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); sched.step(); step += 1
            for k, v in items.items():
                agg[k] = agg.get(k, 0.0) + v
        line = f"epoch {ep:>3}/{args.epochs}  " + "  ".join(
            f"{k} {v/len(tl):.3f}" for k, v in agg.items())

        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval()
            m = DetMetrics(NEU_CLASSES)
            hit = tot = 0
            conf = torch.zeros(MAX_SLOTS, MAX_SLOTS, dtype=torch.long)
            with torch.no_grad():
                for imgs, targets, _ in vl:
                    dets = model.predict(imgs.to(args.device),
                                         region_thresh=args.region_thresh)
                    for i in range(imgs.shape[0]):
                        t = filt(targets[targets[:, 0] == i])
                        gt = xywhn_to_xyxy(t[:, 1:], args.imgsz)
                        d = dets[i]
                        m.update(d[d[:, 5].long() == args.cls] if d.numel() else d, gt)
                        if t.shape[0]:            # count accuracy on positives
                            true_k = min(t.shape[0], MAX_SLOTS)
                            pred_k = min(int((d[:, 5].long() == args.cls).sum()),
                                         MAX_SLOTS) if d.numel() else 0
                            tot += 1
                            hit += int(pred_k == true_k)
                            if pred_k >= 1:
                                conf[true_k - 1, pred_k - 1] += 1
            r = m.compute()
            ap = r["per_class"].get(NEU_CLASSES[args.cls], {}).get("AP50", 0.0)
            line += (f"  |  {NEU_CLASSES[args.cls]} AP50 {ap:.4f}"
                     f"  数量准确率 {hit/max(tot,1):.3f} ({hit}/{tot})")
            if ep == args.epochs:
                print("数量混淆矩阵 (行=真值K, 列=预测K):")
                print(conf.numpy())
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    torch.save({"cfg": model.cfg, "state_dict": model.state_dict()},
               os.path.join(out, "best.pt"))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
