"""Smoke test: build every config, run a forward/backward pass, check shapes.

    python -m model.selftest [--imgsz 320] [--data neu-det-yolo]
"""

import argparse
import glob
import os

import torch

from .head import non_max_suppression
from .yolo_sam2 import YOLOSAM2, build_model, load_config

CFG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def check_model(cfg_path, imgsz, device):
    name = os.path.basename(cfg_path)
    cfg = load_config(cfg_path, imgsz=imgsz)
    model = YOLOSAM2(cfg).to(device)
    counts = model.param_counts()

    x = torch.rand(2, 3, imgsz, imgsz, device=device)
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.3],
                            [0, 3, 0.2, 0.7, 0.1, 0.1],
                            [1, 5, 0.4, 0.4, 0.5, 0.05]], device=device)
    model.train()
    loss, items = model(x, targets)
    loss.backward()
    grads = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                for p in model.trainable_parameters())
    frozen_with_grad = [n for n, p in model.named_parameters()
                        if not p.requires_grad and p.grad is not None]

    model.eval()
    with torch.no_grad():
        preds = model(x)
    dets = non_max_suppression(preds, conf_thres=0.0, iou_thres=0.65, max_det=10)

    assert preds.shape[0] == 2 and preds.shape[2] == 4 + model.nc, preds.shape
    assert not frozen_with_grad, f"frozen params received gradients: {frozen_with_grad[:3]}"
    assert grads > 0, "no trainable parameter received a gradient"
    assert torch.isfinite(loss), "loss is not finite"

    print(f"[ok] {name:<28} stages={counts['adapter_stages']} "
          f"trainable={counts['trainable_M']:.3f}M total={counts['total_M']:.3f}M "
          f"anchors={preds.shape[1]} loss={items['total']:.2f} "
          f"n_pos={items['n_pos']} dets={dets[0].shape[0]}")
    return counts


def check_dataset(root, imgsz):
    from .dataset import build_dataloader
    dl = build_dataloader(root, "train", imgsz=imgsz, batch_size=2, augment=True,
                          shuffle=True, workers=0)
    imgs, targets, paths = next(iter(dl))
    assert imgs.shape[1:] == (3, imgsz, imgsz), imgs.shape
    assert targets.ndim == 2 and targets.shape[1] == 6, targets.shape
    assert targets[:, 2:].min() >= 0 and targets[:, 2:].max() <= 1
    print(f"[ok] dataset {root:<20} images={len(dl.dataset)} "
          f"batch={tuple(imgs.shape)} targets={tuple(targets.shape)}")


def check_metrics():
    from .metrics import DetMetrics
    m = DetMetrics(["a", "b"])
    labels = torch.tensor([[0., 10, 10, 50, 50], [1., 100, 100, 140, 140]])
    perfect = torch.tensor([[10., 10, 50, 50, 0.9, 0.], [100., 100, 140, 140, 0.8, 1.]])
    m.update(perfect, labels)
    res = m.compute()
    assert res["mAP50"] > 0.99, res
    assert res["mAP50-95"] > 0.99, res
    print(f"[ok] metrics: perfect predictions -> mAP50={res['mAP50']:.3f} "
          f"mAP50-95={res['mAP50-95']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    for cfg in sorted(glob.glob(os.path.join(CFG_DIR, "*.yaml"))):
        if "base_plus" in cfg or "small" in cfg:
            continue  # heavy variants: structure is covered by the stage sweep
        check_model(cfg, args.imgsz, args.device)

    # the branch add/remove knob itself
    for stages in ([], [4], [3, 4], [1, 2, 3, 4]):
        model = build_model(imgsz=args.imgsz,
                            backbone={"variant": "tiny", "adapter_stages": stages},
                            neck={"width": 96})
        c = model.param_counts()
        print(f"[ok] adapter_stages={str(stages):<12} adapters={c['adapters_M']:.3f}M "
              f"trainable={c['trainable_M']:.3f}M")

    check_metrics()
    if os.path.isdir(args.data):
        check_dataset(args.data, args.imgsz)
    else:
        print(f"[skip] dataset {args.data} not found")
    print("all checks passed")


if __name__ == "__main__":
    main()
