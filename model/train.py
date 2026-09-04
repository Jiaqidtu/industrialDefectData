"""Training loop for the YOLO + SAM2(Hiera) hybrid on NEU-DET.

Only the adapters, the neck and the head receive gradients; the Hiera trunk is
frozen (and kept in eval mode), so a single run fits comfortably on one GPU.

Example::

    python -m model.train --data neu-det-yolo --variant tiny \
        --adapter-stages 1,2,3,4 --sam2-ckpt /path/sam2.1_hiera_tiny.pt \
        --epochs 100 --batch-size 8 --name t_all
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .dataset import build_dataloader
from .val import efficiency_report, evaluate
from .yolo_sam2 import YOLOSAM2, load_config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_stages(text: str) -> List[int]:
    text = str(text).strip().lower()
    if text in ("", "none", "off"):
        return []
    return [int(t) for t in text.replace(" ", "").split(",") if t]


class EMA:
    """Exponential moving average of the whole model."""

    def __init__(self, model, decay: float = 0.999, warmup: int = 2000):
        # every floating-point tensor, as in the reference implementation.
        # Averaging only part of a network is wrong: the smoothed head then
        # sits on top of un-smoothed backbone features that have drifted for
        # the whole EMA window (~11 epochs at decay 0.999).
        self.ema = {k: v.detach().clone().float()
                    for k, v in model.state_dict().items()
                    if v.dtype.is_floating_point}
        self.decay, self.warmup, self.updates = decay, warmup, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup))
        for k, v in model.state_dict().items():
            if k in self.ema:
                self.ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def state_dict(self):
        return {"ema": self.ema, "updates": self.updates}

    def load_state_dict(self, sd):
        self.ema = {k: v for k, v in sd["ema"].items()}
        self.updates = sd["updates"]

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.ema.items():
            sd[k].copy_(v.to(sd[k].dtype))


def cosine_lr(step: int, total: int, warmup: int, lr0: float, lrf: float) -> float:
    if step < warmup:
        return lr0 * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return lrf * lr0 + (lr0 - lrf * lr0) * 0.5 * (1 + math.cos(math.pi * p))


def _dump_results(out_dir, args, cfg, counts, best, history,
                  done: bool, efficiency=None) -> Dict:
    """Write results.json atomically; safe to call after every epoch."""
    summary = {
        "name": args.name,
        "complete": done,
        "epochs_done": history[-1]["epoch"] if history else 0,
        "cfg": cfg,
        "args": {k: v for k, v in vars(args).items()},
        "params": counts,
        "best": best,
        "history": history,
    }
    if efficiency is not None:
        summary["efficiency"] = efficiency
    path = os.path.join(out_dir, "results.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, path)
    return summary


def build_optimizer(model, lr: float, weight_decay: float, adapter_lr_mult: float):
    """Adapters get their own (usually larger) learning rate; no decay on norms."""
    groups = {"adapter_decay": [], "adapter_no_decay": [],
              "head_decay": [], "head_no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_adapter = name.startswith("backbone.")
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        key = ("adapter_" if is_adapter else "head_") + \
              ("no_decay" if no_decay else "decay")
        groups[key].append(p)
    param_groups = [
        {"params": groups["adapter_decay"], "weight_decay": weight_decay,
         "lr": lr * adapter_lr_mult},
        {"params": groups["adapter_no_decay"], "weight_decay": 0.0,
         "lr": lr * adapter_lr_mult},
        {"params": groups["head_decay"], "weight_decay": weight_decay, "lr": lr},
        {"params": groups["head_no_decay"], "weight_decay": 0.0, "lr": lr},
    ]
    param_groups = [g for g in param_groups if g["params"]]
    opt = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
    for g in opt.param_groups:
        g["lr_scale"] = g["lr"] / lr
    return opt


def train(cfg: Dict, args) -> Dict:
    set_seed(args.seed)
    # class names travel with the dataset, not with the code
    from .dataset import dataset_classes
    names = dataset_classes(args.data)
    cfg["names"], cfg["nc"] = list(names), len(names)
    device = args.device
    out_dir = os.path.join(args.project, args.name)
    os.makedirs(out_dir, exist_ok=True)

    if cfg["backbone"].get("type") == "ultra":
        from .refarch import RefArchModel
        model = RefArchModel(cfg).to(device)
    else:
        model = YOLOSAM2(cfg).to(device)
    counts = model.param_counts()
    print(f"[model] {counts}")

    train_loader = build_dataloader(args.data, args.train_split, classes=names,
                                    imgsz=args.imgsz,
                                    batch_size=args.batch_size, augment=not args.no_aug,
                                    shuffle=True, workers=args.workers,
                                    mosaic=args.mosaic, scale=args.scale,
                                    translate=args.translate, limit=args.limit,
                                    hflip=args.hflip, vflip=args.vflip,
                                    rot90=args.rot90, brightness=args.brightness,
                                    affine_scale=args.affine_scale,
                                    affine_translate=args.affine_translate,
                                    min_box_visibility=args.min_box_visibility)
    val_loader = build_dataloader(args.data, args.val_split, classes=names,
                                  imgsz=args.imgsz,
                                  batch_size=args.batch_size, augment=False,
                                  shuffle=False, workers=args.workers,
                                  limit=args.limit)
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    opt = build_optimizer(model, args.lr, args.weight_decay, args.adapter_lr_mult)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_epochs * steps_per_epoch, total_steps // 2)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.startswith("cuda"))
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    history, best = [], {"mAP50": -1.0, "epoch": -1}
    step, start_epoch = 0, 1
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        history, best = ck["history"], ck["best"]
        step, start_epoch = ck["step"], ck["epoch"] + 1
        print(f"[resume] 从第 {ck['epoch']} 轮继续，当前最佳 "
              f"mAP50 {best['mAP50']:.4f}")
    elif args.resume:
        print(f"[resume] 没有找到 {ckpt_path}，从头开始")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if args.close_mosaic and epoch == args.epochs - args.close_mosaic + 1:
            # the last epochs see undistorted images, which lets the model
            # settle on the real data distribution before validation.  The
            # loader is rebuilt because dataloader workers hold their own copy
            # of the dataset and would keep using the old setting.
            train_loader = build_dataloader(
                args.data, args.train_split, classes=names, imgsz=args.imgsz,
                batch_size=args.batch_size, augment=not args.no_aug,
                shuffle=True, workers=args.workers,
                mosaic=0.0, scale=0.0, translate=0.0, limit=args.limit,
                hflip=args.hflip, vflip=args.vflip,
                rot90=args.rot90, brightness=args.brightness,
                affine_scale=args.affine_scale,
                affine_translate=args.affine_translate,
                min_box_visibility=args.min_box_visibility)
            print(f"[data] mosaic disabled for the final {args.close_mosaic} epochs")
        t0 = time.time()
        agg = {"box": 0.0, "cls": 0.0, "dfl": 0.0, "obj": 0.0, "total": 0.0}
        if cfg["backbone"].get("type") == "freqdual":
            agg["aux"] = 0.0
        pos = {}
        for imgs, targets, _ in train_loader:
            step += 1
            lr = cosine_lr(step, total_steps, warmup, args.lr, args.lrf)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_scale"]

            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss, items = model(imgs, targets)
                aux = getattr(model.backbone, "aux_loss", None)
                if aux is not None:
                    a = aux(targets, imgs.shape[0])
                    loss = loss + a
                    items["aux"] = float(a.detach())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)
            for k in agg:
                agg[k] += items[k]
            for k, v in items.items():
                if k.startswith("p") and k[1:].isdigit():
                    pos[k] = pos.get(k, 0) + v

        agg = {k: v / max(steps_per_epoch, 1) for k, v in agg.items()}
        line = (f"epoch {epoch:>3}/{args.epochs}  lr {lr:.2e}  "
                f"box {agg['box']:.3f}  cls {agg['cls']:.3f}  dfl {agg['dfl']:.3f}  "
                f"obj {agg['obj']:.3f}  ({time.time() - t0:.1f}s)")
        if "aux" in agg:
            line += f"  aux {agg['aux']:.3f}"
        if pos:
            n = max(steps_per_epoch, 1)
            line += "  pos/" + " ".join(
                f"{k[1:]}:{v / n:.1f}" for k, v in sorted(
                    pos.items(), key=lambda kv: int(kv[0][1:])))

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_model = model
            if ema:
                import copy as _copy
                eval_model = _copy.deepcopy(model)
                ema.copy_to(eval_model)
            res = evaluate(eval_model, val_loader, device=device,
                           conf_thres=args.conf, iou_thres=args.iou,
                           verbose=args.verbose_eval)
            line += f"  mAP50 {res['mAP50']:.4f}  mAP50-95 {res['mAP50-95']:.4f}"
            record = {"epoch": epoch, **agg, "mAP50": res["mAP50"],
                      "mAP50-95": res["mAP50-95"], "precision": res["precision"],
                      "recall": res["recall"]}
            history.append(record)
            if res["mAP50"] > best["mAP50"]:
                best = {"epoch": epoch, **{k: res[k] for k in
                                           ("mAP50", "mAP50-95", "precision", "recall")},
                        "per_class": res["per_class"]}
                eval_model.save(os.path.join(out_dir, "best.pt"),
                                extra={"epoch": epoch, "metrics": best})
            if ema:
                del eval_model
        else:
            history.append({"epoch": epoch, **agg})
        print(line, flush=True)
        # Append to a log file and rewrite results.json every epoch.  Writing
        # only at the end means a dropped connection at epoch 140 loses the
        # whole curve and every per-class number, keeping just best.pt.
        with open(os.path.join(out_dir, "train.log"), "a") as fh:
            fh.write(line + "\n")
        _dump_results(out_dir, args, cfg, counts, best, history, done=False)
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema else None,
                    "epoch": epoch, "step": step,
                    "history": history, "best": best}, tmp)
        os.replace(tmp, ckpt_path)

    if args.save_last:
        model.save(os.path.join(out_dir, "last.pt"), extra={"epoch": args.epochs})
    # The resume checkpoint carries optimizer and EMA state and is 2-4x the
    # size of the weights; it is dead weight once a run finishes, and a full
    # quota kills the *next* run at its first torch.save with an error that
    # looks nothing like "disk full".
    if not args.keep_ckpt and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    summary = _dump_results(out_dir, args, cfg, counts, best, history, done=True,
                            efficiency=efficiency_report(model, args.imgsz, device))
    print(f"[done] best mAP50 {best['mAP50']:.4f} @ epoch {best['epoch']} "
          f"-> {out_dir}")
    return summary


def build_cfg_from_args(args) -> Dict:
    return load_config(
        args.cfg,
        imgsz=args.imgsz,
        backbone={"type": args.backbone,
                  "width": args.cnn_width,
                  "depth": args.cnn_depth,
                  "variant": args.variant,
                  "adapter_stages": parse_stages(args.adapter_stages),
                  "adapter_ratio": args.adapter_ratio,
                  "adapter_type": args.adapter_type,
                  "adapter_kernel": args.adapter_kernel,
                  "unfreeze_stages": parse_stages(args.unfreeze_stages),
                  "freeze_trunk": not args.unfreeze_trunk,
                  "train_norm": args.train_norm,
                  "checkpoint": args.sam2_ckpt,
                  "patch": ([int(v) for v in str(args.freq_patch).split(",")]
                            if "," in str(args.freq_patch)
                            else int(args.freq_patch)),
                  "gate_mode": args.gate_mode,
                  "aux_seg_weight": args.aux_seg_weight,
                  "use_fft": args.use_fft,
                  "hann": args.hann,
                  "phase": args.phase,
                  "band_routing": args.band_routing,
                  "fusion": args.fusion,
                  "log_radial": args.log_radial,
                  "mps_bond": args.mps_bond,
                  "coco_pretrained": args.coco_pretrained,
                  "imgsz": args.imgsz,
                  "freq_width": args.freq_width,
                  "n_radial": args.freq_radial,
                  "n_orient": args.freq_orient,
                  "aux_cls_weight": args.aux_cls_weight},
        arch=args.ultra_arch,
        neck={"name": args.neck, "width": args.neck_width, "depth": args.neck_depth},
        head={"reg_max": args.reg_max, "use_obj": args.use_obj},
        loss={"iou_type": args.iou_type,
              "size_aware_assign": args.size_aware_assign,
              "size_tol": args.size_tol},
    )


def get_parser():
    ap = argparse.ArgumentParser("train YOLO-SAM2 on NEU-DET")
    ap.add_argument("--data", default="neu-det-yolo", help="dataset root")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--cfg", default=None, help="yaml config, see model/configs")
    # architecture
    ap.add_argument("--backbone", default="hiera",
                    choices=["hiera", "cnn", "ultra", "freqdual",
                             "freqhiera"],
                    help="cnn = CSPDarknet control group; ultra = ultralytics' "
                         "YOLOv8 network in our training loop (bisection)")
    ap.add_argument("--ultra-arch", default="yolov8n.yaml")
    # freqdual: spectral path + class gate
    ap.add_argument("--freq-patch", default="16",
                    help="FFT patch size(s), comma separated. Each patch p "
                         "feeds strides p and 2p, so '8,16,32' covers P2-P5 "
                         "and matches the 10x spread in class texture scale")
    ap.add_argument("--gate-mode", default="class",
                    choices=["class", "spatial", "off"],
                    help="off = fixed 0.5 (ablation); spatial adds a per-pixel "
                         "modulation on top of the per-image class gate")
    ap.add_argument("--aux-seg-weight", type=float, default=0.0,
                    help="dense defect-region supervision from rasterised "
                         "boxes; training only, no inference cost")
    ap.add_argument("--no-fft", dest="use_fft", action="store_false",
                    default=True, help="ablation: same path, no transform")
    ap.add_argument("--hann", action="store_true",
                    help="taper each patch before the FFT; without it the "
                         "patch edges leak into the high bands")
    ap.add_argument("--phase", action="store_true",
                    help="add the circular mean of phase per band -- where the "
                         "texture sits, which magnitude alone cannot say")
    ap.add_argument("--log-radial", action="store_true",
                    help="log-spaced radial bins: measured band contrast falls "
                         "monotonically with frequency, so linear bins spend "
                         "half the descriptor where there is no signal")
    ap.add_argument("--fusion", default="gate", choices=["gate", "xattn"],
                    help="how the spectral path joins the spatial one: scalar "
                         "class gate (validated) or cross-attention where "
                         "spatial positions query the spectral map")
    ap.add_argument("--band-routing", action="store_true",
                    help="one attention weight per band instead of one scalar "
                         "for the whole spectral path")
    ap.add_argument("--mps-bond", type=int, default=0,
                    help="tensor-network (MPS) mixing over the radial "
                         "band axis; 0 disables it, 16 is the default bond")
    ap.add_argument("--freq-width", type=int, default=96)
    ap.add_argument("--freq-radial", type=int, default=8)
    ap.add_argument("--freq-orient", type=int, default=4)
    ap.add_argument("--aux-cls-weight", type=float, default=0.2,
                    help="weight of the gate classifier's image-level loss")
    ap.add_argument("--coco-pretrained", nargs="?", const="yolov8s.pt",
                    default=None, metavar="CKPT",
                    help="initialise the CSP trunk from COCO-pretrained "
                         "ultralytics weights (default yolov8s.pt).  Our "
                         "width 0.5 matches v8s, NOT v8n.")
    ap.add_argument("--cnn-width", type=float, default=0.5)
    ap.add_argument("--cnn-depth", type=float, default=0.34)
    ap.add_argument("--variant", default="tiny",
                    choices=["tiny", "small", "base_plus", "large"])
    ap.add_argument("--adapter-stages", default="1,2,3,4",
                    help="which Hiera stages get an adapter, e.g. '3,4' or 'none'")
    ap.add_argument("--adapter-ratio", type=int, default=4)
    ap.add_argument("--adapter-type", default="mlp", choices=["mlp", "conv"],
                    help="conv adds a depthwise 3x3 inside the adapter, which "
                         "the texture-defined defect classes need")
    ap.add_argument("--adapter-kernel", type=int, default=3)
    ap.add_argument("--unfreeze-stages", default="",
                    help="fully fine-tune these Hiera stages, e.g. '3,4'")
    ap.add_argument("--sam2-ckpt", default=None, help="official SAM2 .pt checkpoint")
    ap.add_argument("--unfreeze-trunk", action="store_true")
    ap.add_argument("--train-norm", action="store_true",
                    help="also train the trunk LayerNorms")
    ap.add_argument("--neck", default="panet", choices=["panet", "bifpn", "fpn"])
    ap.add_argument("--neck-width", type=int, default=192)
    ap.add_argument("--neck-depth", type=int, default=1)
    ap.add_argument("--iou-type", default="ciou", choices=["ciou", "siou", "iou"])
    # Off by default: with TAL the classification target is already IoU-weighted,
    # so scoring with cls*obj discounts the confidence twice.  The penalty falls
    # hardest on large boxes, which the branch scores low even when they are
    # localised well -- they get found and then ranked away.  Measured on
    # NEU-DET (#19 vs #18): pitted_surface 0.384 -> 0.535, overall +0.013.
    # YOLOv8 has no objectness branch either.  --use-obj keeps it available for
    # the ablation table.
    ap.add_argument("--use-obj", dest="use_obj", action="store_true", default=False,
                    help="re-enable the objectness branch (measured harmful)")
    ap.add_argument("--no-obj", dest="use_obj", action="store_false",
                    help="explicit form of the default")
    ap.add_argument("--reg-max", type=int, default=16,
                    help="DFL bins; the reach of one anchor is (reg_max-1)*stride "
                         "per side, so raise it for datasets with very large boxes")
    ap.add_argument("--size-aware-assign", action="store_true", default=False,
                    help="DFL-consistent level assignment; measured at -0.10 mAP "
                         "on NEU-DET, kept for the ablation table")
    ap.add_argument("--size-tol", type=float, default=1.5,
                    help="tolerance factor of --size-aware-assign")
    # optimisation
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lrf", type=float, default=0.01, help="final lr fraction")
    ap.add_argument("--adapter-lr-mult", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=float, default=3)
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-aug", action="store_true")
    # Off by default: measured at -0.10 mAP50 on NEU-DET (#21 vs #19).  Our
    # mosaic crops the 2S canvas and resamples to S, which is not what the
    # reference does, and on 1440 texture images the resampling costs more
    # than the extra layouts are worth.  Kept for the ablation table.
    ap.add_argument("--mosaic", type=float, default=0.0,
                    help="probability of building a 4-image mosaic (measured "
                         "harmful on NEU-DET, see experiments.md)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="mosaic crop scale jitter (0 = always crop the full 2S canvas)")
    ap.add_argument("--translate", type=float, default=0.1,
                    help="mosaic crop centre jitter, in units of imgsz")
    ap.add_argument("--min-box-visibility", type=float, default=0.1,
                    help="drop an augmented box below this fraction of its "
                         "original area (the reference uses 0.10)")
    ap.add_argument("--affine-scale", type=float, default=0.5,
                    help="per-image scale jitter, applied to every training "
                         "image (the reference does this via RandomPerspective)")
    ap.add_argument("--affine-translate", type=float, default=0.1)
    ap.add_argument("--hflip", type=float, default=0.5)
    ap.add_argument("--vflip", type=float, default=0.3,
                    help="NEU-DET is rolled steel: every image shares the "
                         "rolling direction, so vertical flips and rotations "
                         "invent orientations the validation set never shows")
    ap.add_argument("--rot90", type=float, default=0.3)
    ap.add_argument("--brightness", type=float, default=0.3)
    ap.add_argument("--close-mosaic", type=int, default=10,
                    help="turn mosaic off for the last N epochs (YOLOv8 recipe)")
    # eval / bookkeeping
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--verbose-eval", action="store_true")
    ap.add_argument("--save-last", action="store_true",
                    help="also write last.pt. Off by default: every evaluation "
                         "path here uses best.pt, and last.pt was 2.5 GB of "
                         "never-read weights")
    ap.add_argument("--keep-ckpt", action="store_true",
                    help="keep runs/<name>/ckpt.pt after a finished run "
                         "(deleted by default; it is only for --resume)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from runs/<name>/ckpt.pt if it exists")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N images of each split; with "
                         "--val-split train this overfits a tiny subset, which "
                         "is the fastest way to tell a broken training loop "
                         "from a weak one")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main(argv=None):
    args = get_parser().parse_args(argv)
    return train(build_cfg_from_args(args), args)


if __name__ == "__main__":
    main()
