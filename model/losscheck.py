"""Compare our DetectionLoss against ultralytics' v8DetectionLoss.

Same network, same weights, same batch out of OUR dataloader.  crosscheck.py
proved the assigner identical but fed it ready-made ground truth, so the target
preprocessing and the loss assembly around the assigner were never tested.
The bisection runs (M1 0.672 vs M2 0.212, identical network) put the defect in
this file or in the training loop; this tells us which.

    python -m model.losscheck
"""

import argparse
from types import SimpleNamespace

import torch

from .dataset import build_dataloader
from .loss import DetectionLoss
from .refarch import RefHeadShim


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default=None,
                    help="ultralytics checkpoint; without it the model is "
                         "randomly initialised.  TAL is prediction-dependent, "
                         "so a level starved of positives may be a symptom of "
                         "bad predictions rather than a bad assigner")
    ap.add_argument("--filter", default=None,
                    help="only images whose filename contains this, e.g. "
                         "pitted_surface -- the large-box classes were never "
                         "covered by the default alphabetical first batch")
    args = ap.parse_args(argv)

    try:
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils.loss import v8DetectionLoss
    except ImportError as e:
        raise SystemExit(f"需要 ultralytics 作为参照: {e}")

    torch.manual_seed(0)
    det = DetectionModel(cfg="yolov8n.yaml", nc=args.nc, verbose=False).to(args.device)
    if args.weights:
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        src = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        det.load_state_dict({k: v.float() for k, v in src.state_dict().items()},
                            strict=False)
        print(f"[weights] {args.weights}")
    det.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    det.train()

    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=args.batch_size, augment=False,
                              shuffle=False, workers=0)
    if args.filter:
        ds = loader.dataset
        keep = [j for j, f in enumerate(ds.img_files) if args.filter in f]
        if not keep:
            raise SystemExit(f"没有文件名包含 {args.filter!r} 的图片")
        ds.img_files = [ds.img_files[j] for j in keep]
        ds.lbl_files = [ds.lbl_files[j] for j in keep]
        if ds.labels is not None:
            ds.labels = [ds.labels[j] for j in keep]
        print(f"[filter] {args.filter}: {len(keep)} 张")
    imgs, targets, _ = next(iter(loader))
    imgs = imgs.to(args.device)
    print(f"[batch] images {tuple(imgs.shape)}  targets {tuple(targets.shape)}")
    if targets.numel():
        wh = targets[:, 4:6]
        print(f"[batch] 框面积占比 min={float((wh[:,0]*wh[:,1]).min()):.4f} "
              f"max={float((wh[:,0]*wh[:,1]).max()):.4f}")

    with torch.no_grad():
        raw = det(imgs)

    # ---- reference ------------------------------------------------------- #
    ref_crit = v8DetectionLoss(det)
    ref_batch = {"batch_idx": targets[:, 0].to(args.device),
                 "cls": targets[:, 1:2].to(args.device),
                 "bboxes": targets[:, 2:6].to(args.device),
                 "img": imgs}
    _, ref_items = ref_crit(raw, ref_batch)
    if isinstance(ref_items, dict):          # newer ultralytics returns a dict
        ref_items = list(ref_items.values())
    ref_box, ref_cls, ref_dfl = (float(v) for v in ref_items)

    # ---- ours ------------------------------------------------------------ #
    detect = det.model[-1]
    shim = RefHeadShim(args.nc, int(detect.reg_max),
                       [int(s) for s in detect.stride]).to(args.device)
    my_crit = DetectionLoss(shim, nc=args.nc)
    _, my_items = my_crit(raw, targets, shim)
    # ultralytics reports the gain-multiplied values; ours are raw
    my_box, my_cls, my_dfl = (my_items["box"] * 7.5, my_items["cls"] * 0.5,
                              my_items["dfl"] * 1.5)

    print(f"\n{'项':<8}{'参考':>12}{'我们':>12}{'相对差':>12}")
    for name, r, m in (("box", ref_box, my_box), ("cls", ref_cls, my_cls),
                       ("dfl", ref_dfl, my_dfl)):
        rel = abs(r - m) / max(abs(r), 1e-9)
        flag = "  <-- 不一致" if rel > 0.01 else ""
        print(f"{name:<8}{r:>12.5f}{m:>12.5f}{rel:>11.2%}{flag}")
    # ---- positives per pyramid level, both implementations --------------- #
    strides = [int(v) for v in det.model[-1].stride]
    sizes = [(args.imgsz // s) ** 2 for s in strides]
    ref_fg = ref_crit.get_assigned_targets_and_loss(raw, ref_batch)[0][0]
    print(f"\n{'层级':<12}{'参考正样本':>12}{'我们正样本':>12}")
    start = 0
    for s, n in zip(strides, sizes):
        print(f"stride {s:<5}{int(ref_fg[:, start:start + n].sum()):>12}"
              f"{my_items['p' + str(s)]:>12}")
        start += n
    print(f"\n正样本总数 {my_items['n_pos']}")
    print("全部一致 -> 损失没问题，bug 在训练循环；有项不一致 -> bug 在 loss.py")


if __name__ == "__main__":
    main()
