"""Diagnose *why* a class fails: localisation, classification, or both.

For every ground-truth box it finds the best-overlapping prediction regardless
of class, which separates the two failure modes:

* low best-IoU            -> the model cannot put a box there at all
  (regression range, level assignment, anchor coverage)
* high best-IoU, wrong class -> localisation is fine, the features cannot tell
  the classes apart (domain gap)

It also reports class-agnostic AP@0.5 per ground-truth class, the size of the
boxes the model does produce, and how the DFL regression range compares with
the ground-truth box sizes.

    python3 -m model.diagnose --weights runs/conv_adapter/best.pt
"""

import argparse
from collections import defaultdict

import numpy as np
import torch

from .dataset import build_dataloader
from .head import box_iou, non_max_suppression
from .metrics import compute_ap
from .val import xywhn_to_xyxy
from .yolo_sam2 import YOLOSAM2


@torch.no_grad()
def diagnose(model, loader, device="cuda", conf_thres=0.001, iou_thres=0.65,
             match_iou=0.5):
    model.eval().to(device)
    imgsz = loader.dataset.imgsz
    names = model.names

    best_iou = defaultdict(list)      # gt class -> best IoU over all predictions
    cls_match = defaultdict(list)     # gt class -> was the best box's class right
    pred_count = defaultdict(int)     # predicted class -> number of boxes kept
    pred_area = defaultdict(list)     # predicted class -> box area fraction
    gt_area = defaultdict(list)
    agn_stats = defaultdict(lambda: {"tp": [], "conf": [], "n_gt": 0})

    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        dets = non_max_suppression(preds, conf_thres=conf_thres,
                                   iou_thres=iou_thres, max_det=300)
        for i, det in enumerate(dets):
            lab = targets[targets[:, 0] == i][:, 1:].to(device)
            gt = xywhn_to_xyxy(lab, imgsz)
            for d in det:
                pred_count[int(d[5])] += 1
                pred_area[int(d[5])].append(
                    float((d[2] - d[0]) * (d[3] - d[1])) / (imgsz * imgsz))
            if gt.shape[0] == 0:
                continue
            for g in gt:
                c = int(g[0])
                gt_area[c].append(float((g[3] - g[1]) * (g[4] - g[2])) / (imgsz * imgsz))
            if det.shape[0] == 0:
                for g in gt:
                    best_iou[int(g[0])].append(0.0)
                    cls_match[int(g[0])].append(0)
                    agn_stats[int(g[0])]["n_gt"] += 1
                continue

            iou = box_iou(gt[:, 1:], det[:, :4])            # (n_gt, n_det)
            best, arg = iou.max(1)
            for k, g in enumerate(gt):
                c = int(g[0])
                best_iou[c].append(float(best[k]))
                cls_match[c].append(int(det[arg[k], 5]) == c)

            # class-agnostic AP bookkeeping: greedy one-to-one match by score
            order = det[:, 4].argsort(descending=True)
            taken = set()
            for j in order.tolist():
                c_of_best = None
                ious_j = iou[:, j]
                for k in ious_j.argsort(descending=True).tolist():
                    if ious_j[k] >= match_iou and k not in taken:
                        taken.add(k)
                        c_of_best = int(gt[k, 0])
                        break
                if c_of_best is not None:
                    agn_stats[c_of_best]["tp"].append(1)
                    agn_stats[c_of_best]["conf"].append(float(det[j, 4]))
            for g in gt:
                agn_stats[int(g[0])]["n_gt"] += 1

    print(f"\n{'class':<18}{'n_gt':>6}{'medIoU':>8}{'IoU>=.5':>9}{'cls ok':>8}"
          f"{'agnAP50':>9}{'gt area':>9}{'pred n':>8}{'pred area':>10}")
    for c, name in enumerate(names):
        ious = np.array(best_iou[c]) if best_iou[c] else np.zeros(1)
        loc_ok = float((ious >= 0.5).mean())
        matched = np.array(cls_match[c]) if cls_match[c] else np.zeros(1)
        # class-agnostic recall-based AP proxy
        st = agn_stats[c]
        agn = (len(st["tp"]) / st["n_gt"]) if st["n_gt"] else 0.0
        pa = np.array(pred_area[c]) if pred_area[c] else np.zeros(1)
        ga = np.array(gt_area[c]) if gt_area[c] else np.zeros(1)
        print(f"{name:<18}{len(best_iou[c]):>6}{np.median(ious):>8.3f}"
              f"{loc_ok:>9.3f}{matched.mean():>8.3f}{agn:>9.3f}"
              f"{np.median(ga):>9.3f}{pred_count[c]:>8}{np.median(pa):>10.3f}")

    print("\n读法：")
    print("  medIoU / IoU>=.5 低  -> 定位失败（回归范围、层级分配、anchor 覆盖）")
    print("  IoU>=.5 高但 cls ok 低 -> 定位没问题，是特征分不开类（域差距）")
    print("  agnAG50 = 忽略类别时被找到的 GT 比例（类别无关召回）")
    print("  pred area 远小于 gt area -> 模型倾向于预测小框，回归范围可能不够")


@torch.no_grad()
def obj_stats(model, loader, device="cuda", n_batches=8):
    """Is the objectness branch alive, and does it agree with the class score?

    Reports the distribution of sigmoid(obj) over all anchors plus the maximum
    per image.  A branch that has collapsed to "nothing is an object" shows a
    max close to zero; a working one peaks near the IoU of the best anchors.
    """
    model.eval().to(device)
    if not model.head.use_obj:
        print("\n[obj] head.use_obj=False -- 没有 objectness 分支")
        return
    obj_all, obj_max, cls_max, prod_max = [], [], [], []
    for bi, (imgs, _, _) in enumerate(loader):
        if bi >= n_batches:
            break
        feats = model.neck(model.backbone(imgs.to(device)))
        raw = model.head(feats)[1]
        cls, reg, obj, _, _ = model.head.flatten(raw)
        o = obj.sigmoid().squeeze(-1)                      # (B, A)
        c = cls.sigmoid().amax(-1)                         # (B, A)
        obj_all.append(o.flatten().cpu())
        obj_max.append(o.amax(-1).cpu())
        cls_max.append(c.amax(-1).cpu())
        prod_max.append((o * c).amax(-1).cpu())
    a = torch.cat(obj_all).numpy()
    print("\n[obj] sigmoid(obj) 分布："
          f" p50={np.percentile(a, 50):.4f} p99={np.percentile(a, 99):.4f} "
          f"max={a.max():.4f}")
    print(f"[obj] 每图最大值 均值={torch.cat(obj_max).mean():.4f}  "
          f"cls 每图最大值 均值={torch.cat(cls_max).mean():.4f}  "
          f"两者相乘 均值={torch.cat(prod_max).mean():.4f}")
    print("      每图最大 obj 接近 0 -> 分支塌掉了；接近正样本 IoU -> 工作正常")


@torch.no_grad()
def level_stats(model, loader, device="cuda", n_batches=8, score_thres=0.25):
    """Which pyramid level produces the confident detections, and how big are
    the boxes it can emit?

    A level whose median box size sits exactly at ``(reg_max-1)*stride*2`` is
    saturating its DFL range: it has been made responsible for objects it
    cannot physically describe.
    """
    model.eval().to(device)
    head = model.head
    counts, sizes = defaultdict(int), defaultdict(list)
    for bi, (imgs, _, _) in enumerate(loader):
        if bi >= n_batches:
            break
        preds = model(imgs.to(device))                    # (B, A, 4+nc)
        boxes, scores = preds[..., :4], preds[..., 4:].amax(-1)
        # anchor index ranges per level
        start = 0
        feats = None
        for lvl, stride in enumerate(head.strides):
            n = (loader.dataset.imgsz // stride) ** 2
            sel = scores[:, start:start + n] > score_thres
            b = boxes[:, start:start + n][sel]
            counts[stride] += int(sel.sum())
            if b.numel():
                side = torch.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
                sizes[stride] += side.cpu().tolist()
            start += n

    print(f"\n[level] score>{score_thres} 的预测按层级分布：")
    for stride in head.strides:
        s_ = np.array(sizes[stride]) if sizes[stride] else np.zeros(1)
        cap = 2 * (head.reg_max - 1) * stride
        print(f"  stride {stride:>2}: {counts[stride]:>6} 个  "
              f"框长边 p50={np.median(s_):>6.1f} p90={np.percentile(s_, 90):>6.1f} "
              f"max={s_.max():>6.1f} px   (该层上限 {cap} px)")
    print("  某层 max 紧贴上限 -> 它在被迫描述表示不了的目标")


def main(argv=None):
    ap = argparse.ArgumentParser("diagnose per-class failure modes")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    if ck.get("ref_arch"):
        from .refarch import RefArchModel
        model = RefArchModel.load(args.weights, map_location="cpu")
    elif "model" in ck and hasattr(ck["model"], "state_dict"):
        # a raw ultralytics checkpoint: wrap its network so the same diagnosis
        # runs on both sides.  Those weights were trained on plain /255 inputs,
        # so ImageNet normalisation must be off for them.
        from .refarch import RefArchModel
        model = RefArchModel({"nc": 6, "imgsz": args.imgsz})
        model.det.load_state_dict(
            {k: v.float() for k, v in ck["model"].state_dict().items()},
            strict=False)
        args.normalize = False
        print(f"[weights] ultralytics 权重，已关闭 ImageNet 归一化")
    else:
        model = YOLOSAM2.load(args.weights, map_location="cpu")
    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=args.batch_size, augment=False,
                              shuffle=False, workers=args.workers,
                              normalize=args.normalize)
    diagnose(model, loader, device=args.device, conf_thres=args.conf)
    obj_stats(model, loader, device=args.device)
    level_stats(model, loader, device=args.device)

    # how far the DFL can reach at each detection level, vs the GT box sizes
    reg_max = model.head.reg_max
    print(f"\nDFL 回归范围：reg_max={reg_max}，单边最远 {reg_max - 1} 格")
    for s in model.head.strides:
        print(f"  stride {s:>2}: 单边 {(reg_max - 1) * s:>4} px  "
              f"=> 中心 anchor 最大可表示框边长 {2 * (reg_max - 1) * s:>4} px")


if __name__ == "__main__":
    main()
