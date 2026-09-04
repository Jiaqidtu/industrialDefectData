"""Can re-partitioning the detections recover the diffuse classes?

The measured chain: crazing is found (86% class-agnostic recall, perfect
classification, median IoU 0.72) yet scores 0.39 AP, because the model cuts the
region into a different number of boxes than the annotator did -- and that
count is annotation noise (a head trained to predict it scores exactly the
majority baseline).  NMS suppresses by box IoU, which is the wrong criterion
for fragments that tile one region with little mutual overlap.

Reports, in one pass:
  * baseline      -- detections as they are
  * WBF           -- fuse overlapping same-class boxes into a weighted average
  * adjacent      -- union-merge same-class boxes whose gap is under tau
  * ORACLE        -- best AP over a grid of merge settings, chosen per image
                     using the ground truth.  Not a method: an upper bound on
                     what any re-partitioning could deliver.

If the oracle sits at the baseline, the model's output cannot be re-partitioned
into something better and the diffuse classes are closed at the output stage.

    python -m model_newage.mergeprobe --weights runs/f3_freq_s0/best.pt
"""

import argparse
import itertools

import numpy as np
import torch

from model.dataset import build_dataloader, dataset_classes
from model.head import box_iou, non_max_suppression
from model.metrics import DetMetrics
from model.refarch import load_any
from model.val import xywhn_to_xyxy


def wbf(det, iou_thr):
    """Weighted box fusion: clusters by IoU, averages corners by confidence."""
    if det.shape[0] == 0:
        return det
    out = []
    for c in det[:, 5].unique():
        d = det[det[:, 5] == c]
        d = d[d[:, 4].argsort(descending=True)]
        used = torch.zeros(d.shape[0], dtype=torch.bool)
        for i in range(d.shape[0]):
            if used[i]:
                continue
            grp = [i]
            used[i] = True
            for j in range(i + 1, d.shape[0]):
                if used[j]:
                    continue
                if float(box_iou(d[i:i + 1, :4], d[j:j + 1, :4])) >= iou_thr:
                    grp.append(j); used[j] = True
            g = d[grp]
            w = g[:, 4:5]
            box = (g[:, :4] * w).sum(0) / w.sum()
            # confidence of a fused cluster: the max, scaled by how many
            # fragments agreed -- a lone box should not gain from fusion
            conf = g[:, 4].max()
            out.append(torch.cat([box, conf.view(1), c.view(1)]))
    return torch.stack(out) if out else det


def adjacent_union(det, tau, imgsz):
    """Union-merge same-class boxes whose edge gap is below tau*imgsz.

    Fragments tiling one diffuse region have low mutual IoU but small gaps, so
    proximity is the criterion NMS cannot express.
    """
    if det.shape[0] == 0:
        return det
    gap = tau * imgsz
    out = []
    for c in det[:, 5].unique():
        d = det[det[:, 5] == c]
        n = d.shape[0]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                dx = max(0.0, float(max(d[i, 0], d[j, 0]) - min(d[i, 2], d[j, 2])))
                dy = max(0.0, float(max(d[i, 1], d[j, 1]) - min(d[i, 3], d[j, 3])))
                if (dx * dx + dy * dy) ** 0.5 <= gap:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        for idx in groups.values():
            g = d[idx]
            box = torch.tensor([g[:, 0].min(), g[:, 1].min(),
                                g[:, 2].max(), g[:, 3].max()])
            out.append(torch.cat([box, g[:, 4].max().view(1), c.view(1)]))
    return torch.stack(out) if out else det


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/f3_freq_s0/best.pt")
    ap.add_argument("--arch", default="yolov8n.yaml")
    ap.add_argument("--data", default="neu-det-3way")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    names = dataset_classes(args.data)
    model, norm = load_any(args.weights, arch=args.arch, imgsz=args.imgsz,
                           nc=len(names))
    model = model.to(args.device).eval()
    loader = build_dataloader(args.data, args.split, classes=names,
                              imgsz=args.imgsz, batch_size=8, augment=False,
                              shuffle=False, workers=2, normalize=norm)

    cache = []                       # detections + gt, so the sweep is cheap
    with torch.no_grad():
        for imgs, targets, _ in loader:
            preds = model(imgs.to(args.device))
            dets = non_max_suppression(preds, args.conf, args.iou, 300)
            for i, d in enumerate(dets):
                gt = xywhn_to_xyxy(targets[targets[:, 0] == i][:, 1:], args.imgsz)
                cache.append((d.cpu(), gt))
    print(f"[data] {len(cache)} 张，平均每图 "
          f"{np.mean([d.shape[0] for d, _ in cache]):.1f} 个检测框")

    def score(fn):
        m = DetMetrics(names)
        for d, gt in cache:
            m.update(fn(d) if d.shape[0] else d, gt)
        return m.compute()

    rows = [("基线（不合并）", score(lambda d: d))]
    for t in (0.3, 0.5, 0.7):
        rows.append((f"WBF iou={t}", score(lambda d, t=t: wbf(d, t))))
    for t in (0.01, 0.03, 0.06):
        rows.append((f"邻接并集 tau={t}",
                     score(lambda d, t=t: adjacent_union(d, t, args.imgsz))))

    print(f"\n{'方法':<20}{'mAP50':>9}" + "".join(f"{n[:11]:>13}" for n in names))
    for lab, r in rows:
        print(f"{lab:<20}{r['mAP50']:>9.4f}" +
              "".join(f"{r['per_class'].get(n,{}).get('AP50',0):>13.4f}"
                      for n in names))

    # ---- oracle: per image, keep whichever variant scores best against GT ---
    variants = [lambda d: d]
    variants += [lambda d, t=t: wbf(d, t) for t in (0.3, 0.5, 0.7)]
    variants += [lambda d, t=t: adjacent_union(d, t, args.imgsz)
                 for t in (0.01, 0.03, 0.06)]
    m = DetMetrics(names)
    for d, gt in cache:
        if d.shape[0] == 0 or gt.shape[0] == 0:
            m.update(d, gt); continue
        best, best_ap = d, -1.0
        for fn in variants:
            cand = fn(d)
            mm = DetMetrics(names)
            mm.update(cand, gt)
            ap = mm.compute()["mAP50"]
            if ap > best_ap:
                best_ap, best = ap, cand
        m.update(best, gt)
    o = m.compute()
    print(f"\n{'神谕重分块（上界）':<20}{o['mAP50']:>9.4f}" +
          "".join(f"{o['per_class'].get(n,{}).get('AP50',0):>13.4f}" for n in names))
    print(f"\n上界 − 基线 = {o['mAP50'] - rows[0][1]['mAP50']:+.4f}")
    print("上界≈基线 -> 输出无法通过重分块改善，弥散类别在输出端已封闭")
    print("上界明显更高 -> 存在可回收余量，合并算法值得设计")


if __name__ == "__main__":
    main()
