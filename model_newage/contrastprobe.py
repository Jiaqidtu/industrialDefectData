"""Is 'contrast to this image's own background' a usable confidence signal?

Three measurements say a defect here is a *relational* quantity, not an
appearance class: patches is the only class whose band energy is lower than
its surroundings; unprompted SAM2 sees the whole plate as one object; and true
and false positives are inseparable by confidence even though localisation and
classification are near-perfect.  Every backbone we tested computes absolute
features, so none of them can represent "differs from its own plate".

Before building a background-contrast head, this measures whether the signal
exists at all: for each detection, the distance between the box's band-energy
descriptor and the same image's background descriptor, scored by TP/FP AUC
against the detector's own confidence.  No training involved.

    python -m model_newage.contrastprobe --weights runs/f3_freq_s0/best.pt
"""

import argparse

import numpy as np
import torch

from model.dataset import build_dataloader
from model.head import box_iou, non_max_suppression
from model.refarch import load_any
from model.val import xywhn_to_xyxy
from model_newage.bandprobe import band_index


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    r = ranks[: len(pos)].sum()
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def band_map(gray: np.ndarray, patch: int, idx: np.ndarray, n_bands: int):
    """(H,W) -> (g,g,n_bands) per-patch band energies."""
    S = gray.shape[0]
    g = S // patch
    p = gray[: g * patch, : g * patch].reshape(g, patch, g, patch)
    p = p.transpose(0, 2, 1, 3).reshape(-1, patch, patch)
    spec = np.log1p(np.abs(np.fft.rfft2(p, norm="ortho"))).reshape(len(p), -1)
    out = np.zeros((len(p), n_bands), dtype=np.float32)
    for b in range(n_bands):
        out[:, b] = spec[:, idx == b].mean(1)
    return out.reshape(g, g, n_bands)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/f3_freq_s0/best.pt")
    ap.add_argument("--arch", default="yolov8n.yaml")
    ap.add_argument("--data", default="neu-det-3way")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--n-radial", type=int, default=4)
    ap.add_argument("--n-orient", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    from PIL import Image

    model, norm = load_any(args.weights, arch=args.arch, imgsz=args.imgsz)
    model = model.to(args.device).eval()
    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=8, augment=False, shuffle=False,
                              workers=2, normalize=norm)

    idx = band_index(args.patch, args.n_radial, args.n_orient)
    n_bands = args.n_radial * args.n_orient
    S, p = args.imgsz, args.patch
    g = S // p

    feats = {k: {"tp": [], "fp": []} for k in ("检测器置信度", "背景对比距离",
                                               "置信度x对比")}
    for imgs, targets, paths in loader:
        with torch.no_grad():
            preds = model(imgs.to(args.device))
        dets = non_max_suppression(preds, args.conf, args.iou, 300)
        for i, det in enumerate(dets):
            if det.shape[0] == 0:
                continue
            det = det.cpu()[: args.topk]
            lab = xywhn_to_xyxy(targets[targets[:, 0] == i][:, 1:], args.imgsz)
            is_tp = np.zeros(det.shape[0], dtype=bool)
            if lab.shape[0]:
                iou = box_iou(lab[:, 1:], det[:, :4]).numpy()
                same = lab[:, :1].numpy() == det[:, 5].numpy()[None, :]
                is_tp = ((iou * same) >= 0.5).any(0)

            gray = np.asarray(Image.open(paths[i]).convert("L")
                              .resize((S, S)), dtype=np.float32)
            bm = band_map(gray, p, idx, n_bands)          # (g,g,K)

            # the image's own background: the median descriptor over patches
            # OUTSIDE every detection (robust to the defects themselves)
            occ = np.zeros((g, g), dtype=bool)
            for b in det.tolist():
                x1, y1, x2, y2 = (int(v // p) for v in b[:4])
                occ[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1] = True
            bg_src = bm[~occ] if (~occ).sum() >= 8 else bm.reshape(-1, n_bands)
            bg = np.median(bg_src, axis=0)
            scale = bg_src.std(axis=0) + 1e-6

            for j, b in enumerate(det.tolist()):
                x1, y1 = max(int(b[0] // p), 0), max(int(b[1] // p), 0)
                x2, y2 = int(np.ceil(b[2] / p)), int(np.ceil(b[3] / p))
                region = bm[y1:y2, x1:x2].reshape(-1, n_bands)
                if region.size == 0:
                    continue
                # normalised deviation of the box's texture from the plate's
                d = float(np.abs((region.mean(0) - bg) / scale).mean())
                key = "tp" if is_tp[j] else "fp"
                conf = float(b[4])
                feats["检测器置信度"][key].append(conf)
                feats["背景对比距离"][key].append(d)
                feats["置信度x对比"][key].append(conf * d)

    n_tp = len(feats["检测器置信度"]["tp"])
    n_fp = len(feats["检测器置信度"]["fp"])
    print(f"\n真阳性 {n_tp}   假阳性 {n_fp}")
    print(f"\n{'信号':<14}{'TP均值':>10}{'FP均值':>10}{'AUC':>8}")
    base = None
    for k, v in feats.items():
        pos, neg = np.array(v["tp"]), np.array(v["fp"])
        a = auc(pos, neg)
        if base is None:
            base = a
        mark = "  <- 参照" if k == "检测器置信度" else (
            "  ** 超过置信度" if a > base + 0.02 else "")
        print(f"{k:<14}{pos.mean():>10.3f}{neg.mean():>10.3f}{a:>8.3f}{mark}")
    print("\n背景对比的 AUC 若不超过置信度，或组合无增量 -> 该原语不含新信息，"
          "不必建；明显超过 -> 值得做成可学习的头")


if __name__ == "__main__":
    main()
