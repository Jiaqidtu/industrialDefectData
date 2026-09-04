"""Do the six defect classes actually occupy different frequency bands?

The dual path works -- removing the Fourier transform drops mAP50 from 0.748
back to 0.730, the baseline -- so the spectrum carries the signal.  The next
structural step would route each band separately instead of mixing the whole
spectral path with one scalar gate.  That is only worth building if the classes
separate in band space, which is measurable from the pixels with no training.

Reports, per class, the mean band energy inside the ground-truth boxes minus
the mean outside them: a band that is merely bright everywhere is not
discriminative, a band that lights up on the defect is.

    python -m model_newage.bandprobe
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image

from model.dataset import NEU_CLASSES


def band_index(patch: int, n_radial: int, n_orient: int):
    fy = np.fft.fftfreq(patch).reshape(-1, 1)
    fx = np.fft.rfftfreq(patch).reshape(1, -1)
    fy, fx = np.broadcast_arrays(fy, fx)
    r = np.sqrt(fy ** 2 + fx ** 2)
    th = np.mod(np.arctan2(fy, fx), np.pi)
    ri = np.clip((r / (r.max() + 1e-9) * n_radial).astype(int), 0, n_radial - 1)
    ti = np.clip((th / np.pi * n_orient).astype(int), 0, n_orient - 1)
    return (ri * n_orient + ti).ravel()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--n-radial", type=int, default=4)
    ap.add_argument("--n-orient", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args(argv)

    idx = band_index(args.patch, args.n_radial, args.n_orient)
    n_bands = args.n_radial * args.n_orient
    inside = {c: np.zeros(n_bands) for c in range(len(NEU_CLASSES))}
    outside = {c: np.zeros(n_bands) for c in range(len(NEU_CLASSES))}
    count = {c: 0 for c in range(len(NEU_CLASSES))}

    files = sorted(glob.glob(os.path.join(args.data, args.split, "images", "*")))
    for f in files:
        lbl = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl):
            continue
        rows = [l.split() for l in open(lbl) if l.strip()]
        if not rows:
            continue
        c = int(rows[0][0])
        if count[c] >= args.per_class:
            continue
        im = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
        S, p = im.shape[0], args.patch
        g = S // p
        # defect mask on the patch grid
        mask = np.zeros((g, g), dtype=bool)
        for r in rows:
            _, cx, cy, bw, bh = (float(v) for v in r)
            x1, y1 = int((cx - bw / 2) * g), int((cy - bh / 2) * g)
            x2, y2 = int(np.ceil((cx + bw / 2) * g)), int(np.ceil((cy + bh / 2) * g))
            mask[max(y1, 0):y2, max(x1, 0):x2] = True
        if mask.all() or not mask.any():
            continue                       # no contrast to measure on this image

        patches = im.reshape(g, p, g, p).transpose(0, 2, 1, 3).reshape(-1, p, p)
        spec = np.log1p(np.abs(np.fft.rfft2(patches, norm="ortho")))
        spec = spec.reshape(spec.shape[0], -1)
        energy = np.zeros((spec.shape[0], n_bands))
        for b in range(n_bands):
            sel = idx == b
            energy[:, b] = spec[:, sel].mean(1)
        m = mask.ravel()
        inside[c] += energy[m].mean(0)
        outside[c] += energy[~m].mean(0)
        count[c] += 1

    print(f"每类取 {args.per_class} 张，patch={args.patch}，"
          f"{args.n_radial} 径向 x {args.n_orient} 方向\n")
    print("值 = 框内平均频带能量 - 框外平均频带能量（越大越有判别力）")
    hdr = "  ".join(f"r{r}o{o}" for r in range(args.n_radial)
                    for o in range(args.n_orient))
    print(f"{'类别':<18}{hdr}")
    sig = {}
    for c, name in enumerate(NEU_CLASSES):
        if count[c] == 0:
            continue
        d = (inside[c] - outside[c]) / count[c]
        sig[name] = d
        print(f"{name:<18}" + "  ".join(f"{v:+5.2f}" for v in d))

    if len(sig) > 1:
        names = list(sig)
        M = np.stack([sig[n] for n in names])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        print(f"\n类间余弦相似度（1.0 = 频带特征完全相同，无法区分）")
        print(f"{'':<18}" + "".join(f"{n[:9]:>10}" for n in names))
        for i, n in enumerate(names):
            print(f"{n:<18}" + "".join(f"{float(M[i] @ M[j]):>10.3f}"
                                       for j in range(len(names))))
        off = [float(M[i] @ M[j]) for i in range(len(names))
               for j in range(len(names)) if i != j]
        print(f"\n非对角平均相似度 {np.mean(off):.3f}")
        print("  < 0.8  -> 类别在频带空间可分，按频带路由有依据")
        print("  > 0.95 -> 各类频带特征几乎相同，标量门控就够了")


if __name__ == "__main__":
    main()
