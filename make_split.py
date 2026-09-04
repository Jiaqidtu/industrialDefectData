"""Build a train/val/test split so the reported numbers are not tuned on.

Everything measured so far used one 360-image validation set for both choosing
(NMS threshold, best epoch, which of ~20 configurations to keep) and reporting.
With a seed standard deviation of 0.003-0.008 and that many comparisons, some
of the reported margin is selection.  This pools the 1800 images and cuts them
into three, stratified by the image's primary defect class.

Symlinks, not copies: the home quota is 30 GB and the images are 575 MB.

    python make_split.py                  # 1260 / 270 / 270, seed 0
    python make_split.py --check          # report an existing split only
"""

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
         "rolled-in_scale", "scratches"]


def primary_class(label_path):
    """The class of the first box -- the one the filename is named after."""
    with open(label_path) as fh:
        for line in fh:
            if line.strip():
                return int(line.split()[0])
    return -1


def collect(src):
    items = []
    for split in ("train", "validation"):
        for img in sorted(glob.glob(os.path.join(src, split, "images", "*"))):
            stem = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(src, split, "labels", stem + ".txt")
            if os.path.exists(lbl):
                items.append((os.path.abspath(img), os.path.abspath(lbl),
                              primary_class(lbl)))
    return items


def report(dst, splits):
    print(f"\n{'split':<8}{'图数':>7}" + "".join(f"{n[:12]:>15}" for n in NAMES))
    for name, items in splits.items():
        c = Counter(cls for _, _, cls in items)
        print(f"{name:<8}{len(items):>7}" +
              "".join(f"{c.get(i, 0):>15}" for i in range(len(NAMES))))
    # box-level counts matter more than image counts for AP
    print(f"\n{'split':<8}{'框数':>7}" + "".join(f"{n[:12]:>15}" for n in NAMES))
    for name, items in splits.items():
        c = Counter()
        for _, lbl, _ in items:
            for line in open(lbl):
                if line.strip():
                    c[int(line.split()[0])] += 1
        print(f"{name:<8}{sum(c.values()):>7}" +
              "".join(f"{c.get(i, 0):>15}" for i in range(len(NAMES))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="neu-det-yolo")
    ap.add_argument("--dst", default="neu-det-3way")
    ap.add_argument("--ratios", default="0.70,0.15,0.15")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        man = json.load(open(os.path.join(args.dst, "split.json")))
        splits = {k: [(i, l, c) for i, l, c in v] for k, v in man["splits"].items()}
        report(args.dst, splits)
        return

    items = collect(args.src)
    print(f"[data] 共 {len(items)} 张（{args.src} 的 train + validation 合并）")

    by_cls = defaultdict(list)
    for it in items:
        by_cls[it[2]].append(it)

    r_tr, r_va, r_te = (float(x) for x in args.ratios.split(","))
    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for cls in sorted(by_cls):                 # stratified, class by class
        group = sorted(by_cls[cls])
        rng.shuffle(group)
        n = len(group)
        n_tr, n_va = round(n * r_tr), round(n * r_va)
        splits["train"] += group[:n_tr]
        splits["val"] += group[n_tr:n_tr + n_va]
        splits["test"] += group[n_tr + n_va:]

    seen = set()
    for name, group in splits.items():
        for img, _, _ in group:
            assert img not in seen, f"图像出现在多个 split: {img}"
            seen.add(img)
    assert len(seen) == len(items), "有图像丢失"

    for name, group in splits.items():
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.dst, name, sub), exist_ok=True)
        for img, lbl, _ in group:
            for src_path, sub in ((img, "images"), (lbl, "labels")):
                dst_path = os.path.join(args.dst, name, sub,
                                        os.path.basename(src_path))
                if os.path.lexists(dst_path):
                    os.remove(dst_path)
                os.symlink(src_path, dst_path)

    with open(os.path.join(args.dst, "split.json"), "w") as fh:
        json.dump({"seed": args.seed, "ratios": [r_tr, r_va, r_te],
                   "source": os.path.abspath(args.src),
                   "splits": {k: [[i, l, c] for i, l, c in v]
                              for k, v in splits.items()}}, fh, indent=1)

    yaml_path = os.path.join(os.path.abspath(args.dst), "neu-det-3way.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"# stratified 3-way split, seed {args.seed}\n")
        fh.write(f"path: {os.path.abspath(args.dst)}\n")
        fh.write("train: train/images\nval: val/images\ntest: test/images\n")
        fh.write(f"nc: {len(NAMES)}\nnames:\n")
        for i, n in enumerate(NAMES):
            fh.write(f"  {i}: {n}\n")

    report(args.dst, splits)
    print(f"\n已写入 {args.dst}/（符号链接，不占额外空间）")
    print(f"ultralytics 配置: {yaml_path}")
    print(f"划分清单: {args.dst}/split.json（seed {args.seed}，可完全复现）")


if __name__ == "__main__":
    main()
