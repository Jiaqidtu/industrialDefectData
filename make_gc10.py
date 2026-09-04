"""GC10-DET (VOC XML) -> YOLO txt, with the same stratified 3-way split.

Why this dataset: NEU-DET's images are 200x200 upsampled to 640, so the pixel
domain is information-saturated and every capacity or architecture change we
measured came back null.  GC10-DET is natively 2048x1000 with 10 classes, so
it can tell us whether those null results are a property of surface-defect
detection or an artefact of NEU-DET's resolution.

The protocol is copied deliberately: 70/15/15, stratified on each image's
primary class, seed 0, symlinks rather than copies.  Anything else and the two
datasets cannot share a table.

    python make_gc10.py
    python make_gc10.py --check
"""

import argparse
import glob
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# GC10-DET's XML names carry a numeric prefix; keep the canonical order
CLASSES = ["1_chongkong", "2_hanfeng", "3_yueyawan", "4_shuiban", "5_youban",
           "6_siban", "7_yiwu", "8_yahen", "9_zhehen", "10_yaozhe"]
EN = ["punching_hole", "weld_line", "crescent_gap", "water_spot", "oil_spot",
      "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding"]
IDX = {c: i for i, c in enumerate(CLASSES)}
# The released annotations spell class 10 three ways; matching strictly drops
# 132 images and leaves waist_folding with one box in val and one in test,
# which cannot be evaluated.  Normalise instead of discarding.
ALIASES = {"10_yaozhed": "10_yaozhe", "d": "10_yaozhe", "10_yaozh": "10_yaozhe"}


def parse(xml_path):
    """-> (width, height, [(cls_idx, x1, y1, x2, y2), ...]) or None."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    if w <= 0 or h <= 0:
        return None
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        name = ALIASES.get(name, name)
        if name not in IDX:
            return ("UNKNOWN", name)
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in
                          ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append((IDX[name], x1, y1, x2, y2))
    return (w, h, boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="gc10-raw/VOC2007")
    ap.add_argument("--dst", default="gc10-3way")
    ap.add_argument("--ratios", default="0.70,0.15,0.15")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    lbl_cache = os.path.join(args.dst, "_labels")
    if args.check:
        man = json.load(open(os.path.join(args.dst, "split.json")))
        for name, items in man["splits"].items():
            c = Counter(cls for _, _, cls in items)
            print(f"{name:<7}{len(items):>6} 图  " +
                  " ".join(f"{EN[i][:9]}:{c.get(i,0)}" for i in range(len(EN))))
        return

    items, unknown, skipped = [], Counter(), 0
    for xml in sorted(glob.glob(os.path.join(args.src, "Annotations", "*.xml"))):
        stem = os.path.splitext(os.path.basename(xml))[0]
        img = os.path.join(args.src, "JPEGImages", stem + ".jpg")
        if not os.path.exists(img):
            skipped += 1
            continue
        p = parse(xml)
        if p is None:
            skipped += 1
            continue
        if p[0] == "UNKNOWN":
            unknown[p[1]] += 1
            continue
        w, h, boxes = p
        if not boxes:
            skipped += 1
            continue
        items.append((os.path.abspath(img), stem, w, h, boxes, boxes[0][0]))

    print(f"[data] 可用 {len(items)} 张 (跳过 {skipped}"
          + (f"，未知类别 {dict(unknown)}" if unknown else "") + ")")

    # write YOLO labels once, into a cache the splits symlink to
    os.makedirs(lbl_cache, exist_ok=True)
    for img, stem, w, h, boxes, _ in items:
        with open(os.path.join(lbl_cache, stem + ".txt"), "w") as fh:
            for c, x1, y1, x2, y2 in boxes:
                fh.write(f"{c} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n")

    by_cls = defaultdict(list)
    for it in items:
        by_cls[it[5]].append(it)
    r_tr, r_va, _ = (float(x) for x in args.ratios.split(","))
    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for cls in sorted(by_cls):
        g = sorted(by_cls[cls], key=lambda t: t[1])
        rng.shuffle(g)
        n = len(g)
        a, b = round(n * r_tr), round(n * r_va)
        splits["train"] += g[:a]
        splits["val"] += g[a:a + b]
        splits["test"] += g[a + b:]

    seen = set()
    for group in splits.values():
        for it in group:
            assert it[0] not in seen, f"重复: {it[0]}"
            seen.add(it[0])

    for name, group in splits.items():
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.dst, name, sub), exist_ok=True)
        for img, stem, *_ in group:
            for src_p, sub, ext in ((img, "images", ".jpg"),
                                    (os.path.abspath(
                                        os.path.join(lbl_cache, stem + ".txt")),
                                     "labels", ".txt")):
                dst_p = os.path.join(args.dst, name, sub, stem + ext)
                if os.path.lexists(dst_p):
                    os.remove(dst_p)
                os.symlink(src_p, dst_p)

    with open(os.path.join(args.dst, "split.json"), "w") as fh:
        json.dump({"seed": args.seed, "source": os.path.abspath(args.src),
                   "classes": CLASSES, "classes_en": EN,
                   "splits": {k: [[i[0], i[1], i[5]] for i in v]
                              for k, v in splits.items()}}, fh, indent=1)
    yaml_p = os.path.join(os.path.abspath(args.dst), "gc10-3way.yaml")
    with open(yaml_p, "w") as fh:
        fh.write(f"path: {os.path.abspath(args.dst)}\n")
        fh.write("train: train/images\nval: val/images\ntest: test/images\n")
        fh.write(f"nc: {len(EN)}\nnames:\n")
        for i, n in enumerate(EN):
            fh.write(f"  {i}: {n}\n")

    print(f"\n{'split':<7}{'图':>6}{'框':>7}   逐类框数")
    for name, group in splits.items():
        c = Counter(b[0] for _, _, _, _, boxes, _ in group for b in boxes)
        print(f"{name:<7}{len(group):>6}{sum(c.values()):>7}   " +
              " ".join(f"{EN[i][:8]}:{c.get(i,0)}" for i in range(len(EN))))
    hw = Counter((w, h) for _, _, w, h, _, _ in items)
    print(f"\n分辨率分布: {dict(list(hw.items())[:4])}")
    print(f"已写入 {args.dst}/  |  ultralytics 配置: {yaml_p}")


if __name__ == "__main__":
    main()
