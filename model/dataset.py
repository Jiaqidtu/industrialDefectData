"""NEU-DET dataset in YOLO txt format (PIL only, no torchvision / cv2)."""

import cv2
import glob
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def dataset_classes(root: str) -> Sequence[str]:
    """Class names for a dataset root, from its split.json if it has one.

    NEU-DET's six classes were hard-coded everywhere; GC10-DET has ten, so the
    names have to travel with the data rather than the code.  make_split.py and
    make_gc10.py both record them.
    """
    import json as _json
    for key in ("classes_en", "classes"):
        f = os.path.join(root, "split.json")
        if os.path.exists(f):
            try:
                v = _json.load(open(f)).get(key)
            except Exception:
                v = None
            if v:
                return list(v)
    return NEU_CLASSES


NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
               "rolled-in_scale", "scratches"]

# SAM2 was pretrained with ImageNet statistics (sam2/utils/transforms.py).
# The Hiera trunk is frozen here, so it cannot adapt to a different input
# distribution -- feeding plain [0, 1] images measurably degrades the features.
SAM2_MEAN = (0.485, 0.456, 0.406)
SAM2_STD = (0.229, 0.224, 0.225)


def letterbox(im: Image.Image, size: int = 640, color: int = 114
              ) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """Resize keeping aspect ratio and pad to a square canvas."""
    w, h = im.size
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (color, color, color))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im, (left, top))
    return canvas, r, (left, top)


class NEUDetDataset(Dataset):
    """Returns ``(image CHW float tensor, labels (n,5) [cls, cx, cy, w, h])``.

    Labels stay normalised to the letterboxed canvas, which is what the loss
    and the mAP evaluation expect.
    """

    def __init__(self, root: str, split: str = "train", imgsz: int = 640,
                 augment: bool = False, classes: Sequence[str] = NEU_CLASSES,
                 hflip: float = 0.5, vflip: float = 0.3, rot90: float = 0.3,
                 brightness: float = 0.3, affine_scale: float = 0.5,
                 affine_translate: float = 0.1, cache_labels: bool = True,
                 normalize: bool = True, limit: int = 0, mosaic: float = 0.0,
                 scale: float = 0.0, translate: float = 0.0,
                 min_box_visibility: float = 0.1):
        self.root = root
        self.split = split
        self.imgsz = imgsz
        self.augment = augment
        self.classes = list(classes)
        self.hflip, self.vflip, self.rot90, self.brightness = (
            hflip, vflip, rot90, brightness)
        self.affine_scale, self.affine_translate = affine_scale, affine_translate
        self.normalize = normalize
        self.mosaic = mosaic
        self.scale = scale
        self.translate = translate
        # 0.1 matches the reference's area_thr.  The filter is asymmetric in
        # object size: a small box is usually wholly inside or wholly outside
        # a mosaic crop, while a near-full-height box is clipped almost every
        # time, so a strict threshold quietly starves the large-box classes.
        self.min_box_visibility = min_box_visibility
        self._mean = torch.tensor(SAM2_MEAN).view(3, 1, 1)
        self._std = torch.tensor(SAM2_STD).view(3, 1, 1)

        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"no images directory at {img_dir}")
        self.img_files = sorted(
            f for f in glob.glob(os.path.join(img_dir, "*"))
            if f.lower().endswith(IMG_EXT)
        )
        if not self.img_files:
            raise FileNotFoundError(f"no images found in {img_dir}")
        if limit:                      # tiny fixed subset, for overfitting tests
            self.img_files = self.img_files[:limit]
        self.lbl_files = [
            os.path.join(lbl_dir, os.path.splitext(os.path.basename(f))[0] + ".txt")
            for f in self.img_files
        ]
        self.labels = [self._load_label(p) for p in self.lbl_files] if cache_labels \
            else None

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def _load_label(path: str) -> np.ndarray:
        if not os.path.exists(path):
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                rows.append([float(x) for x in parts[:5]])
        return np.array(rows, dtype=np.float32) if rows else \
            np.zeros((0, 5), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _affine(self, arr: np.ndarray, lab: np.ndarray):
        """Random scale + translation about the image centre.

        The reference implementation applies this to EVERY image through
        RandomPerspective; ours only ever jittered scale inside the mosaic
        crop, so `--mosaic 0` left no geometric augmentation at all.  On 1440
        images that is the difference between learning and memorising: train
        mAP 0.64 against val 0.22 before this was added.
        """
        S = arr.shape[0]
        s = random.uniform(1 - self.affine_scale, 1 + self.affine_scale)
        tx = random.uniform(-self.affine_translate, self.affine_translate) * S
        ty = random.uniform(-self.affine_translate, self.affine_translate) * S
        # scale about the centre, then translate
        ox = (1 - s) * S / 2 + tx
        oy = (1 - s) * S / 2 + ty
        M = np.array([[s, 0.0, ox], [0.0, s, oy]], dtype=np.float32)
        arr = cv2.warpAffine(arr, M, (S, S), flags=cv2.INTER_LINEAR,
                             borderValue=(114, 114, 114))
        if len(lab):
            cx, cy = lab[:, 1] * S, lab[:, 2] * S
            bw, bh = lab[:, 3] * S, lab[:, 4] * S
            x1, y1 = (cx - bw / 2) * s + ox, (cy - bh / 2) * s + oy
            x2, y2 = (cx + bw / 2) * s + ox, (cy + bh / 2) * s + oy
            area0 = ((x2 - x1) * (y2 - y1)).clip(1e-6)
            x1, x2 = x1.clip(0, S), x2.clip(0, S)
            y1, y2 = y1.clip(0, S), y2.clip(0, S)
            keep = ((x2 - x1) * (y2 - y1)) / area0 >= self.min_box_visibility
            lab = lab[keep]
            x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
            if len(lab):
                lab[:, 1] = (x1 + x2) / 2 / S
                lab[:, 2] = (y1 + y2) / 2 / S
                lab[:, 3] = (x2 - x1) / S
                lab[:, 4] = (y2 - y1) / S
        return arr, lab

    def _augment(self, arr: np.ndarray, lab: np.ndarray, affine: bool = True):
        if affine and self.affine_scale + self.affine_translate > 0:
            arr, lab = self._affine(arr, lab)
        if random.random() < self.hflip:
            arr = arr[:, ::-1]
            if len(lab):
                lab[:, 1] = 1.0 - lab[:, 1]
        if random.random() < self.vflip:
            arr = arr[::-1]
            if len(lab):
                lab[:, 2] = 1.0 - lab[:, 2]
        if random.random() < self.rot90:
            k = random.choice([1, 2, 3])
            arr = np.rot90(arr, k)
            for _ in range(k):
                if len(lab):
                    cx, cy, w, h = lab[:, 1].copy(), lab[:, 2].copy(), \
                        lab[:, 3].copy(), lab[:, 4].copy()
                    # 90 deg counter-clockwise on a square canvas
                    lab[:, 1], lab[:, 2] = cy, 1.0 - cx
                    lab[:, 3], lab[:, 4] = h, w
        if random.random() < self.brightness:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-30, 30)
            arr = np.clip(arr.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr), lab

    def _load_resized(self, i: int):
        """One image as an uint8 array at ``imgsz`` plus its labels."""
        im = Image.open(self.img_files[i]).convert("RGB")
        lab = (self.labels[i] if self.labels is not None
               else self._load_label(self.lbl_files[i])).copy()
        w0, h0 = im.size
        if (w0, h0) != (self.imgsz, self.imgsz):
            im, r, (dx, dy) = letterbox(im, self.imgsz)
            if len(lab):
                lab[:, 1] = (lab[:, 1] * w0 * r + dx) / self.imgsz
                lab[:, 2] = (lab[:, 2] * h0 * r + dy) / self.imgsz
                lab[:, 3] = lab[:, 3] * w0 * r / self.imgsz
                lab[:, 4] = lab[:, 4] * h0 * r / self.imgsz
        return np.asarray(im, dtype=np.uint8), lab

    def _mosaic(self, i: int):
        """Stitch four images into a 2S canvas, then crop a random S window.

        Standard YOLO mosaic: it multiplies the effective number of training
        layouts, puts several defect classes in one image, and -- through the
        random crop -- gives scale augmentation for free.  On a 1440-image
        dataset this is usually worth several mAP points.
        """
        S = self.imgsz
        idxs = [i] + [random.randrange(len(self.img_files)) for _ in range(3)]
        canvas = np.full((2 * S, 2 * S, 3), 114, dtype=np.uint8)
        boxes = []  # absolute xyxy on the 2S canvas + class
        for k, idx in enumerate(idxs):
            arr, lab = self._load_resized(idx)
            ox, oy = (k % 2) * S, (k // 2) * S
            canvas[oy:oy + S, ox:ox + S] = arr
            if len(lab):
                cx, cy = lab[:, 1] * S + ox, lab[:, 2] * S + oy
                bw, bh = lab[:, 3] * S, lab[:, 4] * S
                boxes.append(np.stack([lab[:, 0], cx - bw / 2, cy - bh / 2,
                                       cx + bw / 2, cy + bh / 2], 1))
        box = (np.concatenate(boxes, 0) if boxes
               else np.zeros((0, 5), dtype=np.float32))

        # One affine on the 2S canvas, cropped to S -- the reference's
        # Mosaic + RandomPerspective.  The previous version cropped a window
        # and resized it with PIL, whose BILINEAR filter antialiases on
        # downscale; that smooths exactly the texture statistics crazing and
        # rolled-in_scale are recognised by, while validation images are never
        # downscaled.  cv2.warpAffine does not antialias, so the texture the
        # model trains on matches the texture it is tested on.
        f = random.uniform(1.0 - self.scale, 1.0 + self.scale)
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)
        # centre the 2S canvas, scale about it, then place the S window
        M = np.array([[f, 0.0, (0.5 + tx) * S - f * S],
                      [0.0, f, (0.5 + ty) * S - f * S]], dtype=np.float32)
        arr = cv2.warpAffine(canvas, M, (S, S), flags=cv2.INTER_LINEAR,
                             borderValue=(114, 114, 114))

        lab = np.zeros((0, 5), dtype=np.float32)
        if len(box):
            x1, y1, x2, y2 = box[:, 1], box[:, 2], box[:, 3], box[:, 4]
            nx1, nx2 = f * x1 + M[0, 2], f * x2 + M[0, 2]
            ny1, ny2 = f * y1 + M[1, 2], f * y2 + M[1, 2]
            area0 = ((nx2 - nx1) * (ny2 - ny1)).clip(1e-6)
            nx1, nx2 = nx1.clip(0, S), nx2.clip(0, S)
            ny1, ny2 = ny1.clip(0, S), ny2.clip(0, S)
            w, h = nx2 - nx1, ny2 - ny1
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            keep = ((w > 2) & (h > 2) & (w * h / area0 >= self.min_box_visibility)
                    & (ar < 100))
            if keep.any():
                lab = np.zeros((int(keep.sum()), 5), dtype=np.float32)
                lab[:, 0] = box[keep, 0]
                lab[:, 1] = (nx1[keep] + nx2[keep]) / 2 / S
                lab[:, 2] = (ny1[keep] + ny2[keep]) / 2 / S
                lab[:, 3] = w[keep] / S
                lab[:, 4] = h[keep] / S
        return arr, lab

    def __getitem__(self, i: int):
        if self.augment and self.mosaic > 0 and random.random() < self.mosaic:
            arr, lab = self._mosaic(i)
            arr, lab = self._augment(arr, lab, affine=False)
            img = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float().div_(255.0)
            if self.normalize:
                img = (img - self._mean) / self._std
            if len(lab):
                lab[:, 1:] = np.clip(lab[:, 1:], 0.0, 1.0)
                lab = lab[(lab[:, 3] > 1e-3) & (lab[:, 4] > 1e-3)]
            return img, torch.from_numpy(lab.astype(np.float32)), self.img_files[i]

        im = Image.open(self.img_files[i]).convert("RGB")
        lab = (self.labels[i] if self.labels is not None
               else self._load_label(self.lbl_files[i])).copy()

        w0, h0 = im.size
        if (w0, h0) != (self.imgsz, self.imgsz):
            im, r, (dx, dy) = letterbox(im, self.imgsz)
            if len(lab):
                lab[:, 1] = (lab[:, 1] * w0 * r + dx) / self.imgsz
                lab[:, 2] = (lab[:, 2] * h0 * r + dy) / self.imgsz
                lab[:, 3] = lab[:, 3] * w0 * r / self.imgsz
                lab[:, 4] = lab[:, 4] * h0 * r / self.imgsz

        arr = np.asarray(im, dtype=np.uint8)
        if self.augment:
            arr, lab = self._augment(arr, lab)

        img = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float().div_(255.0)
        if self.normalize:
            img = (img - self._mean) / self._std
        if len(lab):
            lab[:, 1:] = np.clip(lab[:, 1:], 0.0, 1.0)
            keep = (lab[:, 3] > 1e-3) & (lab[:, 4] > 1e-3)
            lab = lab[keep]
        return img, torch.from_numpy(lab.astype(np.float32)), self.img_files[i]

    # ------------------------------------------------------------------ #
    @staticmethod
    def collate_fn(batch):
        imgs, labels, paths = zip(*batch)
        targets = []
        for i, lab in enumerate(labels):
            if lab.numel():
                bi = torch.full((lab.shape[0], 1), float(i))
                targets.append(torch.cat([bi, lab], 1))
        targets = torch.cat(targets, 0) if targets else torch.zeros(0, 6)
        return torch.stack(imgs, 0), targets, list(paths)


def build_dataloader(root: str, split: str, imgsz: int = 640, batch_size: int = 8,
                     augment: bool = False, shuffle: bool = None,
                     workers: int = 4, classes: Sequence[str] = NEU_CLASSES,
                     **ds_kwargs) -> DataLoader:
    if not augment:            # validation never uses mosaic / scale jitter
        for k in ("mosaic", "scale", "translate", "hflip", "vflip", "rot90",
                  "brightness", "affine_scale", "affine_translate",
                  "min_box_visibility"):
            ds_kwargs.pop(k, None)
    ds = NEUDetDataset(root, split, imgsz=imgsz, augment=augment,
                       classes=classes, **ds_kwargs)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False, collate_fn=NEUDetDataset.collate_fn,
                      persistent_workers=workers > 0)
