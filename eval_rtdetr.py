"""Score a trained RT-DETR run on an arbitrary split of the 3-way dataset.

Separate from ref_yolov8.py because the training run for f3_rtdetr_s0 lost its
EMA to a NaN at epoch 107: the val metrics in results.csv read 0 from there on
while the raw weights kept training, so the numbers in that file cannot be
trusted and best.pt has to be re-scored from disk.
"""
import argparse, json, os, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--data", default="neu-det-3way/neu-det-3way.yaml")
ap.add_argument("--split", default="test")
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()

from ultralytics import RTDETR, YOLO
sd = torch.load(a.weights, map_location="cpu", weights_only=False)
bad = [n for n, p in sd["model"].state_dict().items() if not torch.isfinite(p).all()]
print(f"[check] {len(bad)} non-finite tensors in {a.weights}")
if bad:
    print("        first few:", bad[:5])
    sys.exit(f"{a.weights} 已被 NaN 污染，不能用于评分")

name = os.path.basename(str(sd["model"].yaml.get("yaml_file", "")) or a.weights)
is_rtdetr = "rtdetr" in name.lower() or "rtdetr" in a.weights.lower()
m = (RTDETR if is_rtdetr else YOLO)(a.weights)
r = m.val(data=a.data, split=a.split, imgsz=a.imgsz, batch=16, workers=2,
          device=0, plots=False, project="runs", name=f"eval_{a.split}",
          exist_ok=True)
names = m.names
out = {"weights": a.weights, "split": a.split,
       "mAP50": float(r.box.map50), "mAP50-95": float(r.box.map),
       "per_class": {}}
print(f"\n{'class':<18}{'AP50':>9}{'AP50-95':>10}")
for i, c in enumerate(r.box.ap_class_index):
    n = names[int(c)]
    out["per_class"][n] = {"AP50": float(r.box.ap50[i]), "AP50-95": float(r.box.ap[i])}
    print(f"{n:<18}{r.box.ap50[i]:>9.4f}{r.box.ap[i]:>10.4f}")
print(f"{'all':<18}{r.box.map50:>9.4f}{r.box.map:>10.4f}")
dst = os.path.join(os.path.dirname(os.path.dirname(a.weights)),
                   f"{a.split}_ultra.json")
json.dump(out, open(dst, "w"), indent=1)
print("->", dst)
