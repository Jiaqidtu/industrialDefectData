#!/bin/bash
# One table of every finished run, with seed means where seeds exist.
cd "$(dirname "$0")"
python3 - <<'PY'
import json, os, glob, statistics as st

def load(name):
    f = f"runs/{name}/results.json"
    if not os.path.exists(f):
        return None
    return json.load(open(f))


def score(d, name):
    """Prefer the re-scored file: results.json carries whatever --iou that run
    was trained with, and the default changed from 0.7 to 0.45 partway through,
    so mixing them silently hands the later runs a ~0.035 advantage."""
    f = f"runs/{name}/val0.45.json"
    if os.path.exists(f):
        return json.load(open(f))["mAP50"], 0.45
    return d["best"]["mAP50"], d["args"].get("iou")

groups = {
    "SAM2 无 adapter":  ["v2_ada_none", "w_ada_none_s1", "w_ada_none_s2"],
    "SAM2 adapter@4":   ["v2_ada_4", "w_ada_4_s1", "w_ada_4_s2"],
    "SAM2 adapter@3,4": ["v2_ada_34", "w_ada_34_s1", "w_ada_34_s2"],
    "SAM2 全 adapter":  ["v2_ada_1234", "w_ada_1234_s1", "w_ada_1234_s2"],
    "CNN 从零":         ["u_nomosaic", "w_cnn_s1", "w_cnn_s2"],
}
print(f"{'配置':<20}{'n':>3}{'mAP50 均值':>12}{'标准差':>9}{'各次':>28}")
for label, names in groups.items():
    ds = [d for d in (load(n) for n in names) if d and d["complete"]]
    if not ds:
        pend = [n for n in names if not (load(n) or {}).get("complete")]
        print(f"{label:<20}  未完成: {' '.join(pend)}")
        continue
    scored = [score(d, n) for d, n in zip(ds, [n for n in names if load(n)
                                                and load(n)["complete"]])]
    v = [x for x, _ in scored]
    ious = {i for _, i in scored}
    if len(ious) > 1:
        print(f"{label:<20}  !! NMS 口径不一致 {sorted(ious)}，不可比")
        continue
    sd = st.stdev(v) if len(v) > 1 else float("nan")
    print(f"{label:<20}{len(v):>3}{st.mean(v):>12.4f}{sd:>9.4f}"
          f"   {' '.join(f'{x:.4f}' for x in v):>25}")
print("\n参考: YOLOv8n 从零 0.7135 (+mosaic) | YOLO26n 端到端 0.6602")
print("判读: 若种子标准差 >= 0.015，adapter 各组之间的差异就是噪声")
PY
