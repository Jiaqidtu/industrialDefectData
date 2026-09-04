#!/bin/bash
# One table: every configuration on the held-out test split, same protocol.
cd "$(dirname "$0")"
python3 - "${IOU:-0.45}" <<'PY'
import json, os, sys, statistics as st
iou = sys.argv[1]
GROUPS = [("频域双路(本文)", "f3_freq"), ("CNN 基线", "f3_cnn"),
          ("SAM2-Hiera 冻结", "f3_sam2"), ("YOLOv8n 从零", "f3_v8n"),
          ("YOLOv8s 从零", "f3_v8s")]
NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
         "rolled-in_scale", "scratches"]

data = {}
for label, base in GROUPS:
    runs = [json.load(open(f)) for f in
            (f"runs/{base}_s{s}/test_iou{iou}.json" for s in (0, 1, 2))
            if os.path.exists(f)]
    if runs:
        data[label] = runs

if not data:
    print("还没有任何 test 评分结果"); raise SystemExit

print(f"===== 测试集 (270 张, NMS IoU {iou}) =====\n")
print(f"{'配置':<20}{'n':>3}{'mAP50':>19}{'mAP50-95':>19}")
for label, runs in data.items():
    a = [r["mAP50"] for r in runs]; b = [r["mAP50-95"] for r in runs]
    sa = st.stdev(a) if len(a) > 1 else 0.0
    sb = st.stdev(b) if len(b) > 1 else 0.0
    print(f"{label:<20}{len(a):>3}{st.mean(a):>12.4f}±{sa:.4f}{st.mean(b):>12.4f}±{sb:.4f}")

ref = data.get("CNN 基线")
if ref and len(ref) > 1:
    b = [r["mAP50"] for r in ref]
    print(f"\n{'对比 CNN 基线':<20}{'差':>10}{'t':>8}   (临界 3.18)")
    for label, runs in data.items():
        if label == "CNN 基线" or len(runs) < 2:
            continue
        a = [r["mAP50"] for r in runs]
        sa = st.stdev(a)
        d = st.mean(a) - st.mean(b)
        if sa < 1e-9:
            print(f"{label:<20}{d:>+10.4f}{'n/a':>8}   种子方差为 0，无法检验")
            continue
        se = (sa**2/len(a) + st.stdev(b)**2/len(b))**0.5
        t = d/se if se else float("nan")
        print(f"{label:<20}{d:>+10.4f}{t:>8.2f}   "
              f"{'显著' if abs(t) > 3.18 else '不显著'}")

print(f"\n{'类别':<18}" + "".join(f"{l[:14]:>17}" for l in data))
for c in NAMES:
    row = f"{c:<18}"
    for label, runs in data.items():
        v = [r["per_class"][c]["AP50"] for r in runs if c in r["per_class"]]
        row += (f"{st.mean(v):>11.4f}±{st.stdev(v) if len(v)>1 else 0:.4f}"
                if v else f"{'-':>17}")
    print(row)

# a zero standard deviation means the "seeds" were the same run repeated;
# including it would deflate the pooled estimate and inflate every t
sds = [x for x in (st.stdev([r["mAP50"] for r in v])
                   for v in data.values() if len(v) > 1) if x > 1e-9]
if sds:
    sd = (sum(x**2 for x in sds)/len(sds))**0.5
    print(f"\n合并种子标准差 {sd:.4f}；n=3 时该测试集可分辨的最小效应 "
          f"{2.9*sd*(2/3)**0.5:.4f}")
    print("小于该值的组间差异不应解读为方法差异。")
PY
