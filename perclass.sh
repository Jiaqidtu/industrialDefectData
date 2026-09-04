#!/bin/bash
# Per-class scores for every seed, at the tuned NMS threshold, then aggregate.
# The overall-mAP comparison is not significant; the question is whether SAM2's
# advantage on the two diffuse classes (crazing, rolled-in_scale) is real.
cd "$(dirname "$0")"
IOU="${IOU:-0.45}"
for r in u_nomosaic w_cnn_s1 w_cnn_s2 v2_ada_none w_ada_none_s1 w_ada_none_s2; do
    if [ ! -f "runs/$r/val$IOU.json" ]; then
        echo "评估 $r ..."
        python -m model.val --weights "runs/$r/best.pt" --iou "$IOU" \
            --json "runs/$r/val$IOU.json" >/dev/null 2>&1 \
            || echo "  $r 失败"
    fi
done
python3 - "$IOU" <<'PY'
import json, os, sys, statistics as st
iou = sys.argv[1]
groups = {"CNN 从零":    ["u_nomosaic", "w_cnn_s1", "w_cnn_s2"],
          "SAM2 冻结":   ["v2_ada_none", "w_ada_none_s1", "w_ada_none_s2"]}
data = {}
for label, runs in groups.items():
    per = {}
    for r in runs:
        f = f"runs/{r}/val{iou}.json"
        if not os.path.exists(f):
            continue
        for cls, v in json.load(open(f))["per_class"].items():
            per.setdefault(cls, []).append(v["AP50"])
    data[label] = per
classes = list(next(iter(data.values())).keys())
print(f"\n{'类别':<18}{'CNN 从零':>18}{'SAM2 冻结':>18}{'差':>9}{'t':>7}")
for c in classes:
    a, b = data["CNN 从零"].get(c, []), data["SAM2 冻结"].get(c, [])
    if len(a) < 2 or len(b) < 2:
        continue
    d = st.mean(b) - st.mean(a)
    se = (st.stdev(a)**2/len(a) + st.stdev(b)**2/len(b))**0.5
    t = d/se if se else float("inf")
    mark = " *" if abs(t) > 3.18 else ""
    print(f"{c:<18}{st.mean(a):.4f}±{st.stdev(a):.4f}"
          f"{st.mean(b):>11.4f}±{st.stdev(b):.4f}{d:>+9.4f}{t:>7.2f}{mark}")
print("\n* = |t| > 3.18 (df≈3, p<0.05)。逐类检验是 6 次比较，"
      "Bonferroni 校正后阈值应为 |t| > 4.86")
PY
