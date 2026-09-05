#!/bin/bash
# The pre-registered test for the MPS ablation.  Runs only when both groups
# are complete, so it cannot be peeked at early and stopped on a good seed.
cd "$(cd "$(dirname "$0")" && pwd)"
source ~/miniconda3/bin/activate && conda activate defect
python - <<'PY'
import json, math, statistics as st, sys

def load(n):
    try: d = json.load(open(f"runs/{n}/results.json"))
    except FileNotFoundError: return None
    return d["best"] if d.get("complete") else None

A = [load(f"r256_freq_s{i}")  for i in range(7)]
B = [load(f"r256mps2_s{i}")   for i in range(7)]
missing = [f"r256_freq_s{i}" for i, x in enumerate(A) if x is None] + \
          [f"r256mps2_s{i}"  for i, x in enumerate(B) if x is None]
if missing:
    sys.exit(f"还缺 {len(missing)} 个: {', '.join(missing)}\n"
             f"检验要等两组都满 7 个种子再做 -- 中途看结果就是在挑停手时机。")

def welch(a, b):
    na, nb = len(a), len(b); va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va/na + vb/nb)
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    return (st.mean(b) - st.mean(a)) / se, df

print(f"{'指标':<18}{'freqdual (n=7)':>21}{'+MPS (n=7)':>21}{'Δ':>9}{'t':>8}{'df':>6}")
res = {}
for name, c in [("mAP50", None), ("mAP50-95", None)] + \
               [(k, k) for k in A[0]["per_class"]]:
    a = [x["per_class"][c]["AP50"] if c else x[name] for x in A]
    b = [x["per_class"][c]["AP50"] if c else x[name] for x in B]
    t, df = welch(a, b); res[name] = (st.mean(b) - st.mean(a), t)
    print(f"{name:<18}{st.mean(a):>14.4f}±{st.stdev(a):.4f}"
          f"{st.mean(b):>14.4f}±{st.stdev(b):.4f}"
          f"{st.mean(b)-st.mean(a):>+9.4f}{t:>8.2f}{df:>6.1f}"
          f"{'  ***' if abs(t) > 3.18 else ''}")

d_cr, t_cr = res["crazing"]; d_all, _ = res["mAP50"]
print("\n预先登记的判据: crazing |t| > 3.18 且 mAP50 不下降")
print(f"  crazing t = {t_cr:.2f}   mAP50 Δ = {d_all:+.4f}")
print("  ->", "正结果，写进论文" if abs(t_cr) > 3.18 and t_cr > 0 and d_all >= 0
      else "否定结果，MPS 归档；不再加种子")
PY
