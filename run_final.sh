#!/bin/bash
# One command for the final, protocol-clean numbers.
#
#   ./run_final.sh              # train what is missing, then score on test
#   ./run_final.sh --report     # skip training, just re-print the table
#
# Trains on the 3-way split's train (1260), selects on val (270), and reports
# once on test (270).  Runs that already finished are skipped, so the script is
# safe to re-run after an interruption.
set -uo pipefail
cd "$(dirname "$0")"

DATA=neu-det-3way
SEEDS="0 1 2"
EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-16}
IOU=${IOU:-0.45}
LOGDIR=runs/final_logs
mkdir -p "$LOGDIR"

# name:extra-args
CONFIGS=(
  "f3_freq:--backbone freqdual"
  "f3_cnn:--backbone cnn"
)

finished () {   # $1 = run name
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys
try: sys.exit(0 if json.load(open(f"runs/{sys.argv[1]}/results.json"))["complete"] else 1)
except Exception: sys.exit(1)
PY
}

if [ "${1:-}" != "--report" ]; then
  for cfg in "${CONFIGS[@]}"; do
    base="${cfg%%:*}"; extra="${cfg#*:}"
    for s in $SEEDS; do
      name="${base}_s${s}"
      if finished "$name"; then
        echo "[skip] $name 已完成"
        continue
      fi
      echo "[run ] $name  ($(date '+%H:%M:%S'))"
      python -m model.train --data "$DATA" \
          --train-split train --val-split val \
          $extra --epochs "$EPOCHS" --no-obj --mosaic 0 \
          --batch-size "$BATCH" --workers 2 --iou "$IOU" \
          --eval-every 10 --verbose-eval --seed "$s" --name "$name" \
          > "$LOGDIR/$name.log" 2>&1
      if ! finished "$name"; then
        echo "[FAIL] $name 未完成，详见 $LOGDIR/$name.log"
        tail -5 "$LOGDIR/$name.log"
      fi
    done
  done

  echo
  echo "=== 在 test 上评分（每个模型只测一次）==="
  for cfg in "${CONFIGS[@]}"; do
    base="${cfg%%:*}"
    for s in $SEEDS; do
      name="${base}_s${s}"
      out="runs/$name/test_iou${IOU}.json"
      [ -f "runs/$name/best.pt" ] || { echo "[skip] $name 无权重"; continue; }
      [ -f "$out" ] && { echo "[skip] $name 已评分"; continue; }
      python -m model.val --weights "runs/$name/best.pt" --data "$DATA" \
          --split test --iou "$IOU" --json "$out" ${DROP:+--drop-weights} > /dev/null 2>&1 \
          && echo "[eval] $name" || echo "[FAIL] $name 评分失败"
    done
  done
fi

echo
python3 - "$IOU" <<'PY'
import json, os, sys, statistics as st
iou = sys.argv[1]
groups = {"频域双路": "f3_freq", "CNN 基线": "f3_cnn"}
NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
         "rolled-in_scale", "scratches"]

data = {}
for label, base in groups.items():
    runs = []
    for s in (0, 1, 2):
        f = f"runs/{base}_s{s}/test_iou{iou}.json"
        if os.path.exists(f):
            runs.append(json.load(open(f)))
    if runs:
        data[label] = runs

if not data:
    print("还没有 test 评分结果")
    raise SystemExit

print(f"===== 测试集结果 (n={len(next(iter(data.values())))}, NMS IoU {iou}) =====\n")
print(f"{'配置':<12}{'mAP50':>18}{'mAP50-95':>18}")
for label, runs in data.items():
    a = [r["mAP50"] for r in runs]; b = [r["mAP50-95"] for r in runs]
    sa = st.stdev(a) if len(a) > 1 else 0.0
    sb = st.stdev(b) if len(b) > 1 else 0.0
    print(f"{label:<12}{st.mean(a):>11.4f}±{sa:.4f}{st.mean(b):>11.4f}±{sb:.4f}")

if len(data) == 2:
    (la, ra), (lb, rb) = data.items()
    a = [r["mAP50"] for r in ra]; b = [r["mAP50"] for r in rb]
    if len(a) > 1 and len(b) > 1:
        se = (st.stdev(a) ** 2 / len(a) + st.stdev(b) ** 2 / len(b)) ** 0.5
        d = st.mean(a) - st.mean(b)
        t = d / se if se else float("nan")
        print(f"\n{la} vs {lb}: {d:+.4f}   t = {t:.2f}   (df≈3, 临界 3.18) -> "
              f"{'显著' if abs(t) > 3.18 else '不显著'}")

print(f"\n{'类别':<18}" + "".join(f"{l:>20}" for l in data))
for c in NAMES:
    row = f"{c:<18}"
    for label, runs in data.items():
        v = [r["per_class"][c]["AP50"] for r in runs if c in r["per_class"]]
        if v:
            sd = st.stdev(v) if len(v) > 1 else 0.0
            row += f"{st.mean(v):>13.4f}±{sd:.4f}"
        else:
            row += f"{'-':>20}"
    print(row)

print("\n注意: test 只应评分一次。若据此结果再调整任何配置，"
      "它就变成了第二个验证集。")
PY
