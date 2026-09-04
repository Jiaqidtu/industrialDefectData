#!/bin/bash
# The reference baselines on the same 3-way split, same protocol.
#
# run_final.sh covered our two configurations; without the references on the
# identical split the comparison table has nothing to compare against.  Same
# discipline: train on train, pick the epoch on val, score once on test.
set -uo pipefail
cd "$(dirname "$0")"

DATA=neu-det-3way
YAML="$(pwd)/$DATA/neu-det-3way.yaml"
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-150}
IOU=${IOU:-0.45}
LOGDIR=runs/final_logs
mkdir -p "$LOGDIR"

finished () {
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys
try: sys.exit(0 if json.load(open(f"runs/{sys.argv[1]}/results.json"))["complete"] else 1)
except Exception: sys.exit(1)
PY
}

# ---- SAM2-Hiera：走我们自己的训练器 ---------------------------------- #
for s in $SEEDS; do
  name="f3_sam2_s${s}"
  if finished "$name"; then echo "[skip] $name"; continue; fi
  echo "[run ] $name  ($(date '+%H:%M:%S'))"
  python -m model.train --data "$DATA" --train-split train --val-split val \
      --backbone hiera --adapter-stages none --adapter-type conv \
      --epochs "$EPOCHS" --no-obj --mosaic 0 --batch-size 16 --workers 2 \
      --iou "$IOU" --eval-every 10 --verbose-eval --seed "$s" --name "$name" \
      > "$LOGDIR/$name.log" 2>&1
  finished "$name" || { echo "[FAIL] $name"; tail -5 "$LOGDIR/$name.log"; }
done

# ---- YOLOv8n / YOLOv8s：走 ultralytics 自己的训练流程 ----------------- #
for cfg in "f3_v8n:yolov8n" "f3_v8s:yolov8s"; do
  base="${cfg%%:*}"; arch="${cfg#*:}"
  for s in $SEEDS; do
    name="${base}_s${s}"
    if [ -f "runs/$name/weights/best.pt" ]; then echo "[skip] $name"; continue; fi
    echo "[run ] $name  ($(date '+%H:%M:%S'))"
    python ref_yolov8.py --data "$YAML" --model "${arch}.yaml" \
        --epochs "$EPOCHS" --mosaic 0 --workers 2 --seed "$s" --name "$name" \
        > "$LOGDIR/$name.log" 2>&1
    [ -f "runs/$name/weights/best.pt" ] || { echo "[FAIL] $name"; tail -5 "$LOGDIR/$name.log"; }
  done
done

# ---- 统一用我们的协议在 test 上评分 ----------------------------------- #
echo
echo "=== 在 test 上评分 ==="
for s in $SEEDS; do
  name="f3_sam2_s${s}"; out="runs/$name/test_iou${IOU}.json"
  [ -f "runs/$name/best.pt" ] && [ ! -f "$out" ] && \
    python -m model.val --weights "runs/$name/best.pt" --data "$DATA" \
        --split test --iou "$IOU" --json "$out" ${DROP:+--drop-weights} > "$LOGDIR/eval_$name.log" 2>&1 \
    && echo "[eval] $name" \
    || { echo "[FAIL] $name 评分失败"; tail -3 "$LOGDIR/eval_$name.log"; }
done
for cfg in "f3_v8n:yolov8n" "f3_v8s:yolov8s"; do
  base="${cfg%%:*}"; arch="${cfg#*:}"
  for s in $SEEDS; do
    name="${base}_s${s}"; out="runs/$name/test_iou${IOU}.json"
    [ -f "runs/$name/weights/best.pt" ] && [ ! -f "$out" ] && \
      python -m model.evalcheck --weights "runs/$name/weights/best.pt" \
          --arch "${arch}.yaml" --data "$DATA" --split test --iou "$IOU" \
          --json "$out" ${DROP:+--drop-weights} > "$LOGDIR/eval_$name.log" 2>&1 \
      && echo "[eval] $name" \
      || { echo "[FAIL] $name 评分失败"; tail -3 "$LOGDIR/eval_$name.log"; }
  done
done
echo
echo "汇总: ./report_all.sh"
