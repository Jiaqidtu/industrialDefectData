#!/bin/bash
# Submit every run still missing from the paper's tables, then go to sleep.
#
#   ./run_overnight.sh          # submit what is missing
#   ./run_overnight.sh --dry    # list what it would submit, submit nothing
#
# Runs that already carry a complete results.json (ours) or a finished
# results.csv (ultralytics) are skipped, so this is safe to re-run: it will
# only ever fill gaps.  That matters because some of these may already be
# running interactively on the a100.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
Q="${QUEUE:-gpul40s}"

done_already () {                     # $1 = run name
    local r="runs/$1"
    # ours: train.py writes complete=true only after the final epoch
    if [ -f "$r/results.json" ] && \
       grep -q '"complete": *true' "$r/results.json" 2>/dev/null; then return 0; fi
    # ultralytics: the last results.csv row is epoch == requested epochs
    if [ -f "$r/results.csv" ] && \
       [ "$(tail -1 "$r/results.csv" | cut -d, -f1)" = "150" ]; then return 0; fi
    return 1
}

submit () {                            # $1 = name, rest = args, RUNNER from env
    local name="$1"; shift
    if done_already "$name"; then echo "  跳过 $name (已完成)"; return; fi
    if bjobs -w 2>/dev/null | grep -q "[[:space:]]$name[[:space:]]"; then
        echo "  跳过 $name (已在队列中)"; return
    fi
    if [ $DRY = 1 ]; then echo "  会提交 $name: $*"; return; fi
    QUEUE="$Q" WALL=12:00 MEM=32GB ./submit.sh "$name" "$@" >/dev/null \
        && echo "  已提交 $name" || echo "  !! 提交失败 $name"
}

NEU=neu-det-3way/neu-det-3way.yaml
OURS="--data neu-det-3way --train-split train --val-split val"
GC="--data gc10-3way --train-split train --val-split val"

echo "== ① NEU-DET 主表: YOLOv8 补种子 (640px) =="
for s in 1 2; do
    RUNNER=ref submit f3_v8n_s$s --model yolov8n.pt --data $NEU \
        --epochs 150 --imgsz 640 --batch 16 --seed $s
    RUNNER=ref submit f3_v8s_s$s --model yolov8s.pt --data $NEU \
        --epochs 150 --imgsz 640 --batch 16 --seed $s
done

echo "== ② GC10-DET 跨数据集: 补第三个种子 =="
submit g_freq_s2 --backbone freqdual $GC --epochs 150 --imgsz 640 --seed 2
submit g_cnn_s2  --backbone cnn      $GC --epochs 150 --imgsz 640 --seed 2

echo "== ③ MPS 消融 (256px, 只在 NEU-DET) =="
for s in 0 1 2; do
    submit r256mps2_s$s --backbone freqdual $OURS --imgsz 256 --epochs 150 \
        --mps-bond 16 --seed $s
done

echo
echo "队列: $Q   查看: bjobs"
echo "明早汇总: ./report_all.sh"
