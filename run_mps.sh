#!/bin/bash
# Finish the MPS ablation: bring the +MPS group to 7 seeds so it matches the
# 7 seeds r256_freq already has.
#
#   ./run_mps.sh              # submit seeds 3-6 to gpua100
#   QUEUE=gpul40s ./run_mps.sh
#   ./run_mps.sh --dry
#
# n=7 comes from the observed effect on crazing (delta 0.049, pooled sd 0.031,
# d ~ 1.57): n = 16/d^2 ~ 7 per group for 80% power.  n=3 could only have
# resolved d > 2.6.  The decision rule is fixed BEFORE these run:
#
#   crazing |t| > 3.18 AND overall mAP50 not lower  -> positive result
#   otherwise                                       -> archived as null
#
# No seeds get added after this either way.
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
Q="${QUEUE:-gpua100}"

pend=$(bqueues "$Q" 2>/dev/null | awk 'NR==2{print $8}')
echo "队列 $Q: $pend 个任务在排队"
[ "${pend:-0}" -gt 100 ] && echo "  !! 排队很长，考虑 QUEUE=gpul40s ./run_mps.sh"

for s in 3 4 5 6; do
    name="r256mps2_s$s"
    if [ -f "runs/$name/results.json" ] && \
       grep -q '"complete": *true' "runs/$name/results.json" 2>/dev/null; then
        echo "  跳过 $name (已完成)"; continue
    fi
    if bjobs -w 2>/dev/null | grep -q "[[:space:]]$name[[:space:]]"; then
        echo "  跳过 $name (已在队列中)"; continue
    fi
    if [ $DRY = 1 ]; then echo "  会提交 $name"; continue; fi
    QUEUE="$Q" WALL=4:00 MEM=16GB ./submit.sh "$name" \
        --backbone freqdual --data neu-det-3way \
        --train-split train --val-split val \
        --imgsz 256 --epochs 150 --mps-bond 16 --seed "$s" >/dev/null \
        && echo "  已提交 $name" || echo "  !! 提交失败 $name"
done

echo
echo "查看: bjobs        汇总: ./mps_test.sh"
