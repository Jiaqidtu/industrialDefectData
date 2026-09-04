#!/bin/bash
# Adapter-placement ablation: the paper's core trade-off figure.
# Four runs differing only in which Hiera stages carry an adapter.
set -euo pipefail
cd "$(dirname "$0")"
COMMON="--backbone hiera --adapter-type conv --epochs 150 --no-obj \
        --batch-size 16 --eval-every 10"
MEM=16GB WALL=4:00 ./submit.sh v2_ada_none  $COMMON --adapter-stages none
MEM=16GB WALL=4:00 ./submit.sh v2_ada_4     $COMMON --adapter-stages 4
MEM=16GB WALL=4:00 ./submit.sh v2_ada_34    $COMMON --adapter-stages 3,4
MEM=16GB WALL=4:00 ./submit.sh v2_ada_1234  $COMMON --adapter-stages 1,2,3,4
