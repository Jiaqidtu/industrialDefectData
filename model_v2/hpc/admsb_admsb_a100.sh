#!/bin/bash
### ADMSB A/B -- treatment: ADMSB in P3/P4
###
###   bsub < model_v2/hpc/admsb_admsb_a100.sh
###
### The arm is HARDCODED below on purpose.  An earlier version selected it from
### $ARM at submit time; LSF did not forward the variable, so both arms ran as
### 'admsb' into the same output directory and one job crashed on a checkpoint
### race.  Two separate scripts remove that failure mode entirely.
###
#BSUB -q gpua100
#BSUB -J admsb_admsb
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -o runs/lsf/admsb_admsb.%J.out
#BSUB -e runs/lsf/admsb_admsb.%J.err

source "$HOME/industrialDefectData/model_v2/hpc/env.sh"

ARM=admsb
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-16}
IMGSZ=${IMGSZ:-640}
LR=${LR:-1e-3}

CFG="model_v2/configs/cnn_${ARM}.yaml"
NAME="admsb_${ARM}_seed${SEED}"

# refuse to clobber a finished run -- this is what silently corrupted the
# first attempt.  Move or delete the old directory if you meant to redo it.
if [ -e "runs/$NAME/results.json" ]; then
    echo "FATAL: runs/$NAME already holds a finished run; move it aside first"
    exit 1
fi

echo "arm=$ARM  cfg=$CFG  seed=$SEED  -> runs/$NAME"

python3 -m model_v2.train \
    --data neu-det-yolo --cfg "$CFG" --backbone cnn \
    --epochs "$EPOCHS" --batch-size "$BATCH" --imgsz "$IMGSZ" \
    --lr "$LR" --seed "$SEED" --workers 8 --eval-every 5 --verbose-eval \
    --project runs --name "$NAME"
