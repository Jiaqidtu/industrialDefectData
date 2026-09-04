#!/bin/bash
### One YOLO-SAM2 training run on a single A100.
###
###   bsub < model/hpc/train_a100.sh                       # defaults
###   STAGES=3,4 NAME=s34 bsub < model/hpc/train_a100.sh   # one ablation point
###
#BSUB -q gpua100
#BSUB -J yolosam2
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -o runs/lsf/train.%J.out
#BSUB -e runs/lsf/train.%J.err

source "$HOME/industrialDefectData/model/hpc/env.sh"

# ---- knobs (override by exporting before bsub) --------------------------- #
VARIANT=${VARIANT:-tiny}
STAGES=${STAGES:-1,2,3,4}          # the branch add/remove ablation variable
NECK=${NECK:-panet}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-16}
IMGSZ=${IMGSZ:-640}
LR=${LR:-1e-3}
SEED=${SEED:-0}
NAME=${NAME:-${VARIANT}_s${STAGES//,/}_${NECK}_seed${SEED}}
SAM2_CKPT=${SAM2_CKPT:-$HOME/sam2/checkpoints/sam2.1_hiera_tiny.pt}

CKPT_ARG=""
if [ -f "$SAM2_CKPT" ]; then
    CKPT_ARG="--sam2-ckpt $SAM2_CKPT"
else
    echo "WARNING: no SAM2 checkpoint at $SAM2_CKPT -- the Hiera trunk will be"
    echo "         randomly initialised and the accuracy will be meaningless."
fi

python3 -m model.train \
    --data neu-det-yolo --variant "$VARIANT" --adapter-stages "$STAGES" \
    --neck "$NECK" --epochs "$EPOCHS" --batch-size "$BATCH" --imgsz "$IMGSZ" \
    --lr "$LR" --seed "$SEED" --workers 8 --eval-every 5 --verbose-eval \
    --project runs --name "$NAME" $CKPT_ARG
