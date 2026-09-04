#!/bin/bash
### Adapter on/off sweep (the core experiment) on a single A100.
###
###   STRATEGY=add SEEDS=0,1,2 bsub < model_v2/hpc/ablation_a100.sh
###
### One job runs every configuration sequentially, so ask for enough wall time:
### `add` = 5 configs x len(SEEDS) runs.  Submit one job per seed instead if
### you would rather spread it over several nodes.
#BSUB -q gpua100
#BSUB -J yolosam2v2_abl
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o runs/lsf/v2_ablation.%J.out
#BSUB -e runs/lsf/v2_ablation.%J.err

source "$HOME/industrialDefectData/model_v2/hpc/env.sh"

STRATEGY=${STRATEGY:-add}
SEEDS=${SEEDS:-0}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-16}
VARIANT=${VARIANT:-tiny}
SAM2_CKPT=${SAM2_CKPT:-$HOME/sam2/checkpoints/sam2.1_hiera_tiny.pt}

CKPT_ARG=""
[ -f "$SAM2_CKPT" ] && CKPT_ARG="--sam2-ckpt $SAM2_CKPT" \
    || echo "WARNING: no SAM2 checkpoint at $SAM2_CKPT (random trunk)"

python3 -m model_v2.ablation \
    --strategy "$STRATEGY" --seeds "$SEEDS" --variant "$VARIANT" \
    --data neu-det-yolo --epochs "$EPOCHS" --batch-size "$BATCH" \
    --workers 8 --eval-every 10 --project "runs/v2_ablation_${STRATEGY}" $CKPT_ARG
