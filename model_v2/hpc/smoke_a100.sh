#!/bin/bash
### 15-minute GPU sanity check: selftest + 2 training epochs + eval + FPS.
#BSUB -q gpua100
#BSUB -J yolosam2v2_smoke
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o runs/lsf/v2_smoke.%J.out
#BSUB -e runs/lsf/v2_smoke.%J.err

source "$HOME/industrialDefectData/model_v2/hpc/env.sh"

SAM2_CKPT=${SAM2_CKPT:-$HOME/sam2/checkpoints/sam2.1_hiera_tiny.pt}
CKPT_ARG=""
[ -f "$SAM2_CKPT" ] && CKPT_ARG="--sam2-ckpt $SAM2_CKPT"

python3 -m model_v2.selftest --imgsz 640
python3 -m model_v2.train --data neu-det-yolo --epochs 2 --batch-size 16 \
    --imgsz 640 --workers 4 --eval-every 1 --verbose-eval \
    --project runs --name v2_gpu_smoke $CKPT_ARG
python3 -c "
from model import build_model
m = build_model('model_v2/configs/yolo_sam2_tiny.yaml').cuda()
print('params', m.param_counts())
print('GFLOPs', m.flops(640))
for b in (1, 8):
    print(f'FPS batch={b}:', m.benchmark_fps(640, batch=b))
"
