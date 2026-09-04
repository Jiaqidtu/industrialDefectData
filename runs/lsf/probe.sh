#!/bin/bash
#BSUB -q gpua100
#BSUB -J probe_gpu
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:15
#BSUB -o /zhome/9e/5/158608/industrialDefectData/runs/lsf/probe.%J.out
#BSUB -e /zhome/9e/5/158608/industrialDefectData/runs/lsf/probe.%J.err
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
