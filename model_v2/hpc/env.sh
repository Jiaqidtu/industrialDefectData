#!/bin/bash
# Shared environment for every LSF job of this project.
#
# NOTE: v2 needs torchvision (ADMSB uses torchvision.ops.deform_conv2d), which
# the cluster's system Python 3.9 does not have.  So unlike v1, this env.sh
# activates the `defect` conda env -- same sequence as guide.md.
#
# No `module load cuda/...` here: this cluster has no such modulefile, and the
# conda env's torch ships its own CUDA 12.8 runtime (nvidia-smi + torch.cuda
# confirm the GPU works without it).

source "$HOME/miniconda3/bin/activate"
conda activate defect

export PROJECT_DIR="$HOME/industrialDefectData"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_DIR" || exit 1

echo "host      : $(hostname)"
echo "date      : $(date)"
echo "python    : $(which python3) $(python3 -V 2>&1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# fail fast and loudly if torchvision is missing -- ADMSB cannot run without it
python3 -c "import torchvision; from torchvision.ops import deform_conv2d; print('torchvision', torchvision.__version__)" || {
    echo "FATAL: torchvision missing in this env -- model_v2 needs deform_conv2d"
    exit 1
}
