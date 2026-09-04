#!/bin/bash
# Shared environment for every LSF job of this project.
#
# NOTE: the conda env `node_env` mentioned in agent_quantitative_trade/readme.md
# is a Node.js environment (it is what runs the claude CLI) and contains no
# Python.  The detector runs on the cluster's system Python 3.9, which already
# has torch 2.8.0+cu128 installed under ~/.local -- no conda activation needed.
# Uncomment the two conda lines below only if you later create a Python env.

# source "$HOME/miniconda3/bin/activate"
# conda activate <your-python-env>

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
