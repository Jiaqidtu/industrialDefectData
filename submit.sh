#!/bin/bash
# Submit a training run to LSF so it survives a dropped SSH connection.
#
#   ./submit.sh u_nomosaic --backbone cnn --epochs 150 --no-obj --mosaic 0
#
# The first argument is the run name; everything after it is passed through to
# model.train.  stdout lands in runs/<name>/bsub.out, and train.py itself
# rewrites runs/<name>/results.json and train.log every epoch, so a job that
# dies part-way still leaves a usable record.
set -euo pipefail

NAME="${1:?用法: ./submit.sh <run-name> [train.py 的参数...]}"
shift
QUEUE="${QUEUE:-gpua10}"       # gpua10 排队最短; gpua100 更快但常排 1000+
WALL="${WALL:-4:00}"
MEM="${MEM:-16GB}"             # SAM2 (31M 参数 / 100 GFLOPs) 用 8GB 会被静默杀掉
HERE="$(cd "$(dirname "$0")" && pwd)"
# RUNNER=ref 跑 ultralytics 参照脚本，默认跑我们自己的 train
if [ "${RUNNER:-train}" = "ref" ]; then
    RUNCMD="python ref_yolov8.py --workers 2 --name $NAME $*"
else
    RUNCMD="python -m model.train --data neu-det-yolo --workers 2 --verbose-eval --name $NAME $*"
fi
mkdir -p "$HERE/runs/$NAME"

bsub <<LSF
#BSUB -J $NAME
#BSUB -q $QUEUE
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=$MEM]"
#BSUB -M $MEM
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W $WALL
#BSUB -o $HERE/runs/$NAME/bsub.out
#BSUB -e $HERE/runs/$NAME/bsub.err

# torch 自带 CUDA runtime，不需要 module load cuda（该 modulefile 也不存在）
source ~/miniconda3/bin/activate
conda activate defect
cd $HERE
$RUNCMD
LSF
echo "已提交 $NAME (队列 $QUEUE, 内存 $MEM, 时限 $WALL)"
echo "  状态: bjobs      日志: tail -f runs/$NAME/train.log"
echo "  结束后查退出原因: grep -A6 'Resource usage' runs/$NAME/bsub.out"
