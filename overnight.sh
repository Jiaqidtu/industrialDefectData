#!/bin/bash
# Queue everything that is currently blocking a decision:
#   A. the over-prediction comparison (our weights vs the reference's) -- the
#      clean test of "duplicate detections are what caps crazing/rolled-in",
#      run through one diagnosis with only the weights changed
#   B. seeds 1 and 2 for {SAM2 no-adapter, SAM2 all-adapters, CNN}, so the
#      0.015 spread in the adapter ablation can be told from noise
# Every job writes under runs/<name>/ and survives a dropped connection.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

# ---- A. diagnosis comparison (one short GPU job) ---------------------- #
OUT="$HERE/runs/diagnose_compare"
mkdir -p "$OUT"
bsub <<LSF
#BSUB -J diag_compare
#BSUB -q gpua10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -o $OUT/bsub.out
#BSUB -e $OUT/bsub.err

source ~/miniconda3/bin/activate
conda activate defect
cd $HERE
echo "===== 我们的实现 u_nomosaic (mAP50 0.6982) ====="
python -m model.diagnose --weights runs/u_nomosaic/best.pt 2>&1 \
    | tee $OUT/ours.txt
echo
echo "===== 参考实现 ultra_scratch (mAP50 0.7135) ====="
python -m model.diagnose --weights runs/ultra_scratch/weights/best.pt 2>&1 \
    | tee $OUT/reference.txt
LSF

# ---- B. seeds ---------------------------------------------------------- #
SAM2="--backbone hiera --adapter-type conv --epochs 150 --no-obj \
      --batch-size 16 --eval-every 10"
CNN="--backbone cnn --epochs 150 --no-obj --batch-size 16 --eval-every 10"
# All four adapter placements get seeds, not just the two extremes: the
# trade-off figure needs error bars on every point, otherwise a flat curve
# cannot be told from a noisy one.
for s in 1 2; do
    MEM=16GB WALL=4:00 ./submit.sh w_ada_none_s$s $SAM2 --adapter-stages none    --seed $s
    MEM=16GB WALL=4:00 ./submit.sh w_ada_4_s$s    $SAM2 --adapter-stages 4       --seed $s
    MEM=16GB WALL=4:00 ./submit.sh w_ada_34_s$s   $SAM2 --adapter-stages 3,4     --seed $s
    MEM=16GB WALL=4:00 ./submit.sh w_ada_1234_s$s $SAM2 --adapter-stages 1,2,3,4 --seed $s
    MEM=16GB WALL=4:00 ./submit.sh w_cnn_s$s      $CNN  --seed $s
done

echo
echo "全部已提交。醒来后："
echo "  bjobs                                  # 剩余作业"
echo "  cat runs/diagnose_compare/ours.txt      # 我们的过预测倍数"
echo "  cat runs/diagnose_compare/reference.txt # 参考的过预测倍数"
echo "  ./summarize.sh                          # 种子汇总"
