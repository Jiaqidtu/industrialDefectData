#!/bin/bash
# Re-score every finished checkpoint at the tuned NMS threshold. No retraining.
cd "$(dirname "$0")"
IOU="${IOU:-0.45}"
for r in u_nomosaic w_cnn_s1 w_cnn_s2 \
         v2_ada_1234 w_ada_1234_s1 w_ada_1234_s2 \
         v2_ada_none w_ada_none_s1 w_ada_none_s2; do
    [ -f "runs/$r/best.pt" ] || { printf "%-18s (无权重)\n" "$r"; continue; }
    log=$(python -m model.val --weights "runs/$r/best.pt" --iou "$IOU" 2>&1)
    m=$(printf '%s' "$log" | awk '/^all/ {print $4}')
    if [ -z "$m" ]; then
        printf "%-18s 失败: %s\n" "$r" "$(printf '%s' "$log" | tail -3 | tr '\n' ' ')"
    else
        printf "%-18s mAP50@%.2f = %s\n" "$r" "$IOU" "$m"
    fi
done
