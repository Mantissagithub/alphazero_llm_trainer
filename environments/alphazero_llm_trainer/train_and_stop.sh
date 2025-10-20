#!/bin/bash
set -e

python train_vllm.py --num-examples ${1:-500} --use-grpo --grpo-beta ${2:-0.1} --grpo-clip ${3:-3.0}

POD_ID=$(cat /etc/hostname)
echo "Training complete. Terminating pod $POD_ID in 60s..."
sleep 60
prime pods terminate $POD_ID
