#!/bin/bash
# MeanVC2 120ms+40ms training (recommended for quality)
# Usage:
#   bash scripts/train_120ms_40ms.sh 0                # single GPU
#   bash scripts/train_120ms_40ms.sh "0,1,2,3"        # multi GPU

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export WANDB_MODE="offline"

cuda="${1:-0}"
IFS=',' read -ra gpu_arr <<< "$cuda"
num_gpus=${#gpu_arr[@]}
echo ">>> GPUs: $cuda ($num_gpus)"

port=$(comm -23 <(seq 50075 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

accelerate launch \
    --config_file default_config.yaml \
    --main_process_port "$port" \
    --num_processes "$num_gpus" \
    --gpu_ids "$cuda" \
    src/train/train.py \
    --model-config src/config/config_120ms_40ms.json \
    --dataset-path "${DATASET_PATH:-example/example_train.list}" \
    --feature-list "bn mel xvector" \
    --additional-feature-list "inputs_length" \
    --feature-pad-values "0. -1.0 0." \
    --batch-size 16 \
    --max-len 960 \
    --epochs 1000 \
    --learning-rate 1e-4 \
    --num-warmup-updates 20000 \
    --save-per-updates 10000 \
    --grad-accumulation-steps 1 \
    --max-grad-norm 1.0 \
    --num-workers 8 \
    --flow-ratio 0.75 \
    --cfg-ratio 0.2 \
    --cfg-scale 2.0 \
    --cfg-strength 2.0 \
    --p 0.5 \
    --steps 1 \
    --chunk-size 12 \
    --block-size 4 \
    --resumable-with-seed 666 \
    --reset-lr 0 \
    --grad-ckpt 0 \
    --exp-name "${EXP_NAME:-meanvc2_120ms_test}" \
    --result-dir "results"
