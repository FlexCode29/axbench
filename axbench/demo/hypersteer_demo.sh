#!/bin/bash

# Check if nvidia-smi command exists
if ! command -v rocm-smi &> /dev/null; then
    echo "rocm-smi could not be found. Please ensure ROCM drivers are installed."
    exit 1
fi

# Get the number of GPUs
gpu_count=8

python axbench/scripts/generate.py --config axbench/demo/sweep/hypersteer_simple.yaml --dump_dir axbench/demo

torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/hypersteer_simple.yaml --dump_dir axbench/demo

torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py --config axbench/demo/sweep/hypersteer_simple.yaml --mode steering --dump_dir axbench/demo

python axbench/scripts/evaluate.py --config axbench/demo/sweep/hypersteer_simple.yaml --mode steering --dump_dir axbench/demo
