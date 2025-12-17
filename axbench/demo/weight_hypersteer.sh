#!/bin/bash

if ! command -v rocm-smi &> /dev/null; then
    echo "rocm-smi could not be found. Please ensure ROCM drivers are installed."
    exit 1
fi

gpu_count=8
export DEMO_DIR=axbench/demo_weight

python axbench/scripts/generate.py \
  --config axbench/demo/sweep/hypersteer_weight_simple.yaml \
  --dump_dir $DEMO_DIR

torchrun --nproc_per_node=$gpu_count axbench/scripts/train.py \
  --config axbench/demo/sweep/hypersteer_weight_simple.yaml \
  --dump_dir $DEMO_DIR

torchrun --nproc_per_node=$gpu_count axbench/scripts/inference.py \
  --config axbench/demo/sweep/hypersteer_weight_simple.yaml \
  --mode steering \
  --dump_dir $DEMO_DIR

python axbench/scripts/evaluate.py \
  --config axbench/demo/sweep/hypersteer_weight_simple.yaml \
  --mode steering \
  --dump_dir $DEMO_DIR
