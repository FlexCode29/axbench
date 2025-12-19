#!/bin/bash

if ! command -v rocm-smi &> /dev/null; then
    echo "rocm-smi could not be found. Please ensure ROCM drivers are installed."
    exit 1
fi

gpu_count=8
export DEMO_DIR=axbench/demo_weight
export OPENAI_API_KEY="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"

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
