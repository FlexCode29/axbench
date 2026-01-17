#!/bin/bash

if ! command -v rocm-smi &> /dev/null; then
    echo "rocm-smi could not be found. Please ensure ROCM drivers are installed."
    exit 1
fi

gpu_count=8
export DEMO_DIR=axbench/demo_weight

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
    set -a
    . "$repo_root/.env"
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set. Add it to $repo_root/.env or export it."
    exit 1
fi

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
