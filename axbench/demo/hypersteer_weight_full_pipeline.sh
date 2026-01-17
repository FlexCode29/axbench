#!/bin/bash

if ! command -v rocm-smi &> /dev/null; then
    echo "rocm-smi could not be found. Please ensure ROCM drivers are installed."
    exit 1
fi

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

gpu_count=8
config_path="axbench/demo/sweep/hypersteer_weight_full_pipeline.yaml"
dump_dir="axbench/16k_full_pipeline_results"
pregen_dir="axbench/concept16k/prod_2b_l20_v1"

if [[ ! -f "$pregen_dir/generate/train_data.parquet" || ! -f "$pregen_dir/generate/metadata.jsonl" ]]; then
    echo "Pre-generated data not found in $pregen_dir/generate."
    echo "Run axbench/scripts/generate.py or update pregen_dir in this script."
    exit 1
fi

torchrun --nproc_per_node="$gpu_count" axbench/scripts/train.py \
  --config "$config_path" \
  --dump_dir "$dump_dir" \
  --overwrite_data_dir "$pregen_dir/generate"

torchrun --nproc_per_node="$gpu_count" axbench/scripts/inference.py \
  --config "$config_path" \
  --mode steering \
  --dump_dir "$dump_dir" \
  --overwrite_metadata_dir "$pregen_dir/generate"

python axbench/scripts/evaluate.py \
  --config "$config_path" \
  --mode steering \
  --dump_dir "$dump_dir"

python axbench/scripts/evaluate.py \
  --config "$config_path" \
  --mode steering_test \
  --dump_dir "$dump_dir"
