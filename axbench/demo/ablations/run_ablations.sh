#!/bin/bash
set -euo pipefail

if ! command -v rocm-smi &> /dev/null && ! command -v nvidia-smi &> /dev/null; then
    echo "No ROCm or NVIDIA GPU tools found (rocm-smi/nvidia-smi)."
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
if [[ -f "$repo_root/.env" ]]; then
    set -a
    . "$repo_root/.env"
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set. Add it to $repo_root/.env or export it."
    exit 1
fi

gpu_count="${GPU_COUNT:-8}"
config_root="axbench/demo/ablations/ablations"
dump_root="axbench/demo/ablations/dumps"

# "baseline_concept500|$config_root/baseline_concept500.yaml|axbench/demo_weight"

runs=(
  "ablate_gemma_9b_l20|$config_root/ablate_gemma_9b_l20.yaml|axbench/demo_weight"
  "ablate_rank_r4|$config_root/ablate_rank_r4.yaml|axbench/demo_weight"
  "ablate_epochs_3|$config_root/ablate_epochs_3.yaml|axbench/demo_weight"
  "ablate_layer_l20|$config_root/ablate_layer_l20.yaml|axbench/demo_weight"
  "ablate_num_layers_2|$config_root/ablate_num_layers_2.yaml|axbench/demo_weight"
  "ablate_hnet_depth_4|$config_root/ablate_hnet_depth_4.yaml|axbench/demo_weight"
)

for entry in "${runs[@]}"; do
  IFS='|' read -r run_name config_path pregen_dir <<< "$entry"
  dump_dir="$dump_root/$run_name"

  if [[ ! -f "$pregen_dir/generate/train_data.parquet" || ! -f "$pregen_dir/generate/metadata.jsonl" ]]; then
      echo "Pre-generated data not found in $pregen_dir/generate."
      echo "Update pregen_dir for $run_name or generate the data first."
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

done
