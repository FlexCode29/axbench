

python axbench/scripts/train.py \
  --config "/home/lse/hypersteer-weight/axbench/axbench/demo/lora_equal/lora_equal.yaml" \
  --overwrite_data_dir "/home/lse/hypersteer-weight/axbench/axbench/demo/lora_equal/data/concept_05_medieval_european_history_reasoning" \
  --dump_dir "/home/lse/hypersteer-weight/axbench/axbench/demo/lora_equal/runs_equal_concepts/lora_equal_c05" \
  --max_concepts 1 \
  --resume_from_latest true
