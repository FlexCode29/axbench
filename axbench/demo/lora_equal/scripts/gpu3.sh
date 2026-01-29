
python axbench/scripts/train.py \
  --config "/home/lse/hypersteer-weight/axbench/axbench/demo/lora_equal/lora_equal.yaml" \
  --overwrite_data_dir "/home/lse/hypersteer-weight/axbench/axbench/demo/lora_equal/data/concept_03_classical_mechanics_reasoning" \
  --dump_dir "/home/lse/hypersteer-weight/axbench/axbench/demo/lora_equal/runs_equal_concepts/lora_equal_c03" \
  --max_concepts 1 \
  --resume_from_latest true
