# LoRA top-10 concepts (axbench-reasoning)

This folder contains helpers to:
- slice the first 16k rows of `NONO/axbench-reasoning` into top-10 concepts
- train LoRA adapters concept-by-concept, saving checkpoints at epochs 5/11/16

## 1) Prepare data

```bash
python3 axbench/demo/lora_top10/prepare_axbench_reasoning_top10.py \
  --out_dir axbench/demo/lora_top10/data
```

To select specific concepts (comma-separated):

```bash
python3 axbench/demo/lora_top10/prepare_axbench_reasoning_top10.py \
  --out_dir axbench/demo/lora_top10/data \
  --concepts "Basic Arithmetic Reasoning,Computer Programming Reasoning,Epidemiology Reasoning,Classical Mechanics Reasoning,Organic Chemistry Reasoning,Medieval European History Reasoning,Constitutional Law Reasoning,Narrative Structure Reasoning"
```

This writes:
- `axbench/demo/lora_top10/data/combined/train_data.parquet`
- `axbench/demo/lora_top10/data/combined/metadata.jsonl`
- `axbench/demo/lora_top10/data/concept_XX_<slug>/train_data.parquet`
- `axbench/demo/lora_top10/data/concept_XX_<slug>/metadata.jsonl`

## 2) Train LoRA per concept (one GPU per concept via tmux)

```bash
CONFIG=axbench/demo/lora_top10/lora_top10.yaml \
DATA_ROOT=axbench/demo/lora_top10/data \
DUMP_ROOT=axbench/demo/lora_top10/runs \
SESSION=lora_top10 \
axbench/demo/lora_top10/run_lora_top10.sh
```

You can override any training args by appending CLI flags to the script call, e.g.:

```bash
axbench/demo/lora_top10/run_lora_top10.sh --model_param LoRA.lr=0.0005
```

The runner always resumes from the latest epoch checkpoint (if present) and uses GPUs 0-7.
LoRA checkpoint losses are logged to Weights & Biases with one run per concept.

## Checkpoints

LoRA adapters are saved under:
- `.../train/lora/<concept_id>/epoch_5__<concept_name>/`
- `.../train/lora/<concept_id>/epoch_11__<concept_name>/`
- `.../train/lora/<concept_id>/epoch_16__<concept_name>/`

The final adapter is also saved at:
- `.../train/lora/<concept_id>/`

Adjust `save_epochs` and `n_epochs` in `lora_top10.yaml` as needed.
