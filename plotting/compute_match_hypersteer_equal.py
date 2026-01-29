#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import yaml


def parse_iso(dt_str):
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def load_config_params(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train_cfg = config.get("train", {})
    models_cfg = train_cfg.get("models", {})
    lora_cfg = models_cfg.get("LoRA", {})

    batch_size = lora_cfg.get("batch_size", train_cfg.get("batch_size", 1))
    grad_acc = lora_cfg.get("gradient_accumulation_steps", train_cfg.get("gradient_accumulation_steps", 1))
    n_epochs = lora_cfg.get("n_epochs", train_cfg.get("n_epochs", 10))

    return int(batch_size or 1), int(grad_acc or 1), int(n_epochs or 10)


def load_concept_list(data_root: Path):
    concepts = []
    for concept_dir in sorted(data_root.glob("concept_*")):
        if not concept_dir.is_dir():
            continue
        try:
            concept_id = int(concept_dir.name.split("_")[1])
        except Exception:
            continue
        meta_path = concept_dir / "metadata.jsonl"
        concept_name = None
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if int(data.get("concept_id", -1)) == concept_id:
                        concept_name = data.get("concept")
                        break
        if concept_name:
            concepts.append((concept_id, concept_name, concept_dir))
    concepts.sort(key=lambda x: x[0])
    return concepts


def load_num_examples(concept_dir: Path):
    train_path = concept_dir / "train_data.parquet"
    if not train_path.exists():
        return 1
    df = pd.read_parquet(train_path)
    return max(1, int(len(df)))


def load_history(run):
    steps = []
    losses = []
    for row in run.scan_history(keys=["train/loss", "_step"]):
        loss = row.get("train/loss")
        step = row.get("_step")
        if loss is None or step is None:
            continue
        try:
            loss = float(loss)
            step = int(step)
        except Exception:
            continue
        if not math.isfinite(loss):
            continue
        steps.append(step)
        losses.append(loss)
    if not steps:
        return [], []
    order = np.argsort(steps)
    steps = [steps[i] for i in order]
    losses = [losses[i] for i in order]
    return steps, losses


def build_continuous_series(runs):
    ordered = sorted(runs, key=lambda r: parse_iso(r.created_at) or datetime.min)
    all_steps = []
    all_losses = []
    prev_max = None
    for run in ordered:
        steps, losses = load_history(run)
        if not steps:
            continue
        offset = 0
        if prev_max is not None and steps[0] <= prev_max:
            offset = (prev_max + 1) - steps[0]
        steps = [s + offset for s in steps]
        prev_max = max(steps) if steps else prev_max
        all_steps.extend(steps)
        all_losses.extend(losses)
    if not all_steps:
        return [], []
    order = np.argsort(all_steps)
    steps = np.asarray([all_steps[i] for i in order], dtype=float)
    losses = np.asarray([all_losses[i] for i in order], dtype=float)
    # de-dup by step (keep last)
    uniq_steps = []
    uniq_losses = []
    last_step = None
    for s, l in zip(steps, losses):
        if last_step is None or s != last_step:
            uniq_steps.append(s)
            uniq_losses.append(l)
            last_step = s
        else:
            uniq_losses[-1] = l
    return uniq_steps, uniq_losses


def ema_smooth(values, alpha):
    if not values:
        return []
    out = [float(values[0])]
    a = float(alpha)
    for v in values[1:]:
        out.append(a * float(v) + (1.0 - a) * out[-1])
    return out


def steps_to_epochs(steps, num_examples, batch_size, grad_acc):
    len_dataloader = int(math.ceil(num_examples / batch_size))
    steps_per_epoch = max(1, len_dataloader // max(1, grad_acc))
    return [s / steps_per_epoch for s in steps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_path", default=None, type=str, help="entity/project")
    parser.add_argument("--wandb_project", default="axbench-lora-equal", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--config", default="axbench/demo/lora_equal/lora_equal.yaml", type=str)
    parser.add_argument("--data_root", default="axbench/demo/lora_equal/data", type=str)
    parser.add_argument("--hypersteer_loss", default=0.31, type=float)
    parser.add_argument("--hypersteer_epoch", default=10.0, type=float)
    parser.add_argument("--base_pf", default=1.035, type=float)
    parser.add_argument("--base_epoch", default=11.0, type=float)
    parser.add_argument("--ema_alpha", default=0.1, type=float)
    parser.add_argument("--grid_points", default=1000, type=int)
    parser.add_argument("--extrapolate_to", default=1000, type=int)
    parser.add_argument("--out", default="plotting/compute_match_hypersteer_equal.png", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    wandb_path = args.wandb_path
    if not wandb_path:
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        if entity:
            wandb_path = f"{entity}/{args.wandb_project}"
        else:
            wandb_path = args.wandb_project

    batch_size, grad_acc, n_epochs = load_config_params(Path(args.config))
    concept_list = load_concept_list(Path(args.data_root))
    if not concept_list:
        raise SystemExit("No concepts found under data_root")

    api = wandb.Api()
    runs = api.runs(wandb_path)
    if not runs:
        raise SystemExit(f"No wandb runs found at {wandb_path}")

    runs_by_concept = defaultdict(list)
    for run in runs:
        cfg = run.config or {}
        concept_name = cfg.get("concept_name")
        if concept_name:
            runs_by_concept[concept_name].append(run)

    series = {}
    for concept_id, concept_name, concept_dir in concept_list:
        concept_runs = runs_by_concept.get(concept_name, [])
        if not concept_runs:
            continue
        steps, losses = build_continuous_series(concept_runs)
        if not steps:
            continue
        num_examples = load_num_examples(concept_dir)
        epochs = steps_to_epochs(steps, num_examples, batch_size, grad_acc)
        losses = ema_smooth(losses, args.ema_alpha)
        series[concept_name] = {
            "epochs": np.asarray(epochs, dtype=float),
            "loss": np.asarray(losses, dtype=float),
        }
        if args.debug:
            print(f"{concept_name}: epochs={epochs[-1]:.3f} steps={steps[-1]}")

    if not series:
        raise SystemExit("No usable series from wandb")

    concept_order = [name for _, name, _ in concept_list if name in series]
    epoch_grid = np.linspace(0.0, float(n_epochs), int(args.grid_points))

    target = float(args.hypersteer_loss)
    compute_eq = []
    for k in range(1, len(concept_order) + 1):
        names = concept_order[:k]
        losses_grid = []
        for name in names:
            s = series[name]
            losses_grid.append(np.interp(epoch_grid, s["epochs"], s["loss"], left=s["loss"][0], right=s["loss"][-1]))
        avg_loss = np.mean(np.vstack(losses_grid), axis=0)
        idx = np.where(avg_loss <= target)[0]
        if len(idx) == 0:
            compute_eq.append(np.nan)
            continue
        epoch_needed = float(epoch_grid[idx[0]])
        compute_eq.append((k * epoch_needed) / 8.0)

    # Plot compute-equivalent epochs vs concepts added
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = np.arange(1, len(compute_eq) + 1)
    pf_per_epoch = args.base_pf / args.base_epoch
    compute_pf = np.asarray(compute_eq, dtype=float) * pf_per_epoch
    ax.plot(xs, compute_pf, color="#b24a4a", linewidth=2.5, label="LoRA")

    # Extrapolate to larger concept counts (imagination time)
    finite_mask = np.isfinite(compute_eq)
    if finite_mask.any():
        xs_fit = xs[finite_mask]
        ys_fit = np.asarray(compute_eq)[finite_mask]
        if len(xs_fit) >= 2:
            coeffs = np.polyfit(xs_fit, ys_fit, deg=1)
            x_extra = np.arange(xs_fit[-1], args.extrapolate_to + 1)
            y_extra = np.polyval(coeffs, x_extra)
            ax.plot(x_extra, y_extra * pf_per_epoch, color="#b24a4a", linewidth=2.5)
    ax.axhline(args.hypersteer_epoch * pf_per_epoch, color="#4c72b0", linestyle="--", linewidth=1.5, label="HyperSteer compute")
    ax.set_xlabel("Concepts added (sequential)")
    ax.set_ylabel("LoRA compute (PetaFLOPs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")

    print("Compute‑equivalent epochs to reach target:")
    for k, c in enumerate(compute_eq, start=1):
        print(f"{k} concepts: {c:.4f}" if math.isfinite(c) else f"{k} concepts: not reached")


if __name__ == "__main__":
    main()
