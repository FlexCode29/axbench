#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    n_epochs = lora_cfg.get("n_epochs", train_cfg.get("n_epochs", 16))
    train_on_negative = lora_cfg.get("train_on_negative", train_cfg.get("train_on_negative", False))
    binarize_dataset = lora_cfg.get("binarize_dataset", train_cfg.get("binarize_dataset", False))

    return (
        int(batch_size or 1),
        int(grad_acc or 1),
        int(n_epochs or 16),
        bool(train_on_negative),
        bool(binarize_dataset),
    )


def load_concept_info(data_root: Path, train_on_negative: bool, binarize_dataset: bool):
    concept_info = {}
    for concept_dir in sorted(data_root.glob("concept_*")):
        if not concept_dir.is_dir():
            continue
        metadata_path = concept_dir / "metadata.jsonl"
        if not metadata_path.exists():
            continue
        try:
            concept_id = int(concept_dir.name.split("_")[1])
        except Exception:
            concept_id = None
        concept_name = None
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if concept_id is not None and int(data.get("concept_id", -1)) == concept_id:
                    concept_name = data.get("concept")
                    break
        if not concept_name:
            continue

        train_path = concept_dir / "train_data.parquet"
        if train_path.exists():
            df = pd.read_parquet(train_path)
            if binarize_dataset or train_on_negative:
                num_examples = int(len(df))
            else:
                num_examples = int(len(df[(df["output_concept"] == concept_name) & (df["category"] == "positive")]))
        else:
            num_examples = 1

        concept_info[concept_name] = {
            "concept_id": concept_id,
            "dir": concept_dir,
            "num_examples": max(num_examples, 1),
        }
    return concept_info


def fetch_runs(wandb_path: str):
    api = wandb.Api()
    return api.runs(wandb_path)


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


def normalize_label(name):
    return "".join(c.lower() for c in name if c.isalnum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_path", default=None, type=str, help="entity/project")
    parser.add_argument("--wandb_project", default="axbench-lora", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--config", default="axbench/demo/lora_top10/lora_top10.yaml", type=str)
    parser.add_argument("--data_root", default="axbench/demo/lora_top10/data", type=str)
    parser.add_argument("--out", default="plotting/compute_frontier.png", type=str)
    parser.add_argument("--compute_pf", default=1.035, type=float)
    parser.add_argument("--hypersteer_loss", default=0.31, type=float)
    parser.add_argument("--ema_alpha", default=0.1, type=float)
    parser.add_argument("--avg_grid_points", default=1000, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    wandb_path = args.wandb_path
    if not wandb_path:
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        if entity:
            wandb_path = f"{entity}/{args.wandb_project}"
        else:
            wandb_path = args.wandb_project

    batch_size, grad_acc, n_epochs, train_on_negative, binarize_dataset = load_config_params(Path(args.config))
    concept_info = load_concept_info(Path(args.data_root), train_on_negative, binarize_dataset)
    if not concept_info:
        raise SystemExit("No concepts found under data_root")

    runs = fetch_runs(wandb_path)
    if not runs:
        raise SystemExit(f"No wandb runs found at {wandb_path}")

    norm_map = {normalize_label(k): k for k in concept_info.keys()}
    runs_by_concept = defaultdict(list)
    for run in runs:
        cfg = run.config or {}
        concept_name = cfg.get("concept_name")
        if not concept_name:
            norm = normalize_label(run.name or "")
            concept_name = norm_map.get(norm)
        if concept_name in concept_info:
            runs_by_concept[concept_name].append(run)

    if not runs_by_concept:
        raise SystemExit("No runs matched concepts")

    series = {}
    for concept_name, concept_runs in runs_by_concept.items():
        steps, losses = build_continuous_series(concept_runs)
        if not steps:
            continue
        num_examples = concept_info[concept_name]["num_examples"]
        epochs = steps_to_epochs(steps, num_examples, batch_size, grad_acc)
        losses = ema_smooth(losses, args.ema_alpha)
        series[concept_name] = {
            "steps": np.asarray(steps, dtype=float),
            "epochs": np.asarray(epochs, dtype=float),
            "loss": np.asarray(losses, dtype=float),
            "weight": float(num_examples),
        }
        if args.debug:
            print(
                f"{concept_name}: steps={len(steps)} min_step={steps[0]} max_step={steps[-1]} "
                f"min_epoch={epochs[0]:.3f} max_epoch={epochs[-1]:.3f} examples={num_examples}"
            )

    if not series:
        raise SystemExit("No usable series to plot")

    epoch_grid = np.linspace(0.0, float(n_epochs), int(args.avg_grid_points))
    weighted_sum = np.zeros_like(epoch_grid)
    weight_total = 0.0
    for concept_name, s in series.items():
        interp_loss = np.interp(epoch_grid, s["epochs"], s["loss"], left=s["loss"][0], right=s["loss"][-1])
        weighted_sum += interp_loss * s["weight"]
        weight_total += s["weight"]
    avg_loss = weighted_sum / max(weight_total, 1.0)
    avg_loss = np.asarray(ema_smooth(avg_loss.tolist(), args.ema_alpha), dtype=float)
    if not np.isfinite(avg_loss).all():
        avg_loss = pd.Series(avg_loss).replace([np.inf, -np.inf], np.nan).ffill().bfill().to_numpy()

    def epoch_to_pf(epoch):
        return args.compute_pf * (epoch / 11.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    per_label = "LoRA baseline (concept)"
    for concept_name, s in sorted(series.items()):
        ax.plot(
            epoch_to_pf(s["epochs"]),
            s["loss"],
            color="#d6a0a3",
            alpha=0.25,
            linewidth=1.5,
            label=per_label,
        )
        per_label = None

    ax.plot(epoch_to_pf(epoch_grid), avg_loss, color="#b24a4a", linewidth=3.0, label="LoRA average")
    ax.plot([args.compute_pf], [args.hypersteer_loss], marker="*", markersize=12, color="#4c72b0")

    ax.set_xlabel("Compute (PetaFLOPs)")
    ax.set_ylabel("CE Loss")
    ax.set_title("Compute Frontier (steps → epochs)")
    ax.grid(True, alpha=0.2)

    legend_handles = [
        Line2D([0], [0], color="#d6a0a3", alpha=0.25, linewidth=1.5, label="LoRA baseline (concept)"),
        Line2D([0], [0], color="#b24a4a", linewidth=3.0, label="LoRA average"),
        Line2D([0], [0], color="#4c72b0", marker="*", linestyle="None", markersize=10, label="hypersteer"),
    ]
    ax.legend(handles=legend_handles, frameon=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
