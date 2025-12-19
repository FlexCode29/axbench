#!/usr/bin/env python3
"""
Compute confidence that a model beats a baseline using evaluate outputs.

Example:
  python axbench/scripts/evaluate_baseline_confidence.py \
  --dump_dir axbench/demo \
  --mode steering \
  --baseline HyperSteerWeight \
  --baseline-factor 0.0
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def best_factors_from_jsonl(jsonl_path, evaluator_name):
    best_factors = {}
    if not jsonl_path.exists():
        return best_factors
    for record in load_jsonl(jsonl_path):
        concept_id = record.get("concept_id")
        results = record.get("results", {}).get(evaluator_name, {})
        if concept_id is None:
            continue
        for model_name, metrics in results.items():
            factors = metrics.get("factor", [])
            ratings = metrics.get("lm_judge_rating", [])
            if not factors or not ratings:
                continue
            max_idx = max(range(len(ratings)), key=lambda i: ratings[i])
            best_factors.setdefault(concept_id, {})[model_name] = factors[max_idx]
    return best_factors


def build_factor_means_from_jsonl(jsonl_path, evaluator_name):
    factor_means = {}
    if not jsonl_path.exists():
        return factor_means
    for record in load_jsonl(jsonl_path):
        concept_id = record.get("concept_id")
        results = record.get("results", {}).get(evaluator_name, {})
        if concept_id is None:
            continue
        for model_name, metrics in results.items():
            factors = metrics.get("factor", [])
            ratings = metrics.get("lm_judge_rating", [])
            if not factors or not ratings:
                continue
            for factor, rating in zip(factors, ratings):
                factor_means.setdefault(model_name, {}).setdefault(factor, {})[concept_id] = rating
    return factor_means


def best_factors_from_df(df, model_column):
    grouped = df.groupby(["concept_id", "factor"], as_index=False)[model_column].mean()
    idx = grouped.groupby("concept_id")[model_column].idxmax()
    return grouped.loc[idx].set_index("concept_id")["factor"].to_dict()


def filter_best_factor_rows(df, best_factor_map):
    if not best_factor_map:
        return df.iloc[0:0]
    best_df = pd.DataFrame(
        {"concept_id": list(best_factor_map.keys()), "best_factor": list(best_factor_map.values())}
    )
    merged = df.merge(best_df, on="concept_id", how="inner")
    return merged[merged["factor"] == merged["best_factor"]].copy()


def log_sum_exp(log_values):
    if not log_values:
        return float("-inf")
    max_log = max(log_values)
    if max_log == float("-inf"):
        return max_log
    total = sum(math.exp(v - max_log) for v in log_values)
    return max_log + math.log(total)


def binom_test_greater(k, n, p=0.5):
    if n <= 0:
        return 1.0
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    log_terms = []
    for i in range(k, n + 1):
        log_coef = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        log_terms.append(log_coef + i * log_p + (n - i) * log_q)
    return min(1.0, math.exp(log_sum_exp(log_terms)))


def parse_args():
    parser = argparse.ArgumentParser(description="Compute confidence vs baseline from evaluate outputs.")
    parser.add_argument("--dump_dir", type=str, required=True, help="Root dump dir used by evaluate.")
    parser.add_argument("--mode", type=str, default="steering", help="Evaluation mode (steering or latent).")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline model name.")
    parser.add_argument(
        "--baseline-factor",
        type=float,
        default=None,
        help="Use rows with this factor as the baseline within the same model (e.g. 0.0).",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="LMJudgeEvaluator",
        help="Evaluator name used to produce per-example scores.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output txt path; defaults to <dump_dir>/evaluate/confidence_vs_<baseline>.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_dir = Path(args.dump_dir) / "evaluate"
    jsonl_path = evaluate_dir / f"{args.mode}.jsonl"
    parquet_path = evaluate_dir / f"{args.mode}_data.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing evaluate outputs at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    model_suffix = f"_{args.evaluator}"
    model_columns = [c for c in df.columns if c.endswith(model_suffix)]
    model_names = [c[: -len(model_suffix)] for c in model_columns]
    baseline_mode = "factor" if args.baseline_factor is not None else "model"
    baseline_in_columns = baseline_mode == "model" and args.baseline in model_names
    if baseline_mode == "model" and not baseline_in_columns:
        raise ValueError(
            f"Baseline '{args.baseline}' not found in evaluate data. "
            "Provide --baseline-factor to use factor-based baseline."
        )

    best_factors = best_factors_from_jsonl(jsonl_path, args.evaluator)
    factor_means = build_factor_means_from_jsonl(jsonl_path, args.evaluator)

    merge_keys = ["concept_id", "input_id"]
    for extra_key in ["dataset_name", "original_prompt"]:
        if extra_key in df.columns:
            merge_keys.append(extra_key)

    lines = []
    if baseline_mode == "model":
        lines.append(f"baseline: {args.baseline}")
    else:
        lines.append(f"baseline: factor {args.baseline_factor} (same model)")
    lines.append(f"metric: {args.evaluator}")
    lines.append("selection: best factor per concept")
    lines.append("")

    for model_name in sorted(model_names):
        if baseline_mode == "model" and model_name == args.baseline:
            continue
        model_col = f"{model_name}{model_suffix}"
        baseline_col = f"{args.baseline}{model_suffix}"

        model_best = {cid: factors.get(model_name) for cid, factors in best_factors.items()}
        baseline_best = {cid: factors.get(args.baseline) for cid, factors in best_factors.items()}

        if not any(v is not None for v in model_best.values()):
            model_best = best_factors_from_df(df, model_col)
        else:
            fallback = best_factors_from_df(df, model_col)
            for cid, factor in fallback.items():
                if model_best.get(cid) is None:
                    model_best[cid] = factor

        if baseline_mode == "model":
            if not any(v is not None for v in baseline_best.values()):
                baseline_best = best_factors_from_df(df, baseline_col)
            else:
                fallback = best_factors_from_df(df, baseline_col)
                for cid, factor in fallback.items():
                    if baseline_best.get(cid) is None:
                        baseline_best[cid] = factor

        model_df = filter_best_factor_rows(df, {k: v for k, v in model_best.items() if v is not None})
        if baseline_mode == "model":
            baseline_df = filter_best_factor_rows(
                df, {k: v for k, v in baseline_best.items() if v is not None}
            )
        else:
            baseline_df = df[df["factor"] == args.baseline_factor].copy()

        baseline_metric_col = baseline_col if baseline_mode == "model" else model_col
        paired = model_df[merge_keys + [model_col]].merge(
            baseline_df[merge_keys + [baseline_metric_col]],
            on=merge_keys,
            how="inner",
            suffixes=("_best", "_baseline"),
        )

        if paired.empty:
            lines.append(f"{model_name}: no paired rows after factor selection")
            lines.append("")
            continue

        wins = losses = ties = 0
        for _, row in paired.iterrows():
            model_score = row[f"{model_col}_best"] if baseline_mode == "factor" else row[model_col]
            baseline_score = (
                row[f"{baseline_metric_col}_baseline"]
                if baseline_mode == "factor"
                else row[baseline_metric_col]
            )
            if model_score > baseline_score + 1e-9:
                wins += 1
            elif baseline_score > model_score + 1e-9:
                losses += 1
            else:
                ties += 1

        non_ties = wins + losses
        win_rate = wins / non_ties if non_ties > 0 else 0.0
        p_value = binom_test_greater(wins, non_ties, p=0.5) if non_ties > 0 else 1.0
        confidence = 1.0 - p_value

        lines.append(f"{model_name}:")
        lines.append(f"  pairs={len(paired)} wins={wins} losses={losses} ties={ties}")
        lines.append(f"  win_rate_non_tie={win_rate:.4f} p_value_one_sided={p_value:.4g} confidence={confidence:.4g}")
        if baseline_mode == "factor" and float(args.baseline_factor) == 0.0:
            model_factor_means = factor_means.get(model_name, {})
            factor1_means = model_factor_means.get(1.0, {})
            factor0_means = model_factor_means.get(0.0, {})
            concept_ids = sorted(set(factor1_means.keys()) & set(factor0_means.keys()))
            if not concept_ids:
                lines.append("  jsonl_factor1_vs_0: no paired concepts")
            else:
                wins = losses = ties = 0
                for cid in concept_ids:
                    score1 = factor1_means[cid]
                    score0 = factor0_means[cid]
                    if score1 > score0 + 1e-9:
                        wins += 1
                    elif score0 > score1 + 1e-9:
                        losses += 1
                    else:
                        ties += 1
                non_ties = wins + losses
                win_rate = wins / non_ties if non_ties > 0 else 0.0
                p_value = binom_test_greater(wins, non_ties, p=0.5) if non_ties > 0 else 1.0
                confidence = 1.0 - p_value
                lines.append(
                    "  jsonl_factor1_vs_0:"
                    f" concepts={len(concept_ids)}"
                    f" wins={wins}"
                    f" losses={losses}"
                    f" ties={ties}"
                    f" win_rate_non_tie={win_rate:.4f}"
                    f" p_value_one_sided={p_value:.4g}"
                    f" confidence={confidence:.4g}"
                )
        lines.append("")

    output_path = Path(args.output) if args.output else evaluate_dir / f"confidence_vs_{args.baseline}.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote confidence report to {output_path}")


if __name__ == "__main__":
    main()
