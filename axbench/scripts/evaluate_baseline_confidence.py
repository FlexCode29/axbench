#!/usr/bin/env python3
"""
Compute confidence that a model beats a baseline using evaluate outputs.

Example:
python axbench/scripts/evaluate_baseline_confidence.py --dump_dir axbench/DDD --mode steering --baseline HyperSteerWeight --baseline-factor 0.0
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


def best_factors_from_jsonl(jsonl_path, evaluator_name, metric_key="lm_judge_rating"):
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
            ratings = metrics.get(metric_key, [])
            if not factors or not ratings:
                continue
            max_idx = max(range(len(ratings)), key=lambda i: ratings[i])
            best_factors.setdefault(concept_id, {})[model_name] = factors[max_idx]
    return best_factors


def build_factor_means_from_jsonl(jsonl_path, evaluator_name, metric_key="lm_judge_rating"):
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
            ratings = metrics.get(metric_key, [])
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
        "--model-factor",
        type=float,
        default=None,
        help=(
            "When using --baseline-factor, compare this factor against the baseline. "
            "If omitted, uses the best global factor by concept-mean (per metric)."
        ),
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


def compute_confidence_for_dir(args, evaluate_dir):
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

    metric_specs = [("aggregated", "lm_judge_rating", model_suffix)]
    if args.evaluator == "LMJudgeEvaluator":
        metric_specs.insert(
            0,
            ("concept", "relevance_concept_ratings", f"_{args.evaluator}_relevance_concept_ratings"),
        )

    best_factors_by_metric = {
        label: best_factors_from_jsonl(jsonl_path, args.evaluator, metric_key)
        for label, metric_key, _ in metric_specs
    }
    factor_means_by_metric = {
        label: build_factor_means_from_jsonl(jsonl_path, args.evaluator, metric_key)
        for label, metric_key, _ in metric_specs
    }

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
    if baseline_mode == "model":
        lines.append("selection: best factor per concept")
    elif args.model_factor is None:
        lines.append(f"selection: best factor (global concept mean) vs {args.baseline_factor}")
    else:
        lines.append(f"selection: factor {args.model_factor} vs {args.baseline_factor}")
    if len(metric_specs) > 1:
        lines.append(
            "metric_variants: "
            + ", ".join([f"{label}={metric_key}" for label, metric_key, _ in metric_specs])
        )
    lines.append("averages:")
    lines.append(
        "  concept_mean_graph: mean of per-concept factor means from jsonl (matches evaluate.py plots)"
    )
    lines.append(
        "  example_mean: mean over all rows in parquet (concepts weighted by #examples)"
    )
    lines.append("")

    def graph_mean_for_factor(metric_factor_means, model_name, factor):
        model_factor_means = metric_factor_means.get(model_name, {})
        concept_scores = model_factor_means.get(factor, {})
        if not concept_scores:
            return None, 0
        values = list(concept_scores.values())
        return sum(values) / len(values), len(values)

    def graph_mean_for_map(metric_factor_means, model_name, factor_map):
        model_factor_means = metric_factor_means.get(model_name, {})
        values = []
        for cid, factor in factor_map.items():
            if factor is None:
                continue
            concept_scores = model_factor_means.get(factor, {})
            if cid in concept_scores:
                values.append(concept_scores[cid])
        if not values:
            return None, 0
        return sum(values) / len(values), len(values)

    def best_global_factor(metric_factor_means, model_name, model_col):
        model_factor_means = metric_factor_means.get(model_name, {})
        if model_factor_means:
            factor_scores = {}
            for factor, concept_scores in model_factor_means.items():
                if concept_scores:
                    factor_scores[factor] = sum(concept_scores.values()) / len(concept_scores)
            if factor_scores:
                return max(factor_scores, key=factor_scores.get)
        grouped = df.groupby(["factor", "concept_id"], as_index=False)[model_col].mean()
        if grouped.empty:
            return None
        factor_scores = grouped.groupby("factor")[model_col].mean()
        if factor_scores.empty:
            return None
        return factor_scores.idxmax()

    def compute_means(df_slice, metric_col):
        if df_slice.empty:
            return None, None
        concept_means = df_slice.groupby("concept_id")[metric_col].mean()
        concept_mean = concept_means.mean() if not concept_means.empty else None
        example_mean = df_slice[metric_col].mean()
        return concept_mean, example_mean

    for model_name in sorted(model_names):
        if baseline_mode == "model" and model_name == args.baseline:
            continue
        for metric_label, metric_key, metric_suffix in metric_specs:
            model_col = f"{model_name}{metric_suffix}"
            if model_col not in df.columns:
                lines.append(f"{model_name} [{metric_label}]: missing column {model_col}")
                lines.append("")
                continue
            baseline_col = f"{args.baseline}{metric_suffix}"
            if baseline_mode == "model" and baseline_col not in df.columns:
                lines.append(f"{model_name} [{metric_label}]: missing column {baseline_col}")
                lines.append("")
                continue

            metric_best_factors = best_factors_by_metric.get(metric_label, {})
            metric_factor_means = factor_means_by_metric.get(metric_label, {})

            model_best = None
            baseline_best = None
            selected_factor = None

            if baseline_mode == "model":
                model_best = {cid: factors.get(model_name) for cid, factors in metric_best_factors.items()}
                baseline_best = {cid: factors.get(args.baseline) for cid, factors in metric_best_factors.items()}

                if not any(v is not None for v in model_best.values()):
                    model_best = best_factors_from_df(df, model_col)
                else:
                    fallback = best_factors_from_df(df, model_col)
                    for cid, factor in fallback.items():
                        if model_best.get(cid) is None:
                            model_best[cid] = factor

                if not any(v is not None for v in baseline_best.values()):
                    baseline_best = best_factors_from_df(df, baseline_col)
                else:
                    fallback = best_factors_from_df(df, baseline_col)
                    for cid, factor in fallback.items():
                        if baseline_best.get(cid) is None:
                            baseline_best[cid] = factor

                model_df = filter_best_factor_rows(df, {k: v for k, v in model_best.items() if v is not None})
                baseline_df = filter_best_factor_rows(
                    df, {k: v for k, v in baseline_best.items() if v is not None}
                )
            else:
                if args.model_factor is None:
                    selected_factor = best_global_factor(metric_factor_means, model_name, model_col)
                    if selected_factor is None:
                        lines.append(f"{model_name} [{metric_label}]: no valid factors to compare")
                        lines.append("")
                        continue
                    model_df = df[df["factor"] == selected_factor].copy()
                else:
                    selected_factor = float(args.model_factor)
                    model_df = df[df["factor"] == selected_factor].copy()
                baseline_df = df[df["factor"] == args.baseline_factor].copy()

            baseline_metric_col = baseline_col if baseline_mode == "model" else model_col
            paired = model_df[merge_keys + [model_col]].merge(
                baseline_df[merge_keys + [baseline_metric_col]],
                on=merge_keys,
                how="inner",
                suffixes=("_best", "_baseline"),
            )

            if paired.empty:
                lines.append(f"{model_name} [{metric_label}]: no paired rows after factor selection")
                lines.append("")
                continue

            wins = losses = ties = 0
            for _, row in paired.iterrows():
                if baseline_mode == "model":
                    model_score = row[model_col]
                    baseline_score = row[baseline_metric_col]
                else:
                    model_score = row[f"{model_col}_best"]
                    baseline_score = row[f"{baseline_metric_col}_baseline"]
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

            lines.append(f"{model_name} [{metric_label}]:")
            if baseline_mode == "factor" and selected_factor is not None:
                lines.append(f"  best_factor_global={selected_factor}")
            lines.append(f"  pairs={len(paired)} wins={wins} losses={losses} ties={ties}")
            lines.append(
                f"  win_rate_non_tie={win_rate:.4f} p_value_one_sided={p_value:.4g} confidence={confidence:.4g}"
            )
            if baseline_mode == "factor" and float(args.baseline_factor) == 0.0:
                model_factor_means = metric_factor_means.get(model_name, {})
                factor0_means = model_factor_means.get(0.0, {})
                if args.model_factor is None:
                    factor_best_means = model_factor_means.get(selected_factor, {})
                    concept_ids = sorted(set(factor_best_means.keys()) & set(factor0_means.keys()))
                    if not concept_ids:
                        lines.append("  jsonl_best_global_vs_0: no paired concepts")
                    else:
                        wins = losses = ties = 0
                        for cid in concept_ids:
                            score_best = factor_best_means.get(cid)
                            score0 = factor0_means.get(cid)
                            if score_best is None or score0 is None:
                                continue
                            if score_best > score0 + 1e-9:
                                wins += 1
                            elif score0 > score_best + 1e-9:
                                losses += 1
                            else:
                                ties += 1
                        non_ties = wins + losses
                        win_rate = wins / non_ties if non_ties > 0 else 0.0
                        p_value = binom_test_greater(wins, non_ties, p=0.5) if non_ties > 0 else 1.0
                        confidence = 1.0 - p_value
                        lines.append(
                            "  jsonl_best_global_vs_0:"
                            f" factor={selected_factor}"
                            f" concepts={len(concept_ids)}"
                            f" wins={wins}"
                            f" losses={losses}"
                            f" ties={ties}"
                            f" win_rate_non_tie={win_rate:.4f}"
                            f" p_value_one_sided={p_value:.4g}"
                            f" confidence={confidence:.4g}"
                        )
                    lines.append("")
                else:
                    factorX_means = model_factor_means.get(float(args.model_factor), {})
                    concept_ids = sorted(set(factorX_means.keys()) & set(factor0_means.keys()))
                    if not concept_ids:
                        lines.append("  jsonl_factor_vs_0: no paired concepts")
                    else:
                        wins = losses = ties = 0
                        for cid in concept_ids:
                            score1 = factorX_means[cid]
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
                            "  jsonl_factor_vs_0:"
                            f" concepts={len(concept_ids)}"
                            f" wins={wins}"
                            f" losses={losses}"
                            f" ties={ties}"
                            f" win_rate_non_tie={win_rate:.4f}"
                            f" p_value_one_sided={p_value:.4g}"
                            f" confidence={confidence:.4g}"
                        )
                    lines.append("")

            model_concept_mean, model_example_mean = compute_means(model_df, model_col)
            baseline_concept_mean, baseline_example_mean = compute_means(baseline_df, baseline_metric_col)
            if baseline_mode == "model":
                graph_model_mean, graph_model_n = graph_mean_for_map(
                    metric_factor_means, model_name, model_best or {}
                )
                graph_baseline_mean, graph_baseline_n = graph_mean_for_map(
                    metric_factor_means, args.baseline, baseline_best or {}
                )
            elif args.model_factor is None:
                graph_model_mean, graph_model_n = graph_mean_for_factor(
                    metric_factor_means, model_name, selected_factor
                )
                graph_baseline_mean, graph_baseline_n = graph_mean_for_factor(
                    metric_factor_means, model_name, args.baseline_factor
                )
            else:
                graph_model_mean, graph_model_n = graph_mean_for_factor(
                    metric_factor_means, model_name, float(args.model_factor)
                )
                graph_baseline_mean, graph_baseline_n = graph_mean_for_factor(
                    metric_factor_means, model_name, float(args.baseline_factor)
                )
            lines.append("  averages:")
            if graph_model_mean is None or graph_baseline_mean is None:
                lines.append("    concept_mean_graph: no paired concepts")
            else:
                lines.append(
                    "    concept_mean_graph:"
                    f" model={graph_model_mean:.4f} (n={graph_model_n})"
                    f" baseline={graph_baseline_mean:.4f} (n={graph_baseline_n})"
                )
            if model_example_mean is None or baseline_example_mean is None:
                lines.append("    example_mean: no paired rows")
            else:
                lines.append(
                    f"    example_mean: model={model_example_mean:.4f} baseline={baseline_example_mean:.4f}"
                )
            lines.append("")

    return lines


def resolve_output_path(args, evaluate_dir, dataset_label):
    if args.output is None:
        return evaluate_dir / f"confidence_vs_{args.baseline}.txt"
    output_path = Path(args.output)
    if dataset_label is None:
        return output_path
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_{dataset_label}{output_path.suffix}")
    return output_path / f"{dataset_label}.txt"


def main():
    args = parse_args()

    base_evaluate_dir = Path(args.dump_dir) / "evaluate"
    dataset_dirs = []
    if (base_evaluate_dir / "training_data").exists() or (base_evaluate_dir / "test_data").exists():
        dataset_dirs = [
            ("training_data", base_evaluate_dir / "training_data"),
            ("test_data", base_evaluate_dir / "test_data"),
        ]
    else:
        legacy_dirs = [
            ("evaluate", base_evaluate_dir),
            ("test_evaluate", Path(args.dump_dir) / "test_evaluate"),
        ]
        dataset_dirs = [(label, path) for label, path in legacy_dirs if path.exists()]
        if not dataset_dirs:
            dataset_dirs = [("evaluate", base_evaluate_dir)]

    wrote_any = False
    for dataset_label, evaluate_dir in dataset_dirs:
        try:
            lines = compute_confidence_for_dir(args, evaluate_dir)
        except FileNotFoundError as exc:
            print(f"Skipping {dataset_label}: {exc}")
            continue
        output_path = resolve_output_path(args, evaluate_dir, dataset_label)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines).rstrip() + "\n")
        print(f"Wrote confidence report to {output_path}")
        wrote_any = True

    if not wrote_any:
        raise FileNotFoundError("No evaluate outputs found for confidence report.")


if __name__ == "__main__":
    main()
