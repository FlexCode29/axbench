#!/usr/bin/env python3
"""
Summarize composite score deltas (best factor vs baseline) across ablation dumps.

Default behavior:
- Reads evaluate/<dataset>/steering.jsonl for training_data and test_data.
- Uses LMJudgeEvaluator lm_judge_rating as the composite score.
- Picks the best factor by highest mean across concepts.
"""
import argparse
import json
import math
from pathlib import Path

BEST_NAME = "ablate_num_layers_2"
SE_SAMPLE_SIZE = 300


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def one_sided_pvalue(diff, se):
    if se <= 0.0:
        return 0.0 if diff > 0 else 1.0
    z = diff / se
    return 1.0 - normal_cdf(z)


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_factor_means(records, evaluator_name, metric_key):
    # model -> factor -> list of per-concept means
    buckets = {}
    for record in records:
        results = record.get("results", {}).get(evaluator_name, {})
        for model_name, metrics in results.items():
            factors = metrics.get("factor", [])
            values = metrics.get(metric_key, [])
            if not factors or not values:
                continue
            for factor, value in zip(factors, values):
                buckets.setdefault(model_name, {}).setdefault(float(factor), []).append(value)
    # model -> factor -> mean/std
    means = {}
    for model_name, factor_map in buckets.items():
        means[model_name] = {}
        for factor, values in factor_map.items():
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = variance ** 0.5
            else:
                std = 0.0
            means[model_name][factor] = {"mean": mean, "std": std}
    return means


def summarize_jsonl(jsonl_path, evaluator_name, metric_key, baseline_factor):
    if not jsonl_path.exists():
        return None, f"missing {jsonl_path}"
    records = load_jsonl(jsonl_path)
    factor_means = compute_factor_means(records, evaluator_name, metric_key)
    summaries = {}
    for model_name, factor_map in factor_means.items():
        if not factor_map:
            continue
        if baseline_factor not in factor_map:
            summaries[model_name] = {
                "error": f"baseline_factor {baseline_factor} not found",
            }
            continue
        best_factor = max(sorted(factor_map.keys()), key=lambda f: factor_map[f]["mean"])
        baseline_stats = factor_map[baseline_factor]
        best_stats = factor_map[best_factor]
        summaries[model_name] = {
            "baseline_factor": baseline_factor,
            "baseline_mean": baseline_stats["mean"],
            "baseline_se": baseline_stats["std"] / (SE_SAMPLE_SIZE ** 0.5),
            "best_factor": best_factor,
            "best_mean": best_stats["mean"],
            "best_se": best_stats["std"] / (SE_SAMPLE_SIZE ** 0.5),
            "diff": best_stats["mean"] - baseline_stats["mean"],
        }
        diff_se = (summaries[model_name]["best_se"] ** 2 + summaries[model_name]["baseline_se"] ** 2) ** 0.5
        summaries[model_name]["p_value_one_sided"] = one_sided_pvalue(
            summaries[model_name]["diff"], diff_se
        )
    return summaries, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize composite score deltas across ablation dumps."
    )
    parser.add_argument(
        "--dumps-dir",
        type=str,
        default="axbench/demo/ablations/dumps",
        help="Directory containing ablation dump subfolders.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="steering",
        help="Evaluation mode to read (e.g., steering or steering_test).",
    )
    parser.add_argument(
        "--baseline-factor",
        type=float,
        default=0.0,
        help="Baseline factor used for comparisons.",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="LMJudgeEvaluator",
        help="Evaluator name used in jsonl results.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="lm_judge_rating",
        help="Metric key in jsonl evaluator results (composite score).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output txt path; defaults to <dumps-dir>/composite_diffs.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dumps_dir = Path(args.dumps_dir)
    output_path = (
        Path(args.output)
        if args.output is not None
        else dumps_dir / "composite_diffs.txt"
    )

    dataset_labels = ["training_data", "test_data"]
    lines = []
    lines.append(f"mode: {args.mode}")
    lines.append(f"evaluator: {args.evaluator}")
    lines.append(f"metric: {args.metric}")
    lines.append(f"baseline_factor: {args.baseline_factor}")
    lines.append(f"best: {BEST_NAME}")
    lines.append(f"se_sample_size: {SE_SAMPLE_SIZE}")
    lines.append("")

    for dump_dir in sorted(p for p in dumps_dir.iterdir() if p.is_dir()):
        label = dump_dir.name
        if dump_dir.name == BEST_NAME:
            label = f"{dump_dir.name} (best)"
        lines.append(f"{label}:")
        evaluate_dir = dump_dir / "evaluate"
        if not evaluate_dir.exists():
            lines.append("  missing evaluate directory")
            lines.append("")
            continue

        for dataset_label in dataset_labels:
            jsonl_path = evaluate_dir / dataset_label / f"{args.mode}.jsonl"
            summaries, error = summarize_jsonl(
                jsonl_path,
                args.evaluator,
                args.metric,
                float(args.baseline_factor),
            )
            if error:
                lines.append(f"  {dataset_label}: {error}")
                continue
            lines.append(f"  {dataset_label}:")
            if not summaries:
                lines.append("    no model metrics found")
                continue
            for model_name in sorted(summaries.keys()):
                summary = summaries[model_name]
                if "error" in summary:
                    lines.append(f"    {model_name}: {summary['error']}")
                    continue
                lines.append(
                    "    "
                    f"{model_name}: best_factor={summary['best_factor']}"
                    f" best_mean={summary['best_mean']:.4f}±{summary['best_se']:.4f}"
                    f" baseline_mean={summary['baseline_mean']:.4f}±{summary['baseline_se']:.4f}"
                    f" diff={summary['diff']:.4f}"
                    f" p_one_sided={summary['p_value_one_sided']:.4g}"
                )
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
