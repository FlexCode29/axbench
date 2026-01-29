#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DEFAULT_EMPTY_CONCEPT = "EEEEE"
DEFAULT_CONCEPTS = [
    "Basic Arithmetic Reasoning",
    "Computer Programming Reasoning",
    "Epidemiology Reasoning",
    "Classical Mechanics Reasoning",
    "Organic Chemistry Reasoning",
    "Medieval European History Reasoning",
    "Constitutional Law Reasoning",
    "Narrative Structure Reasoning",
]


def pick_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def slugify(text, max_len=48):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    if not slug:
        slug = "concept"
    return slug[:max_len]


def normalize_category(value):
    value = str(value).strip().lower()
    if value in {"pos", "positive", "1", "true"}:
        return "positive"
    if value in {"neg", "negative", "0", "false"}:
        return "negative"
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Prepare top-10 concept slices (first N rows) of axbench-reasoning in AxBench format."
    )
    parser.add_argument("--dataset", default="NONO/axbench-reasoning")
    parser.add_argument("--split", default="train")
    parser.add_argument("--rows", type=int, default=16000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--concepts",
        default=",".join(DEFAULT_CONCEPTS),
        help="Comma-separated list of concept names to include instead of top-k.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--empty_concept", default=DEFAULT_EMPTY_CONCEPT)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--keep_extra_columns", action="store_true")
    parser.add_argument("--negative_policy", choices=["all", "by_genre", "none"], default="by_genre")
    parser.add_argument("--no_combined", action="store_true")
    parser.add_argument("--no_per_concept", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total_rows = len(dataset)
    use_rows = min(args.rows, total_rows)
    dataset = dataset.select(range(use_rows))

    df = dataset.to_pandas()
    columns = set(df.columns)

    concept_col = pick_column(columns, ["output_concept", "concept", "label", "concept_name"])
    input_col = pick_column(columns, ["input", "prompt", "question", "instruction"])
    output_col = pick_column(columns, ["output", "response", "answer", "completion"])
    category_col = pick_column(columns, ["category", "label_type", "split"])
    genre_col = pick_column(columns, ["concept_genre", "genre", "domain"])
    ref_col = pick_column(columns, ["ref", "sae_link"])

    missing = [name for name in [concept_col, input_col, output_col] if name is None]
    if missing:
        raise ValueError(
            "Missing required columns. Found columns: " + ", ".join(sorted(columns))
        )

    df = df.rename(
        columns={
            concept_col: "output_concept",
            input_col: "input",
            output_col: "output",
        }
    )
    if category_col and category_col != "category":
        df = df.rename(columns={category_col: "category"})
    if genre_col and genre_col != "concept_genre":
        df = df.rename(columns={genre_col: "concept_genre"})
    if ref_col and ref_col != "ref":
        df = df.rename(columns={ref_col: "ref"})

    if "category" not in df.columns:
        df["category"] = "positive"
    df["category"] = df["category"].map(normalize_category)

    if "concept_genre" not in df.columns:
        df["concept_genre"] = "text"

    df["output_concept"] = df["output_concept"].astype(str)
    df.loc[df["output_concept"] == args.empty_concept, "category"] = "negative"
    df.loc[df["category"] == "negative", "output_concept"] = args.empty_concept

    positive_df = df[df["category"] == "positive"]
    concept_counts = positive_df["output_concept"].value_counts()
    concept_counts = concept_counts[concept_counts.index != args.empty_concept]

    if args.concepts:
        requested = [c.strip() for c in args.concepts.split(",") if c.strip()]
        missing = [c for c in requested if c not in concept_counts.index]
        if missing:
            raise ValueError(
                "Requested concepts not found in first "
                f"{use_rows} rows: {missing}"
            )
        top_concepts = requested
    else:
        if len(concept_counts) < args.topk:
            raise ValueError(
                f"Not enough unique concepts ({len(concept_counts)}) for topk={args.topk}."
            )
        top_concepts = concept_counts.head(args.topk).index.tolist()

    selected_mask = df["output_concept"].isin(top_concepts)
    negative_mask = df["category"] == "negative"
    if args.empty_concept in df["output_concept"].unique():
        negative_mask = negative_mask | (df["output_concept"] == args.empty_concept)

    filtered_df = df[selected_mask | negative_mask].copy()

    concept_id_map = {concept: idx for idx, concept in enumerate(top_concepts)}
    filtered_df["concept_id"] = filtered_df["output_concept"].map(concept_id_map)
    filtered_df.loc[filtered_df["output_concept"] == args.empty_concept, "concept_id"] = -1
    filtered_df["concept_id"] = filtered_df["concept_id"].fillna(-1).astype(int)

    base_columns = [
        "input",
        "output",
        "output_concept",
        "category",
        "concept_id",
        "concept_genre",
    ]
    if args.keep_extra_columns:
        write_df = filtered_df
    else:
        extra_cols = [c for c in base_columns if c not in filtered_df.columns]
        if extra_cols:
            raise ValueError(f"Missing required columns after rename: {extra_cols}")
        write_df = filtered_df[base_columns]

    concept_stats = []
    metadata = []
    for concept, concept_id in concept_id_map.items():
        concept_rows = filtered_df[filtered_df["output_concept"] == concept]
        genres = sorted(set(concept_rows["concept_genre"].dropna().astype(str)))
        if not genres:
            genres = ["text"]
        ref = None
        if "ref" in filtered_df.columns:
            ref_candidates = concept_rows["ref"].dropna().astype(str).tolist()
            if ref_candidates:
                ref = ref_candidates[0]
        if not ref:
            ref = f"https://www.neuronpedia.org/placeholder/{concept_id}"
        metadata.append(
            {
                "concept_id": concept_id,
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: genres},
            }
        )
        concept_stats.append(
            {
                "concept_id": concept_id,
                "concept": concept,
                "rows": int(len(concept_rows)),
                "genres": genres,
            }
        )

    stats_path = out_dir / "concepts.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(concept_stats, f, ensure_ascii=True, indent=2)

    write_combined = not args.no_combined
    write_per_concept = not args.no_per_concept

    if write_combined:
        combined_dir = out_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        write_df.to_parquet(combined_dir / "train_data.parquet", index=False)
        with (combined_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for item in metadata:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")

    if write_per_concept:
        negative_df = filtered_df[negative_mask].copy()
        for concept, concept_id in concept_id_map.items():
            concept_rows = filtered_df[filtered_df["output_concept"] == concept]
            if args.negative_policy == "none":
                merged = concept_rows
            else:
                neg_rows = negative_df
                if args.negative_policy == "by_genre":
                    concept_genres = set(concept_rows["concept_genre"].dropna().astype(str))
                    neg_rows = negative_df[negative_df["concept_genre"].isin(concept_genres)]
                merged = pd.concat([concept_rows, neg_rows], ignore_index=True)

            if not args.keep_extra_columns:
                merged = merged[base_columns]

            slug = slugify(concept)
            concept_dir = out_dir / f"concept_{concept_id:02d}_{slug}"
            concept_dir.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(concept_dir / "train_data.parquet", index=False)
            with (concept_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
                for item in metadata:
                    f.write(json.dumps(item, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
