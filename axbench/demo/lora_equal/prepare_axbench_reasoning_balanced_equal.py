#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset


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


def iter_rows(dataset_dict):
    for split in dataset_dict.values():
        for row in split:
            yield row


def main():
    parser = argparse.ArgumentParser(
        description="Prepare balanced concept slices of axbench-reasoning in AxBench format."
    )
    parser.add_argument("--dataset", default="NONO/axbench-reasoning")
    parser.add_argument("--max_total", type=int, default=16000)
    parser.add_argument("--max_per_concept", type=int, default=20)
    parser.add_argument(
        "--concepts",
        default=",".join(DEFAULT_CONCEPTS),
        help="Comma-separated list of concept names to include.",
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

    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)

    concept_list = [c.strip() for c in args.concepts.split(",") if c.strip()]
    concept_set = set(concept_list)

    max_total = args.max_total
    max_per_concept = args.max_per_concept

    concept_counter = Counter()
    kept_rows = []

    for row in iter_rows(ds):
        if len(kept_rows) >= max_total:
            break
        concept = row.get("output_concept")
        if not concept or concept not in concept_set:
            continue
        if concept_counter[concept] >= max_per_concept:
            continue
        kept_rows.append(row)
        concept_counter[concept] += 1

    if not concept_counter:
        raise SystemExit("No rows collected; check concept names and dataset fields.")

    # Enforce equal counts by downsampling to min count
    min_count = min(concept_counter.values())
    per_concept_rows = defaultdict(list)
    for row in kept_rows:
        concept = row.get("output_concept")
        if concept in concept_set and len(per_concept_rows[concept]) < min_count:
            per_concept_rows[concept].append(row)

    balanced_rows = []
    for concept in concept_list:
        balanced_rows.extend(per_concept_rows.get(concept, []))

    print(f"Collected {len(balanced_rows)} examples")
    print(f"Per-concept count: {min_count}")

    balanced_ds = Dataset.from_list(balanced_rows)
    df = balanced_ds.to_pandas()
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

    concept_id_map = {concept: idx for idx, concept in enumerate(concept_list)}
    filtered_df = df[df["output_concept"].isin(concept_set)].copy()
    filtered_df["concept_id"] = filtered_df["output_concept"].map(concept_id_map)
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
        negative_df = filtered_df[filtered_df["category"] == "negative"].copy()
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
