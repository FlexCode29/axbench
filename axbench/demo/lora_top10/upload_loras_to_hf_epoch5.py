#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_root",
        default="axbench/demo/lora_top10/runs",
        type=str,
        help="Root runs directory containing lora_top10_cXX",
    )
    parser.add_argument(
        "--epoch_tag",
        default="epoch_5",
        type=str,
        help="Checkpoint tag to upload (e.g., epoch_5)",
    )
    parser.add_argument(
        "--namespace",
        default="",
        type=str,
        help="HF username or org (defaults to HF_NAMESPACE env)",
    )
    parser.add_argument(
        "--repo_prefix",
        default="axbench-lora-epoch5-",
        type=str,
        help="Prefix for HF repos",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repos (default public)",
    )
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="HF token (defaults to HF_TOKEN env)",
    )
    args = parser.parse_args()
    namespace = args.namespace or os.environ.get("HF_NAMESPACE")

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Missing --token (or set HF_TOKEN).")

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise SystemExit(f"runs_root not found: {runs_root}")

    ckpt_dirs = sorted(runs_root.rglob(f"{args.epoch_tag}__*"))
    ckpt_dirs = [d for d in ckpt_dirs if d.is_dir()]
    if not ckpt_dirs:
        raise SystemExit(f"No checkpoint dirs found for tag: {args.epoch_tag}")

    api = HfApi(token=token)
    links = []

    for ckpt_dir in ckpt_dirs:
        concept_slug = ckpt_dir.name.split("__", 1)[-1]
        repo_name = f"{args.repo_prefix}{concept_slug}"
        repo_id = f"{namespace}/{repo_name}"

        api.create_repo(
            repo_id=repo_id,
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(ckpt_dir),
            path_in_repo=".",
            commit_message=f"Upload {ckpt_dir.name}",
        )
        links.append(f"https://huggingface.co/{repo_id}")

    print("Uploaded repos:")
    for link in links:
        print(link)


if __name__ == "__main__":
    main()
