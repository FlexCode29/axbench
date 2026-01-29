# inference.py: Inference with existing subspaces.
#
# example launch command:
#     torchrun --nproc_per_node=NUM_GPUS axbench/scripts/inference.py --config axbench/demo/sweep/inference.yaml --mode latent
import os, argparse, yaml, json, glob, pickle, time, itertools, datetime
import shutil
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import atexit
import re
from datasets import load_dataset
from datasets import DownloadMode

from axbench.utils.dataset import (
    DatasetFactory,
    SteeringDatasetFactory
)
from axbench.utils.constants import * 
from axbench.utils.model_utils import get_prefix_length, get_suffix_length
from axbench.scripts.args.dataset_args import DatasetArgs
from axbench.scripts.args.training_args import TrainingArgs
from transformers import set_seed


# all supported methods
import axbench
from openai import AsyncOpenAI
import httpx, asyncio

import logging
import torch.distributed as dist
import sys

# Initialize the logger
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "inference_state.pkl"
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
STEERING_WITH_SHARED_MODELS = {"HyperSteer"}
STEERING_EXCLUDE_MODELS = {"IntegratedGradients", "InputXGradients", "PromptDetection", "BoW"}
LATENT_EXCLUDE_MODELS = {"PromptSteering", "PromptBaseline", "DiReFT", "LoReFT", "LoRA", "SFT", "HyperSteer"}
LATENT_PROMPT_PREFIX = "Generate a random sentence."

# --------------------- Benchmark Prompt Registry ---------------------
BENCHMARK_PROMPTS = {
    "gsm8k": {
        # "base": (
        #     "Question:\n{question}\n\n"
        #     "Please reason step by step. When you are done, give the final answer in one short sentence.\n\n"
        #     "Therefore, the answer is:"
        # ),
        # "steering": (
        #     "Question:\n{question}\n\n"
        #     "Please reason step by step. When you are done, give the final answer in one short sentence.\n\n"
        #     "Therefore, the answer is:"
        # ),
        "base": (
            "{question}\n\n"
            # "Therefore, the answer is:"
        ),
        "steering": (
            "{question}\n\n"
            # "Therefore, the answer is:"
        ),
        "concept": "Basic Arithmetic Reasoning",
    },
    "supergpqa": {
        # "base": (
        #     "Question:\n{question}\n\nOptions:\n{options}\n\n"
        #     "Please reason step by step. When you are done, give the final answer as the correct option letter (A-J).\n\n"
        #     "Therefore, the answer is:"
        # ),
        # "steering": (
        #     "Question:\n{question}\n\nOptions:\n{options}\n\n"
        #     "Please reason step by step. When you are done, give the final answer as the correct option letter (A-J).\n\n"
        #     "Therefore, the answer is:"
        # ),
        "base": (
            "Question:\n{question}\n\nOptions:\n{options}\n\n"
            "Report the final answer as the correct option letter (A-J).\n\n"
            "Response:"
        ),
        "steering": (
            "Question:\n{question}\n\nOptions:\n{options}\n\n"
            "Report give the final answer as the correct option letter (A-J).\n\n"
            "Response:"
        ),
        # concept intentionally omitted; handled by resolve_supergpqa_concept
    },
    "codemmlu": {
        "base": (
            "Question:\n{question}\n\nChoices:\n{options}\n\n"
            "Report the final answer as the correct option letter.\n\n"
            "Response:"
        ),
        "steering": (
            "Question:\n{question}\n\nChoices:\n{options}\n\n"
            "Report the final answer as the correct option letter.\n\n"
            "Response:"
        ),
        "concept": "Computer Science Reasoning",
    },
}

# --- Helper: SuperGPQA Concept Prompt Resolution ---
def resolve_supergpqa_concept(args, dataset_list):
    """
    Resolve the concept prompt for SuperGPQA.

    Priority:
      1) --concept_prompt (explicit override)
      2) --supergpqa_auto_concept -> '<subfield> Reasoning'
      3) Fallback: 'Medical Knowledge Reasoning'
    """
    if getattr(args, "concept_prompt", None):
        return args.concept_prompt

    if getattr(args, "supergpqa_auto_concept", False):
        if not dataset_list:
            raise ValueError("Cannot auto-derive concept: SuperGPQA dataset is empty.")
        subfield = dataset_list[0].get("subfield")
        if subfield is None:
            raise ValueError("SuperGPQA example missing 'subfield'.")
        return f"{subfield} Reasoning"

    return "Epidemiology Reasoning"

# --- Helper: robust dataset loading (avoids broken HF cache issues) ---
def safe_load_dataset(dataset_name, split, dump_dir=None, **kwargs):
    """Load an HF dataset with retries that bypass potentially corrupted caches.

    We occasionally see failures like:
      TypeError: must be called with a dataclass type or instance
    from `datasets` when it tries to read cached DatasetInfo/features.

    We also sometimes see cleanup races like:
      OSError: [Errno 39] Directory not empty: '<...>'
    during `download_and_prepare` when multiple runs share a cache directory.

    Strategy:
      1) Try normal load_dataset
      2) On known cache/feature errors, retry with an *isolated, unique* cache_dir under dump_dir
         and force redownload.
      3) If we hit an "Errno 39" cleanup race, delete the offending directory (best-effort)
         and retry with a fresh cache_dir.
    """
    # First attempt: normal cache behavior
    try:
        return load_dataset(dataset_name, split=split, **kwargs)
    except Exception as e:
        msg = str(e)
        known = (
            "must be called with a dataclass type" in msg
            or "DatasetInfo.from_directory" in msg
            or "Features.from_dict" in msg
        )
        if not known:
            raise

        # Retry with isolated per-run cache directories to avoid cache corruption and rmtree races.
        base_cache_dir = None
        if dump_dir is not None:
            base_cache_dir = Path(dump_dir) / "hf_datasets_cache"
            base_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.warning(
            f"[Warn] load_dataset('{dataset_name}', split='{split}') failed with a cache/features error; "
            f"retrying with force redownload and an isolated cache_dir under {str(base_cache_dir) if base_cache_dir else None!r}. "
            f"Original error: {msg}"
        )

        last_err = e
        for attempt in range(3):
            # Make the cache_dir unique per attempt to avoid partial/incomplete dir cleanup races.
            cache_dir = None
            if base_cache_dir is not None:
                unique = f"{dataset_name.replace('/', '___').replace(' ', '_')}_pid{os.getpid()}_{int(time.time()*1000)}_try{attempt}"
                cache_dir = str(base_cache_dir / unique)
                Path(cache_dir).mkdir(parents=True, exist_ok=True)

            try:
                return load_dataset(
                    dataset_name,
                    split=split,
                    cache_dir=cache_dir,
                    download_mode=DownloadMode.FORCE_REDOWNLOAD,
                    **kwargs,
                )
            except OSError as oe:
                # Sometimes HF datasets hits a cleanup race and fails to rmtree an incomplete dir.
                # Best-effort delete the directory from the error message, then retry.
                if getattr(oe, "errno", None) == 39:  # Directory not empty
                    oe_msg = str(oe)
                    logger.warning(
                        f"[Warn] load_dataset retry hit an incomplete_dir cleanup race (Errno 39). "
                        f"Will best-effort delete the offending directory and retry. Error: {oe_msg}"
                    )
                    # Try to extract the quoted path at the end of the message.
                    m = re.search(r"'([^']+)'\s*$", oe_msg)
                    if m:
                        bad_dir = m.group(1)
                        try:
                            shutil.rmtree(bad_dir, ignore_errors=True)
                        except Exception:
                            pass
                    last_err = oe
                    continue
                last_err = oe
                continue
            except Exception as e2:
                last_err = e2
                continue

        raise last_err

def load_config(config_path):
    """
    Load metadata from a JSON lines file.
    """
    if not os.path.exists(Path(config_path) / CONFIG_FILE):
        return None
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d


def load_state(dump_dir, mode, rank, subfolder="inference"):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}/{subfolder}", f"{mode}_{STATE_FILE}_rank_{rank}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def save_state(dump_dir, state, partition, rank):
    if not isinstance(dump_dir, Path):
        dump_dir = Path(dump_dir)
        
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}_rank_{rank}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    with open(Path(metadata_path) / METADATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref = data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": data["concept_id"]
            }
            metadata += [flatten_data]  # Return the metadata as is
    return metadata

# --- Helper: Load concepts directly from HF dataset if no metadata.jsonl ---
def load_hf_concepts(dataset_name):
    """
    Load concept registry directly from an HF dataset.
    Returns:
        dataset        : full HF dataset
        concept_ids    : sorted list of unique concept_ids
        num_concepts   : max(concept_id) + 1
        concept_map    : dict[int -> concept_name]
    """
    ds = load_dataset(dataset_name, split="train")
    # Use HF's optimized unique() when available
    try:
        concept_ids = sorted(ds.unique("concept_id"))
    except Exception:
        concept_ids = sorted(set(ds["concept_id"]))

    num_concepts = (max(concept_ids) + 1) if concept_ids else 0

    # Build concept_id -> output_concept map in ONE pass (first occurrence wins)
    concept_map = {}
    # Fast path: iterate indices to avoid Python generator restarting costs
    for i in range(len(ds)):
        cid = ds[i]["concept_id"]
        if cid not in concept_map:
            concept_map[cid] = ds[i].get("output_concept", f"concept_{cid}")
            if len(concept_map) == len(concept_ids):
                break

    logger.warning(f"[Info] Loaded {len(concept_ids)} concepts from HF dataset '{dataset_name}'.")
    return ds, concept_ids, num_concepts, concept_map

# --- Helper: Extract max_training_examples from training_args/config ---
def _extract_hypersteer_max_training_examples(training_args, config, dump_dir=None):
    """
    Best-effort extraction of HyperSteer max_training_examples from either
    TrainingArgs (if present) or the saved train config.json.
    Returns int or None.
    """
    # 1) Try TrainingArgs (may be missing if schema drops the field)
    try:
        hs = training_args.models.get("HyperSteer", None)
        if hs is not None and getattr(hs, "max_training_examples", None):
            return int(getattr(hs, "max_training_examples"))
    except Exception:
        pass

    # 2) Try common locations in config.json
    if isinstance(config, dict):
        # a) config["models"]["HyperSteer"]["max_training_examples"]
        try:
            v = config.get("models", {}).get("HyperSteer", {}).get("max_training_examples", None)
            if v is not None:
                return int(v)
        except Exception:
            pass
        # b) config["train"]["models"]["HyperSteer"]["max_training_examples"]
        try:
            v = config.get("train", {}).get("models", {}).get("HyperSteer", {}).get("max_training_examples", None)
            if v is not None:
                return int(v)
        except Exception:
            pass

        # c) last resort: recursive search for the key
        def _recurse(obj):
            if isinstance(obj, dict):
                if "max_training_examples" in obj:
                    return obj["max_training_examples"]
                for vv in obj.values():
                    out = _recurse(vv)
                    if out is not None:
                        return out
            elif isinstance(obj, list):
                for vv in obj:
                    out = _recurse(vv)
                    if out is not None:
                        return out
            return None

        v = _recurse(config)
        if v is not None:
            try:
                return int(v)
            except Exception:
                return None

    # 3) Heuristic: if dump_dir basename ends with "-<int>" (e.g., hypersteer-gemma2b-1000),
    # treat that as max_training_examples when schema/config does not expose it.
    try:
        if dump_dir is not None:
            base = os.path.basename(str(dump_dir).rstrip("/"))
            m = re.search(r"-(\d+)$", base)
            if m:
                return int(m.group(1))
    except Exception:
        pass

    return None


def save(
    dump_dir, partition,
    current_df, rank):
    # This function saves DataFrames per rank per partition (latent or steering)
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save DataFrame
    df_path = os.path.join(dump_dir, f"rank_{rank}_{partition}_data.parquet")
    
    if "defense" in current_df.columns:
        current_df["defense"] = current_df["defense"].apply(lambda x: "["+','.join(x)+"]" if isinstance(x, list) else str(x))
        
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        
        if "defense" in existing_df.columns:
            # Convert defense column to string format if it exists
            existing_df["defense"] = existing_df["defense"].apply(lambda x: "["+','.join(x)+"]" if isinstance(x, list) else str(x))
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df

    combined_df.to_parquet(df_path, engine='pyarrow')


def partition_concept_ids(concept_ids, world_size):
    concept_ids_per_rank = []
    n = len(concept_ids)
    chunk_size = n // world_size
    remainder = n % world_size
    start = 0
    for i in range(world_size):
        end = start + chunk_size + (1 if i < remainder else 0)
        concept_ids_per_rank.append(concept_ids[start:end])
        start = end
    return concept_ids_per_rank


# --------------------- Benchmark Evaluation Abstractions ---------------------

class BenchmarkRunner:
    def __init__(self, model, tokenizer, device, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def run_batches(self, prompts, max_new_tokens=256):
        results = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            toks = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            # GSM8K natural stopping: encourage natural stopping by detecting answer cues and integer answers.
            from transformers import StoppingCriteria, StoppingCriteriaList
            import torch
            import re
            class NaturalGSM8KStop(StoppingCriteria):
                def __init__(self, tokenizer, start_len, eos_token_id, prompt_lens, min_gen_tokens_floor=32):
                    super().__init__()
                    self.tokenizer = tokenizer
                    self.start_len = start_len
                    self.eos_token_id = eos_token_id
                    self.prompt_lens = prompt_lens
                    self.min_gen_tokens_floor = min_gen_tokens_floor
                    self.answer_cue_re = re.compile(
                        r"(therefore|final answer|answer is|the answer is|so the answer is)[^\d\-]*[\$\(]?\s*(-?\d[\d,]*)",
                        re.IGNORECASE
                    )
                    self.int_line_re = re.compile(r"^\s*-?\d[\d,]*\s*$")

                def __call__(self, input_ids, scores, **kwargs):
                    batch_size = input_ids.shape[0]
                    stop_batch = [False] * batch_size
                    for i, seq in enumerate(input_ids):
                        gen_len = seq.shape[0] - self.start_len
                        min_gen_tokens = max(self.min_gen_tokens_floor, int(0.75 * self.prompt_lens[i]))
                        if gen_len < min_gen_tokens:
                            continue
                        # decode only the newly generated suffix
                        generated_ids = seq[self.start_len:]
                        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        # 1. Stop if EOS token generated
                        if seq[-1].item() == self.eos_token_id:
                            stop_batch[i] = True
                            continue
                        # 2. Stop if answer cue followed by integer (optional $/comma/paren) found and min_gen_tokens reached
                        m = self.answer_cue_re.search(decoded)
                        if m:
                            stop_batch[i] = True
                            continue
                        # 3. Fallback: stop if decoded suffix ends with integer-only line (after min gen tokens)
                        lines = decoded.splitlines()
                        for line in reversed(lines):
                            if line.strip() != "":
                                if self.int_line_re.match(line):
                                    stop_batch[i] = True
                                break
                    return any(stop_batch)

            # Pre-compute per-example prompt lengths
            prompt_lens = toks["attention_mask"].sum(dim=1).tolist()
            stopping_criteria = StoppingCriteriaList([
                NaturalGSM8KStop(
                    self.tokenizer,
                    toks.input_ids.shape[1],
                    self.tokenizer.eos_token_id,
                    prompt_lens=prompt_lens,
                )
            ])

            with torch.no_grad():
                outputs = self.model.generate(
                    **toks,
                    max_new_tokens=max_new_tokens,
                    do_sample=(
                        "gemma-2-2b" in getattr(self.model.config, "name_or_path", "").lower()
                    ),
                    temperature=(
                        0.7 if "gemma-2-2b" in getattr(self.model.config, "name_or_path", "").lower() else 1.0
                    ),
                    top_p=(
                        0.95 if "gemma-2-2b" in getattr(self.model.config, "name_or_path", "").lower() else 1.0
                    ),
                    repetition_penalty=(
                        1.05 if "gemma-2-2b" in getattr(self.model.config, "name_or_path", "").lower() else 1.0
                    ),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # No post-generation truncation; return raw decoded outputs
            results.extend(decoded)
        return results


def parse_gsm8k_gold(answer):
    """
    Robustly parse the GSM8K gold answer.
    Handles commas (e.g. '2,125'), negatives, and stray whitespace.
    """
    if "####" in answer:
        tail = answer.split("####")[-1]
    else:
        tail = answer

    nums = re.findall(r"-?\d+", tail.replace(",", ""))
    if not nums:
        return None
    return int(nums[-1])

def parse_supergpqa_pred(text):
    if text is None:
        return None
    m = re.search(r"\b([A-J])\b", text.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None

# --- CodeMMLU prediction parser ---
def parse_codemmlu_pred(text):
    if text is None:
        return None
    m = re.search(r"\b([A-D])\b", text.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None

# --- Helper: Binomial 95% Wilson confidence interval ---
def binomial_ci_95(correct, total):
    """
    Compute a 95% Wilson confidence interval for a binomial proportion.
    Returns (low, high). If total == 0, returns (0.0, 0.0).
    """
    if total == 0:
        return 0.0, 0.0
    import math
    z = 1.96
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (
        z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    ) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def parse_gsm8k_pred(text):
    """
    Parse the predicted GSM8K answer robustly.

    Priority order:
      1) If the model outputs a GSM8K-style marker, prefer the integer after '####'.
      2) If the text contains an '=', take the substring after the last '=' and extract the first integer (allow $/paren/comma/punct).
      3) Scan lines bottom-up and return the first line that is a *pure integer*.
      4) Fallback: return the last "answer-like" integer near the end of the text,
         avoiding common pitfalls:
           - decimals (e.g. 0.67 should not yield 67)
           - digits glued to letters (e.g. mwm654)
           - intermediate fractions (e.g. 3/60)
         while allowing trailing punctuation like '$460.'.
    """
    if text is None:
        return None

    clean = text.replace(",", "")
    # 1) GSM8K marker
    if "####" in clean:
        tail = clean.split("####")[-1]
        m = re.search(r"-?\d+", tail)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass

    # 2) Highest-priority: after last '=' extract first integer (allow $/paren/commas/trailing punct)
    if "=" in clean:
        tail = clean.split("=")[-1]
        # Remove leading/trailing whitespace and possible currency/paren
        tail = tail.strip()
        # Look for $ or ( or whitespace, then integer, possibly with trailing . or )
        # e.g. = $460. or = (460)
        m = re.search(r"[\$\(\s]*(-?\d+)", tail)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

    # 3) Integer-only line (canonical)
    lines = clean.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)

    # 4) Fallback: scan integer tokens with context-aware filtering
    # Token pattern: optional sign + digits, bounded so it's not glued to letters/digits.
    # Allow trailing punctuation (.,;:!?) and currency symbols around it.
    token_iter = list(re.finditer(r"(?<![A-Za-z0-9])(-?\d+)(?![A-Za-z0-9])", clean))
    if not token_iter:
        return None

    def _is_decimal_like(s, start, end):
        # Exclude if part of a decimal like 0.67 or 3.0
        # If immediately preceded by '.' OR immediately followed by '.' + digit => decimal component.
        if start - 1 >= 0 and s[start - 1] == ".":
            return True
        if end < len(s) and s[end] == "." and (end + 1) < len(s) and s[end + 1].isdigit():
            return True
        return False

    def _is_glued_to_letter(s, start, end):
        # Exclude if adjacent to letters (e.g., mwm654 or 654abc)
        if start - 1 >= 0 and s[start - 1].isalpha():
            return True
        if end < len(s) and s[end].isalpha():
            return True
        return False

    def _is_fraction_denominator(s, start, end):
        # Exclude if immediately preceded by '/' (denominator) or immediately followed by '/' (numerator)
        if start - 1 >= 0 and s[start - 1] == "/":
            return True
        if end < len(s) and s[end] == "/":
            return True
        return False

    # Prefer integers that appear late and are "answer-like":
    # - near 'final', 'answer', or after '='
    # We'll scan from the end and take the first token that passes filters,
    # with a mild preference for answer-ish local context.
    answerish_words = ("final", "answer", "therefore", "thus", "so", "=", "####")
    for m in reversed(token_iter):
        start, end = m.start(1), m.end(1)
        if _is_decimal_like(clean, start, end):
            continue
        if _is_glued_to_letter(clean, start, end):
            continue
        if _is_fraction_denominator(clean, start, end):
            continue

        # Check local context window
        left = max(0, start - 24)
        right = min(len(clean), end + 24)
        ctx = clean[left:right].lower()
        # If it's very late in the string, accept; otherwise require a small hint
        if end >= len(clean) - 128 or any(w in ctx for w in answerish_words):
            try:
                return int(m.group(1))
            except Exception:
                continue

    # Final fallback: last token that passes filters (even if not near end/keywords)
    for m in reversed(token_iter):
        start, end = m.start(1), m.end(1)
        if _is_decimal_like(clean, start, end):
            continue
        if _is_glued_to_letter(clean, start, end):
            continue
        if _is_fraction_denominator(clean, start, end):
            continue
        try:
            return int(m.group(1))
        except Exception:
            continue

    return None


# --------------------- Benchmark Inference Function ---------------------

def infer_benchmark(args, rank, world_size, device, logger, training_args):
    # --- Sanity check: fail fast if use_steering is set but no steering is applied here ---
    if getattr(args, "use_steering", False):
        raise RuntimeError(
            "Sanity check failed: use_steering=True in benchmark mode, "
            "but infer_benchmark does not apply HyperSteer. "
            "Use infer_steering or implement a steered benchmark path."
        )

    # Load dataset according to benchmark
    if args.benchmark == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
        if getattr(args, "max_questions", None) is not None:
            dataset = dataset.select(range(min(len(dataset), args.max_questions)))
        # shard dataset across ranks
        dataset = dataset.shard(num_shards=world_size, index=rank)
        dataset_list = list(dataset)
    elif args.benchmark == "supergpqa":
        dataset = safe_load_dataset("m-a-p/SuperGPQA", split="train", dump_dir=args.dump_dir)
        # Filter by discipline and field if specified
        if getattr(args, "supergpqa_discipline", None) is not None:
            want = str(args.supergpqa_discipline).strip().lower()
            dataset = dataset.filter(lambda ex: str(ex.get("discipline", "")).strip().lower() == want)
        if getattr(args, "supergpqa_field", None) is not None:
            want = str(args.supergpqa_field).strip().lower()
            dataset = dataset.filter(lambda ex: str(ex.get("field", "")).strip().lower() == want)
        if getattr(args, "max_questions", None) is not None:
            dataset = dataset.select(range(min(len(dataset), args.max_questions)))
        # shard dataset across ranks
        dataset = dataset.shard(num_shards=world_size, index=rank)
        dataset_list = list(dataset)
    elif args.benchmark == "codemmlu":
        dataset = load_dataset("Fsoft-AIC/CodeMMLU", "software_principles")
        if getattr(args, "max_questions", None) is not None:
            dataset = dataset.select(range(min(len(dataset), args.max_questions)))
        # shard dataset across ranks
        dataset = dataset.shard(num_shards=world_size, index=rank)
        dataset_list = list(dataset)
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if getattr(args, "use_bf16", False) else None,
        device_map=device
    ).eval()

    # --- Sanity check: log that this is a vanilla model ---
    logger.warning(
        f"[SanityCheck] Benchmark model class = {model.__class__.__name__}. "
        "This should be a plain HF model (no HyperSteer hooks)."
    )

    runner = BenchmarkRunner(
        model,
        tokenizer,
        device,
        batch_size=getattr(args, "benchmark_batch_size", 8)
    )

    # Use centralized benchmark prompt registry
    prompt_cfg = BENCHMARK_PROMPTS.get(args.benchmark)
    if prompt_cfg is None:
        raise ValueError(f"No prompt configuration found for benchmark: {args.benchmark}")

    template = prompt_cfg["steering"] if getattr(args, "use_steering", False) else prompt_cfg["base"]
    if args.benchmark == "gsm8k":
        prompts = [
            template.format(question=ex["question"])
            for ex in dataset_list
        ]
    elif args.benchmark == "supergpqa":
        prompts = [
            template.format(
                question=ex["question"],
                options="\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(ex["options"])])
            )
            for ex in dataset_list
        ]
    elif args.benchmark == "codemmlu":
        prompts = [
            template.format(
                question=ex["question"],
                options="\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(ex["choices"])])
            )
            for ex in dataset_list
        ]
    # For SuperGPQA, set concept_prompt using helper
    if args.benchmark == "supergpqa":
        args.concept_prompt = resolve_supergpqa_concept(args, dataset_list)
        logger.warning(f"[Benchmark] Using concept_prompt='{args.concept_prompt}'")

    outputs = runner.run_batches(
        prompts,
        max_new_tokens=int(getattr(args, "benchmark_output_length", 256))
    )

    correct = 0
    total = 0
    records = []

    pbar = tqdm(
        zip(dataset_list, outputs),
        total=len(dataset_list),
        desc=f"{args.benchmark.upper()} Rank {rank}",
        leave=True,
    )

    for ex, out in pbar:
        prompt_str = prompts[total]
        resp = out
        if args.benchmark == "gsm8k":
            gold = parse_gsm8k_gold(ex["answer"])
            pred = parse_gsm8k_pred(resp)
            is_correct = (pred == gold) and (gold is not None) and (pred is not None)
        elif args.benchmark == "supergpqa":
            gold = ex["answer_letter"]
            pred = parse_supergpqa_pred(resp)
            is_correct = (pred == gold) and (gold is not None) and (pred is not None)
        elif args.benchmark == "codemmlu":
            gold = ex["answer"]
            pred = parse_codemmlu_pred(resp)
            is_correct = (pred == gold) and (gold is not None) and (pred is not None)
        correct += int(is_correct)
        total += 1

        rec = {
            "question": ex["question"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "response": resp,
            "response_length": len(resp),
        }
        if args.benchmark == "supergpqa":
            rec["options"] = ex["options"]
            rec["answer_letter"] = ex["answer_letter"]
            rec["discipline"] = ex["discipline"]
            rec["field"] = ex["field"]
        if args.benchmark == "codemmlu":
            rec["choices"] = ex["choices"]
            rec["answer"] = ex["answer"]
            rec["task_id"] = ex.get("task_id")
        records.append(rec)

        if total > 0:
            pbar.set_postfix(acc=f"{correct / total:.3f}")

    correct_t = torch.tensor(correct, device=device)
    total_t = torch.tensor(total, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)

    if rank == 0:
        acc = correct_t.item() / max(1, total_t.item())
        ci_low, ci_high = binomial_ci_95(correct_t.item(), total_t.item())
        logger.warning(
            f"[Benchmark:{args.benchmark.upper()}] Accuracy={acc:.4f} "
            f"95% CI=({ci_low:.4f}, {ci_high:.4f}) "
            f"({correct_t.item()}/{total_t.item()})"
        )
        out_dir = Path(args.dump_dir) / f"benchmark_{args.benchmark}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{args.benchmark}_results.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        summary = {
            "benchmark": args.benchmark,
            "accuracy": acc,
            "num_questions": total_t.item(),
            "ci_95": {"low": ci_low, "high": ci_high}
        }
        if args.benchmark == "supergpqa":
            summary["discipline"] = getattr(args, "supergpqa_discipline", None)
            summary["field"] = getattr(args, "supergpqa_field", None)
        with open(out_dir / f"{args.benchmark}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


# --------------------- Benchmark Steered Inference Function ---------------------

def infer_benchmark_steered(args, rank, world_size, device, logger, training_args):
    assert world_size == 1, "benchmark_steered currently supports single-process only"

    # Load dataset according to benchmark
    if args.benchmark == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
        if getattr(args, "max_questions", None) is not None:
            dataset = dataset.select(range(min(len(dataset), args.max_questions)))
        dataset_list = list(dataset)
    elif args.benchmark == "supergpqa":
        dataset = safe_load_dataset("m-a-p/SuperGPQA", split="train", dump_dir=args.dump_dir)

        # Case-insensitive filtering (dataset values are usually Title Case like "Medicine")
        if getattr(args, "supergpqa_discipline", None) is not None:
            want = str(args.supergpqa_discipline).strip().lower()
            dataset = dataset.filter(lambda ex: str(ex.get("discipline", "")).strip().lower() == want)

        if getattr(args, "supergpqa_field", None) is not None:
            want = str(args.supergpqa_field).strip().lower()
            dataset = dataset.filter(lambda ex: str(ex.get("field", "")).strip().lower() == want)

        if getattr(args, "supergpqa_subfield", None) is not None:
            want = str(args.supergpqa_subfield).strip().lower()
            dataset = dataset.filter(lambda ex: str(ex.get("subfield", "")).strip().lower() == want)

        if len(dataset) == 0:
            raise ValueError(
                "SuperGPQA filter produced 0 examples. "
                f"discipline={getattr(args, 'supergpqa_discipline', None)!r}, "
                f"field={getattr(args, 'supergpqa_field', None)!r}, "
                f"subfield={getattr(args, 'supergpqa_subfield', None)!r}. "
                "Try different casing or remove the filter."
            )

        if getattr(args, "max_questions", None) is not None:
            dataset = dataset.select(range(min(len(dataset), args.max_questions)))

        logger.warning(f"[Benchmark:SUPERGPQA] Loaded {len(dataset)} examples after filtering (before max_questions).")
        dataset_list = list(dataset)
    elif args.benchmark == "codemmlu":
        # CodeMMLU uses HF "config" names for subsets; "execution_prediction" is a config, not a split.
        # The correct signature is load_dataset(path, name, split=...).
        dataset = load_dataset("Fsoft-AIC/CodeMMLU", "execution_prediction", split="test")
        if getattr(args, "max_questions", None) is not None:
            dataset = dataset.select(range(min(len(dataset), args.max_questions)))
        dataset_list = list(dataset)
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    if len(dataset_list) == 0:
        raise ValueError(f"No benchmark examples loaded for {args.benchmark}. Check your filters and max_questions.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if getattr(args, "use_bf16", False) else None,
        device_map=device
    ).eval()

    # ---- Load HyperSteer ----
    from axbench.models.hypersteer import HyperSteer

    hs_args = training_args.models["HyperSteer"]

    benchmark_model = HyperSteer(
        model,
        tokenizer,
        layer=args.steering_layer,
        low_rank_dimension=1,
        device=device,
        training_args=hs_args,
        lm_model_name=training_args.model_name,
    )

    benchmark_model._sanity_hook_called = False

    benchmark_model.load(
        dump_dir=args.train_dir,
        low_rank_dimension=1,
        mode="steering",
        hypernet_initialize_from_pretrained=hs_args.hypernet_initialize_from_pretrained,
        hypernet_name_or_path=hs_args.hypernet_name_or_path,
        num_hidden_layers=hs_args.num_hidden_layers,
    )

    # Wrap hook to assert steering fires
    orig_forward = benchmark_model.ax.forward
    def _wrapped_forward(*a, **kw):
        benchmark_model._sanity_hook_called = True
        return orig_forward(*a, **kw)
    benchmark_model.ax.forward = _wrapped_forward

    # ---- Build prompts ----
    prompt_cfg = BENCHMARK_PROMPTS[args.benchmark]
    if args.benchmark == "gsm8k":
        prompts = [
            prompt_cfg["base"].format(question=ex["question"])
            for ex in dataset_list
        ]
        input_concepts = [str("Basic Arithmetic Reasoning")] * len(prompts)
    elif args.benchmark == "supergpqa":
        prompts = [
            prompt_cfg["base"].format(
                question=ex["question"],
                options="\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(ex["options"])])
            )
            for ex in dataset_list
        ]
        # Resolve concept per-example
        input_concepts = []
        for ex in dataset_list:
            if getattr(args, "supergpqa_auto_concept", False):
                subfield = ex.get("subfield")
                if subfield is None or str(subfield).strip() == "":
                    input_concept = "Epidemiology Reasoning"
                else:
                    input_concept = f"{str(subfield)} Reasoning"
            else:
                cp = getattr(args, "concept_prompt", None)
                if cp is None or str(cp).strip() == "":
                    input_concept = "Epidemiology Reasoning"
                else:
                    input_concept = str(cp)
            input_concepts.append(str(input_concept))
        # Logging: state that supergpqa_auto_concept is being resolved per-example
        unique_concepts = list({c for c in input_concepts})
        logger.warning(
            f"[Benchmark+Steering] Resolving supergpqa_auto_concept per-example. "
            f"First 3 unique concepts: {unique_concepts[:3]}"
        )
    elif args.benchmark == "codemmlu":
        prompts = [
            prompt_cfg["base"].format(
                question=ex["question"],
                options="\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(ex["choices"])])
            )
            for ex in dataset_list
        ]
        base_concept = getattr(args, "concept_prompt", None)
        if base_concept is None or str(base_concept).strip() == "":
            base_concept = "Data Structures Reasoning"
        base_concept = str(base_concept)
        input_concepts = [base_concept for _ in prompts]

    # Defensive: HyperSteer concept tokenizer requires strings
    bad = [c for c in input_concepts if not isinstance(c, str)]
    if bad:
        raise ValueError(
            f"Non-string concept prompts detected (n={len(bad)}). "
            f"Example type={type(bad[0])}, value={bad[0]!r}."
        )

    # ---- Parse steering factors ----
    if getattr(args, "steering_factors", None) is None:
        steering_factors = [1.0]
    elif isinstance(args.steering_factors, str):
        steering_factors = [float(x) for x in args.steering_factors.split(",") if x.strip() != ""]
    else:
        steering_factors = list(args.steering_factors)

    batch_size = int(getattr(args, "benchmark_batch_size", 8))
    # Prefer benchmark_output_length for benchmark runs; fallback to steering_output_length; then 256.
    eval_output_length = int(
        (getattr(args, "benchmark_output_length", None)
         or getattr(args, "steering_output_length", None)
         or 256)
    )

    # ---- Set prefix_length for chat models ----
    prefix_length = 1
    try:
        if args.model_name in CHAT_MODELS:
            prefix_length = get_prefix_length(tokenizer)
    except Exception:
        pass

    # Gemma-2-2B benefits from mild sampling for GSM8K-style reasoning
    if "gemma-2-2b" in args.model_name.lower():
        args.temperature = 0.7
        args.top_p = 0.95
        args.do_sample = True
    else:
        args.temperature = 1.0
        args.top_p = 1.0
        args.do_sample = False

    sweep_summary = []
    out_dir = Path(args.dump_dir) / f"benchmark_{args.benchmark}_steered"
    out_dir.mkdir(parents=True, exist_ok=True)

    for factor in steering_factors:
        df = pd.DataFrame({
            "input": prompts,
            "output": ["" for _ in prompts],
            "input_concept": input_concepts,
            "concept_id": [0 for _ in prompts],
            "factor": [float(factor) for _ in prompts],
            "input_id": list(range(len(prompts))),
        })

        results = benchmark_model.predict_steer(
            df,
            concept_id=0,
            sae_link=None,
            sae_id=None,
            batch_size=batch_size,
            eval_output_length=eval_output_length,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            do_sample=bool(args.do_sample),
            prefix_length=prefix_length,
            positions=hs_args.intervention_positions,
            use_synergy=False,
            disable_neuronpedia_max_act=getattr(args, "disable_neuronpedia_max_act", False),
            intervene_on_prompt=False,
            return_vector=False,
        )

        # --- Robust extraction of generations across model implementations ---
        outputs = None
        if isinstance(results, dict):
            # Common keys
            for k in ["steered_generation", "generation", "output", "outputs", "text"]:
                v = results.get(k, None)
                if isinstance(v, list) and len(v) == len(prompts):
                    outputs = v
                    break
            # Fallback: pick any list-valued entry that matches expected length
            if outputs is None:
                for k, v in results.items():
                    if isinstance(v, list) and len(v) == len(prompts):
                        outputs = v
                        logger.warning(f"[Benchmark+Steering] Using generations from results['{k}']")
                        break

        if outputs is None:
            available = list(results.keys()) if isinstance(results, dict) else type(results)
            raise RuntimeError(
                "HyperSteer.predict_steer did not return a usable generations list. "
                f"Expected a list of length {len(prompts)} under a known key. "
                f"Available keys/type: {available}"
            )

        correct, total = 0, 0
        records = []

        for ex, out in zip(dataset_list, outputs):
            if args.benchmark == "gsm8k":
                gold = parse_gsm8k_gold(ex["answer"])
                pred = parse_gsm8k_pred(out)
                ok = (gold is not None) and (pred == gold)
            elif args.benchmark == "supergpqa":
                gold = ex["answer_letter"]
                pred = parse_supergpqa_pred(out)
                ok = (gold is not None) and (pred == gold)
            elif args.benchmark == "codemmlu":
                gold = ex["answer"]
                pred = parse_codemmlu_pred(out)
                ok = (gold is not None) and (pred == gold)
            correct += int(ok)
            total += 1
            rec = {
                "question": ex["question"],
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "factor": factor,
                "response": out,
                "response_length": len(out),
            }
            if args.benchmark == "supergpqa":
                rec["options"] = ex["options"]
                rec["answer_letter"] = ex["answer_letter"]
                rec["discipline"] = ex["discipline"]
                rec["field"] = ex["field"]
            if args.benchmark == "codemmlu":
                rec["choices"] = ex["choices"]
                rec["answer"] = ex["answer"]
                rec["task_id"] = ex.get("task_id")
            records.append(rec)

        acc = correct / max(1, total)
        ci_low, ci_high = binomial_ci_95(correct, total)
        logger.warning(
            f"[Benchmark:{args.benchmark.upper()}+Steering] "
            f"factor={factor} Accuracy={acc:.4f} "
            f"95% CI=({ci_low:.4f}, {ci_high:.4f}) "
            f"({correct}/{total})"
        )

        with open(out_dir / f"{args.benchmark}_results_factor_{factor}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        summary = {
            "benchmark": args.benchmark,
            "steered": True,
            "factor": factor,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "ci_95": {"low": ci_low, "high": ci_high}
        }
        if args.benchmark == "supergpqa":
            summary["discipline"] = getattr(args, "supergpqa_discipline", None)
            summary["field"] = getattr(args, "supergpqa_field", None)
        # For codemmlu, no extra fields required; just write summary.
        with open(out_dir / f"{args.benchmark}_summary_factor_{factor}.json", "w") as f:
            json.dump(summary, f, indent=2)

        sweep_summary.append({
            "factor": factor,
            "accuracy": acc,
            "ci_95": {"low": ci_low, "high": ci_high},
            "correct": correct,
            "total": total,
        })

    # Save sweep summary
    with open(out_dir / f"{args.benchmark}_sweep_summary.json", "w") as f:
        json.dump(sweep_summary, f, indent=2)


def create_data_latent(dataset_factory, metadata, concept_id, num_of_examples, args):
    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    sae_id = int(sae_link.split("/")[-1]) 
    concept_genres_map = metadata[concept_id]["concept_genres_map"]
    _, eval_contrast_concepts_map = \
        dataset_factory.prepare_concepts(
            [concept], 
            concept_genres_map=concept_genres_map,
            contrast_concepts_map={}, api_tag="inference")
    current_df = dataset_factory.create_eval_df(
        [concept], num_of_examples, concept_genres_map, {},
        eval_contrast_concepts_map, input_length=args.input_length, 
        output_length=args.output_length, concept_id=concept_id
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id
    return current_df


def create_data_steering(
    dataset_factory, metadata, concept_id, num_of_examples, 
    n_steering_factors, steering_datasets, args, generate_args):

    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    try:
        sae_id = int(sae_link.split("/")[-1]) 
    except:
        sae_id = 0

    current_df = dataset_factory.create_eval_df(
        [concept], num_of_examples, n_steering_factors, steering_datasets, concept_id=concept_id,
        steering_model_name=args.steering_model_name, steer_data_type=generate_args.steer_data_type,
        n_shots=args.n_shot, defense=args.defense, dump_dir=args.dump_dir, multishot_factors_parquet=args.multishot_factors_parquet,
        suppress_eval_dir=args.suppress_eval_dir
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id

    return current_df, (concept_id, sae_link, sae_id)


def prepare_df(current_df, tokenizer, is_chat_model, model_name):
    suffix_length, _ = get_suffix_length(tokenizer)
    if is_chat_model:
        if model_name == "meta-llama/Llama-3.1-8B-Instruct":
            def apply_chat_template(row):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]}
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[1:-suffix_length]
                return tokenizer.decode(tokens)
            current_df['input'] = current_df.apply(apply_chat_template, axis=1)
        else:
            def apply_chat_template(row):
                messages = [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]}
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[1:-suffix_length]
                return tokenizer.decode(tokens)
            current_df['input'] = current_df.apply(apply_chat_template, axis=1)
    return current_df


def infer_steering(args, rank, world_size, device, logger, training_args, generate_args, suppress_eval_dir=None):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    overwrite_inference_dump_dir = Path(args.overwrite_inference_dump_dir) if args.overwrite_inference_dump_dir is not None else Path(dump_dir) / "inference"
    num_of_examples = args.steering_num_of_examples
    config = load_config(train_dir)
    # --- Load metadata.jsonl if available, else load from HF dataset ---
    metadata_path = Path(data_dir) / METADATA_FILE
    if metadata_path.exists():
        metadata = load_metadata_flatten(data_dir)
        concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]
        num_concepts = len(metadata)
        concept_name_map = {m["concept_id"]: m["concept"] for m in metadata}
        hf_df = None
    else:
        logger.warning(
            "[Info] metadata.jsonl not found — loading concepts from HF dataset "
            "'NONO/axbench-reasoning'."
        )
        hf_dataset, concept_ids, num_concepts, concept_name_map = load_hf_concepts(
            "NONO/axbench-reasoning"
        )
        metadata = None
        # Materialize HF dataset to pandas for fast per-concept slicing (58k rows is fine)
        hf_df = hf_dataset.to_pandas()
        # Ensure required columns exist
        for col in ["input", "output", "output_concept", "concept_id"]:
            if col not in hf_df.columns:
                raise ValueError(f"HF dataset missing required column: {col}")
        # Respect training-time max_training_examples when training used only a prefix of the HF dataset.
        hs_max_train = _extract_hypersteer_max_training_examples(
            training_args, config, dump_dir=dump_dir
        )
        if hs_max_train is not None:
            hf_df = hf_df.head(hs_max_train).reset_index(drop=True)
            logger.warning(
                f"[Info] HF-only steering: restricting HF examples to the first "
                f"{hs_max_train} rows to match HyperSteer training (max_training_examples)."
            )
        # In HF-only mode, HyperSteer may be trained with low_rank_dimension=1.
        # We default to steering a single "concept slot" (concept_id=0) using a chosen concept prompt.
        if not hasattr(args, "concept_prompt") or args.concept_prompt is None:
            args.concept_prompt = "Basic Arithmetic Reasoning"
        # If the requested concept isn't present in the training slice, auto-pick the most frequent concept.
        if args.concept_prompt is not None:
            target_concept = str(args.concept_prompt).strip()
            present = (hf_df["output_concept"].astype(str).str.strip() == target_concept).sum()
            if present == 0:
                # Pick the most frequent concept in the slice to ensure non-empty eval data.
                vc = hf_df["output_concept"].astype(str).str.strip().value_counts()
                if len(vc) == 0:
                    raise ValueError("[HF-only steering] No output_concept values found in HF training slice.")
                fallback = str(vc.index[0])
                logger.warning(
                    f"[Warn] HF-only steering: concept_prompt='{target_concept}' has 0 rows in the first {len(hf_df)} "
                    f"training examples. Falling back to concept_prompt='{fallback}'."
                )
                args.concept_prompt = fallback
        logger.warning(f"[Info] HF-only steering: using concept_prompt='{str(args.concept_prompt).strip()}' for eval data.")
    layer = int(args.steering_layer) if args.steering_layer is not None else config["layer"] if config else 0  # default layer for prompt baselines
    steering_layers = args.steering_layers if args.steering_layers is not None else [layer]
    steering_factors = args.steering_factors
    steering_datasets = args.steering_datasets

    # (Optional safety) Ensure num_of_examples defaults to a positive integer in HF-only mode
    use_hf_only = (metadata is None)
    if use_hf_only and (num_of_examples is None):
        # In HF-only mode we need a finite number of examples to build eval data.
        num_of_examples = 32
        logger.warning("[Info] HF-only steering: steering_num_of_examples was None; defaulting to 32.")

    # Defaults for steering generation parameters
    if args.steering_batch_size is None:
        args.steering_batch_size = 8
        logger.warning("[Info] steering_batch_size was None; defaulting to 8.")
    if args.steering_output_length is None:
        # fall back to generic output_length if present, else 128
        fallback_out = getattr(args, "output_length", None)
        args.steering_output_length = int(fallback_out) if fallback_out is not None else 128
        logger.warning(f"[Info] steering_output_length was None; defaulting to {args.steering_output_length}.")

    state = load_state(args.dump_dir, "steering", rank)
    last_concept_id_processed = state.get("last_concept_id", None) if state else None
    logger.warning(f"Rank {rank} last concept_id processed: {last_concept_id_processed}")

    # If we are in HF-only mode, HyperSteer concept space must match the trained low_rank_dimension.
    use_hf_only = (metadata is None)
    trained_hypersteer_dim = None
    try:
        if hasattr(training_args, "models") and "HyperSteer" in training_args.models:
            v = getattr(training_args.models["HyperSteer"], "low_rank_dimension", None)
            if v is not None:
                trained_hypersteer_dim = int(v)
    except Exception:
        trained_hypersteer_dim = None

    # Fallback: read from saved train config.json if TrainingArgs dropped the field
    if trained_hypersteer_dim is None and isinstance(config, dict):
        try:
            v = (
                config.get("models", {}).get("HyperSteer", {}).get("low_rank_dimension", None)
                or config.get("train", {}).get("models", {}).get("HyperSteer", {}).get("low_rank_dimension", None)
            )
            if v is not None:
                trained_hypersteer_dim = int(v)
        except Exception:
            pass

    # Final fallback: HF-only HyperSteer runs default to a single concept slot
    if trained_hypersteer_dim is None and (metadata is None) and ("HyperSteer" in getattr(args, "models", [])):
        trained_hypersteer_dim = 1

    if use_hf_only:
        # HyperSteer-only runs often have a single concept slot; map HF examples into concept_id=0.
        if trained_hypersteer_dim is not None and trained_hypersteer_dim == 1:
            concept_ids = [0]
            num_concepts = 1
            logger.warning(
                "[Info] HF-only steering: HyperSteer low_rank_dimension=1 detected; "
                "mapping selected HF examples to concept_id=0."
            )
        elif trained_hypersteer_dim is not None:
            num_concepts = int(trained_hypersteer_dim)
            # We still default to concept_id=0 for now unless you explicitly extend this.
            concept_ids = [0]
            logger.warning(
                f"[Info] HF-only steering: using trained HyperSteer low_rank_dimension={num_concepts}; "
                "defaulting to concept_id=0."
            )

    # Get list of all concept_ids (now always sorted unique)
    concept_ids = sorted(set(concept_ids))
    if args.max_concepts is not None:
        concept_ids = concept_ids[: int(args.max_concepts)]
        logger.warning(f"[Info] Capping to max_concepts={args.max_concepts}.")

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx+1:]
        else:
            # If last_concept_id_processed is not in my_concept_ids, process all
            pass

    if len(my_concept_ids) == 0:

        # Synchronize all processes
        dist.barrier()

        # Rank 0 merges results
        if rank == 0:
            logger.warning("Rank 0 is merging results.")
            # Merge per-rank results
            all_parquet_files = list((overwrite_inference_dump_dir).glob("rank_*_steering_data.parquet"))
            # Parse filenames to extract rank
            import re
            pattern = re.compile(r'rank_(\d+)_steering_data\.parquet')

            file_info_list = []
            for parquet_file in all_parquet_files:
                match = pattern.match(parquet_file.name)
                if match:
                    rank_str = match.group(1)
                    rank_int = int(rank_str)
                    file_info_list.append({
                        'rank': rank_int,
                        'file': parquet_file
                    })
                else:
                    logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

            # Sort the file_info_list by rank
            file_info_list.sort(key=lambda x: x['rank'])

            # Read and concatenate dataframes
            dfs = []
            for info in file_info_list:
                df = pd.read_parquet(info['file'])
                dfs.append(df)
            if len(dfs) > 0:
                combined_df = pd.concat(dfs, ignore_index=True)
                # Optionally sort combined_df by 'concept_id' if needed
                combined_df = combined_df.sort_values(by=['concept_id', 'input_id', 'factor']).reset_index(drop=True)
                combined_df.to_parquet(overwrite_inference_dump_dir / "steering_data.parquet", engine='pyarrow')
                logger.warning(f"Saved combined steering inference results to {overwrite_inference_dump_dir / 'steering_data.parquet'}")
            else:
                logger.warning("No results to merge.")

            # Optionally, delete per-rank files
            for info in file_info_list:
                os.remove(info['file'])
                logger.warning(f"Deleted {info['file']}")

        logger.warning(f"Rank {rank} has no concepts to process. Exiting.")
        return

    # use_hf_only already computed above
    if not use_hf_only:
        # Create a new OpenAI client (required for dataset_factory.create_eval_df in AxBench mode)
        lm_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=60.0,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=1000
                ),
                headers={"Connection": "close"},
            ),
            max_retries=3,
        )
    else:
        lm_client = None

    # Initialize the dataset factory with the tokenizer.
    if "google/gemma-3" in args.steering_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.steering_model_name, use_fast=False, model_max_length=128000)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.steering_model_name, use_fast=False, model_max_length=1024)
    tokenizer.padding_side = "right"
    if "PromptSteering" in args.models:
        has_prompt_steering = True
    else:
        if "LsReFT" in args.models and training_args.models["LsReFT"].use_synergy:
            has_prompt_steering = True
        else:
            has_prompt_steering = False
    if not use_hf_only:
        dataset_factory = SteeringDatasetFactory(
            tokenizer, dump_dir,
            master_data_dir=args.master_data_dir, lm_client=lm_client,
            lm_model=args.lm_model,
            has_prompt_steering=has_prompt_steering
        )
    else:
        dataset_factory = None
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    if "gemma-3" in args.model_name:
        from transformers import Gemma3ForCausalLM
        model_instance = Gemma3ForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16 if args.use_bf16 else None, device_map=device)
    else:
        model_instance = AutoModelForCausalLM.from_pretrained(
            args.steering_model_name if args.steering_model_name else args.model_name, 
            torch_dtype=torch.bfloat16 if args.use_bf16 else None, device_map=device
        )
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    # Prepare data per concept
    data_per_concept = {}
    for concept_id in my_concept_ids:
        if use_hf_only:
            # HF-only steering: select rows strictly by concept_id (ground truth)
            concept_rows = hf_df[hf_df["concept_id"] == concept_id].head(
                int(num_of_examples) if num_of_examples is not None else 0
            )

            if len(concept_rows) == 0:
                raise ValueError(
                    f"[HF-only steering] No HF rows found for concept_id={concept_id} "
                    f"within the first {len(hf_df)} rows (training slice)."
                )

            # Derive a human-readable label for logging only
            try:
                target_concept = str(concept_rows.iloc[0]["output_concept"])
            except Exception:
                target_concept = f"concept_{concept_id}"

            factors = steering_factors if steering_factors is not None else [1.0]

            records = []
            input_ctr = 0
            for _, row in concept_rows.iterrows():
                for f in factors:
                    records.append({
                        "input": row["input"],
                        "output": row["output"],
                        "input_concept": target_concept,
                        "concept_id": 0,
                        "factor": float(f),
                        "input_id": int(input_ctr),
                    })
                input_ctr += 1

            current_df = pd.DataFrame.from_records(records)
            data_per_concept[concept_id] = (current_df, None, None)
        else:
            current_df, (_, sae_link, sae_id) = create_data_steering(
                dataset_factory, metadata, concept_id, num_of_examples,
                steering_factors, steering_datasets, args, generate_args
            )
            data_per_concept[concept_id] = (current_df, sae_link, sae_id)

    # In HF-only mode, we may skip concepts with no data; only process those we built.
    if len(data_per_concept) == 0:
        logger.warning("[HF-only steering] No data_per_concept entries were created; nothing to run. Exiting.")
        return

    my_concept_ids = [cid for cid in my_concept_ids if cid in data_per_concept]
    
    # Preload models that are shared across concepts, like HyperSteer.
    preloaded_models = dict()
    for model_name in training_args.models.keys():
        if model_name in STEERING_EXCLUDE_MODELS:
            continue
        if use_hf_only and model_name != "HyperSteer":
            logger.warning(
                f"[Info] HF-only steering mode: skipping model '{model_name}' "
                "because it may require metadata/SAE refs."
            )
            continue
        if model_name in STEERING_WITH_SHARED_MODELS:
            model_class = getattr(axbench, model_name)
            logger.warning(f"Loading {model_class} on {device}.")
            benchmark_model = model_class(
                model_instance,
                tokenizer,
                layer=layer,
                low_rank_dimension=(trained_hypersteer_dim or num_concepts),
                device=device,
                training_args=training_args.models[model_name],
                lm_model_name=training_args.model_name,
            )
            # --- Sanity: Add a flag to check if HyperSteer hook is called ---
            benchmark_model._sanity_hook_called = False
            benchmark_model.load(
                dump_dir=train_dir, low_rank_dimension=1, mode="steering",
                hypernet_initialize_from_pretrained=training_args.models[model_name].hypernet_initialize_from_pretrained,
                hypernet_name_or_path=training_args.models[model_name].hypernet_name_or_path,
                num_hidden_layers=training_args.models[model_name].num_hidden_layers,
            )
            # --- Enforce HyperSteer concept-space correctness ---
            expected_dim = (trained_hypersteer_dim or num_concepts)
            assert benchmark_model.ax.low_rank_dimension >= expected_dim, (
                f"HyperSteer low_rank_dimension={benchmark_model.ax.low_rank_dimension} "
                f"is smaller than expected_dim={expected_dim}"
            )
            # --- After load, wrap the ax.forward to fire the sanity flag ---
            if hasattr(benchmark_model, "ax"):
                orig_forward = benchmark_model.ax.forward
                def _wrapped_forward(*args, **kwargs):
                    benchmark_model._sanity_hook_called = True
                    return orig_forward(*args, **kwargs)
                benchmark_model.ax.forward = _wrapped_forward
            preloaded_models[model_name] = benchmark_model

    # Now loop over concept_ids and use preloaded models
    for concept_id in my_concept_ids:
        # --- Sanity check: warn if steering with HF concept ---
        if metadata is None:
            concept_name = getattr(args, "concept_prompt", None)
            logger.warning(
                f"[SanityCheck] Steering with concept_id={concept_id} "
                f"({concept_name}) from HF dataset."
            )
        current_df, sae_link, sae_id = data_per_concept[concept_id]
        for model_name in args.models:
            if model_name in STEERING_EXCLUDE_MODELS:
                continue

            if use_hf_only and model_name != "HyperSteer":
                logger.warning(
                    f"[Info] HF-only steering mode: skipping model '{model_name}' "
                    "because it may require metadata/SAE refs."
                )
                continue

            if model_name not in STEERING_WITH_SHARED_MODELS:
                model_class = getattr(axbench, model_name)
                logger.warning(f"Loading {model_class} on {device}.")

                benchmark_model = model_class(
                    model_instance, tokenizer, layer=layer,
                    training_args=training_args.models[model_name] if model_name not in {"PromptSteering", "GemmaScopeSAE"} else None, # we init with training args as well
                    low_rank_dimension=num_concepts,
                    device=device, steering_layers=steering_layers,
                )
                if model_name in {"PromptSteering", "GemmaScopeSAE"}:
                    lr = 1
                else:
                    lr = training_args.models[model_name].low_rank_dimension if training_args.models[model_name].low_rank_dimension else 1
                benchmark_model.load(
                    dump_dir=train_dir, sae_path=metadata[0]["ref"] if metadata is not None else None,
                    mode="steering",
                    priority_mode="compute_priority",
                    intervention_type=args.steering_intervention_type,
                    concept_id=concept_id,
                    low_rank_dimension=lr
                )
                benchmark_model.to(device)
                if hasattr(benchmark_model, 'ax') and args.use_bf16:
                    if model_name not in {"PreferenceLoReFT", "ConceptLoReFT",}:
                        if isinstance(benchmark_model.ax, list):
                            for ax in benchmark_model.ax:
                                ax.eval()
                            ax.to(torch.bfloat16)
                        else:
                            benchmark_model.ax.eval()
                            benchmark_model.ax.to(torch.bfloat16)
            else:
                benchmark_model = preloaded_models[model_name]

                benchmark_model.to(device)
                if hasattr(benchmark_model, 'ax') and args.use_bf16:
                    benchmark_model.ax.eval()
                    benchmark_model.ax.to(torch.bfloat16)

            # Pre-compute mean activations once, only if metadata is present
            if model_name not in {"LoReFT", "BoW"} and model_name not in LATENT_EXCLUDE_MODELS:
                if metadata is not None:
                    benchmark_model.pre_compute_mean_activations(
                        os.path.join(dump_dir, "inference"),
                        master_data_dir=args.master_data_dir,
                        disable_neuronpedia_max_act=args.disable_neuronpedia_max_act,
                        metadata=metadata,
                    )
            unique_concept_ids = list(set(current_df["concept_id"].tolist()))
            logger.warning(f"Inference steering with {model_name} on {device} for concept {concept_id}.")
            # Run prediction
            results = benchmark_model.predict_steer(
                current_df, concept_id=unique_concept_ids[0] if len(unique_concept_ids) == 1 else unique_concept_ids, sae_link=None, sae_id=None,
                batch_size=int(args.steering_batch_size),
                eval_output_length=int(args.steering_output_length),
                temperature=float(args.temperature),
                prefix_length=prefix_length,
                positions=training_args.models[model_name].intervention_positions if model_name not in {"PromptSteering", "GemmaScopeSAE"} else None,
                use_synergy=False,
                disable_neuronpedia_max_act=args.disable_neuronpedia_max_act,
                intervene_on_prompt=args.intervene_on_prompt if args.intervene_on_prompt is not None else True,
                return_vector=False,
            )
            # --- After prediction, check if HyperSteer hook fired and log ---
            if model_name == "HyperSteer":
                assert benchmark_model._sanity_hook_called, (
                    "Sanity check failed: HyperSteer was loaded but its intervention "
                    "hook never fired. Steering is not being applied."
                )
                logger.warning(
                    "[SanityCheck] HyperSteer hook fired successfully — steering is active."
                )
            # Store the results in current_df
            for k, v in results.items():
                current_df[f"{model_name}_{k}"] = v

            if model_name not in STEERING_WITH_SHARED_MODELS:
                del benchmark_model
            else:
                benchmark_model = benchmark_model.to("cpu") # move shared model to cpu to save memory

            torch.cuda.empty_cache()
        save(overwrite_inference_dump_dir, 'steering', current_df, rank)
        logger.warning(f"Saved inference results for concept {concept_id} to rank_{rank}_steering_data.parquet")
        # After processing, save state
        current_state = {'last_concept_id': concept_id}
        save_state(overwrite_inference_dump_dir, current_state, 'steering', rank)

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list((Path(dump_dir) / "inference").glob("rank_*_steering_data.parquet"))
        # Parse filenames to extract rank
        import re
        pattern = re.compile(r'rank_(\d+)_steering_data\.parquet')

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({
                    'rank': rank_int,
                    'file': parquet_file
                })
            else:
                logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x['rank'])

        # Read and concatenate dataframes
        dfs = []
        for info in file_info_list:
            df = pd.read_parquet(info['file'])
            dfs.append(df)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Optionally sort combined_df by 'concept_id' if needed
            combined_df = combined_df.sort_values(by=['concept_id', 'input_id', 'factor']).reset_index(drop=True)
            combined_df.to_parquet(Path(dump_dir) / "inference" / "steering_data.parquet", engine='pyarrow')
            logger.warning(f"Saved combined steering inference results to {Path(dump_dir) / 'inference' / 'steering_data.parquet'}")
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info['file'])
            logger.warning(f"Deleted {info['file']}")


def infer_latent(args, rank, world_size, device, logger, training_args, generate_args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = Path(args.dump_dir) / "inference"
    num_of_examples = args.latent_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"] if config else 0  # default layer for prompt baselines

    state = load_state(args.dump_dir, "latent", rank)
    last_concept_id_processed = state.get("last_concept_id", None) if state else None
    logger.warning(f"Rank {rank} last concept_id processed: {last_concept_id_processed}")

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx+1:]
        else:
            # If last_concept_id_processed is not in my_concept_ids, process all
            pass

    if len(my_concept_ids) == 0:
        logger.warning(f"Rank {rank} has no concepts to process. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=1024)
    tokenizer.padding_side = "right"

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16 if args.use_bf16 else None, 
        device_map=device
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, generate_args.dataset_category, None, None, args.dump_dir,
        use_cache=False, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model, logger=logger, is_inference=True,
        overwrite_inference_data_dir=training_args.overwrite_inference_data_dir
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    has_latent_model = False
    for model_name in args.models:
        # load model on the fly to save memory
        if model_name not in LATENT_EXCLUDE_MODELS:
            has_latent_model = True
            break

    if not has_latent_model:
        logger.warning("No latent model to infer. Exiting.")
        return

    # Now loop over concept_ids and use preloaded models
    cache_df = {}
    for concept_id in my_concept_ids:
        for model_name in args.models:
            # load model on the fly to save memory
            if model_name in LATENT_EXCLUDE_MODELS:
                continue
            model_class = getattr(axbench, model_name)
            logger.warning(f"Loading {model_class} on {device}.")
            benchmark_model = model_class(
                model_instance, tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device
            )
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent",
                concept_id=concept_id
            )
            benchmark_model.to(device)
            if hasattr(benchmark_model, 'ax') and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)

            dataset_category = generate_args.dataset_category
            if (concept_id, dataset_category) not in cache_df:
                current_df = create_data_latent(
                    dataset_factory, metadata, concept_id, num_of_examples, args)
                logger.warning(f"Inference latent with {model_name} on {device} for concept {concept_id}.")
                current_df = prepare_df(current_df, tokenizer, is_chat_model, args.model_name)
                cache_df[(concept_id, dataset_category)] = current_df
            else:
                current_df = cache_df[(concept_id, dataset_category)]

            results = benchmark_model.predict_latent(
                current_df, batch_size=args.latent_batch_size, prefix_length=prefix_length
            )
            # Store the results in current_df
            for k, v in results.items():
                if k == "tokens":
                    if "tokens" not in current_df:
                        current_df["tokens"] = v  # for tokens, they are global
                    else:
                        continue
                else:
                    current_df[f"{model_name}_{k}"] = v
            del benchmark_model
            torch.cuda.empty_cache()
        save(dump_dir, 'latent', current_df, rank)
        logger.warning(f"Saved inference results for concept {concept_id} to rank_{rank}_latent_data.parquet")
        # After processing, save state
        current_state = {'last_concept_id': concept_id}
        save_state(args.dump_dir, current_state, 'latent', rank)

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list(dump_dir.glob("rank_*_latent_data.parquet"))
        # Parse filenames to extract rank
        import re
        pattern = re.compile(r'rank_(\d+)_latent_data\.parquet')

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({
                    'rank': rank_int,
                    'file': parquet_file
                })
            else:
                logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x['rank'])

        # Read and concatenate dataframes
        dfs = []
        for info in file_info_list:
            df = pd.read_parquet(info['file'])
            dfs.append(df)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_parquet(dump_dir / "latent_data.parquet", engine='pyarrow')
            logger.warning(f"Saved combined latent inference results to {dump_dir / 'latent_data.parquet'}")
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info['file'])
            logger.warning(f"Deleted {info['file']}")

        # Save top logits (optional)
        logger.warning("Saving top logits...")
        if "LsReFT" in args.models:
            model_name = "LsReFT"
            model_class = getattr(axbench, model_name)
            benchmark_model = model_class(
                model_instance, tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device
            )
            benchmark_model.load(dump_dir=train_dir, sae_path=metadata[0]["ref"])
            if hasattr(benchmark_model, 'ax') and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)
            benchmark_model.to(device)
            for concept_id in concept_ids:
                top_logits, neg_logits = benchmark_model.get_logits(concept_id, k=10)
                top_logits_entry = {
                    "concept_id": int(concept_id),
                    "results": {
                        model_name: {
                            "top_logits": top_logits,
                            "neg_logits": neg_logits
                        }
                    }
                }
                with open(dump_dir / "top_logits.jsonl", "a") as f:
                    f.write(json.dumps(top_logits_entry) + "\n")


def infer_latent_imbalance(args, rank, world_size, device, logger, training_args, generate_args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"] if config else 0  # default layer for prompt baselines

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=1024)
    tokenizer.padding_side = "right"

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16 if args.use_bf16 else None, 
        device_map=device
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, generate_args.dataset_category, None, None, dump_dir,
        use_cache=False, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model, logger=logger, is_inference=True,
        overwrite_inference_data_dir=training_args.overwrite_inference_data_dir
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    has_latent_model = False
    for model_name in args.models:
        # load model on the fly to save memory
        if model_name not in LATENT_EXCLUDE_MODELS:
            has_latent_model = True
            break

    if not has_latent_model:
        logger.warning("No latent model to infer. Exiting.")
        return

    logger.warning(f"We are inferencing imbalanced latent once for all concepts with factor {args.imbalance_factor}.")
    all_negative_df = dataset_factory.create_imbalance_eval_df(
        num_of_examples, factor=args.imbalance_factor)
    all_negative_df = prepare_df(all_negative_df, tokenizer, is_chat_model, args.model_name)

    # save all_negative_df to disk
    dump_dir = Path(dump_dir) / "inference_imbalance"
    dump_dir.mkdir(parents=True, exist_ok=True)
    all_negative_df.to_parquet(Path(dump_dir) / "all_negative_df.parquet", engine='pyarrow')

    for model_name in args.models:
        # load model on the fly to save memory
        if model_name in LATENT_EXCLUDE_MODELS:
            continue
        model_class = getattr(axbench, model_name)
        logger.warning(f"Loading {model_class} on {device}.")
        benchmark_model = model_class(
            model_instance, tokenizer, layer=layer,
            low_rank_dimension=len(metadata),
            device=device
        )
        if model_name in {"PromptDetection", "BoW"}:
            for concept_id in concept_ids:
                benchmark_model.load(
                    dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent",
                    concept_id=concept_id
                )
                benchmark_model.to(device)
                if hasattr(benchmark_model, 'ax') and args.use_bf16:
                    benchmark_model.ax.eval()
                    benchmark_model.ax.to(torch.bfloat16)
                results = benchmark_model.predict_latent(
                    all_negative_df, 
                    batch_size=args.latent_batch_size, 
                    prefix_length=prefix_length,
                    concept=metadata[concept_id]["concept"],
                )
                # save results to disk
                with open(dump_dir / f"{model_name}_concept_{concept_id}_latent_results.pkl", "wb") as f:
                    pickle.dump(results, f)
        else:
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent"
            )
            benchmark_model.to(device)
            if hasattr(benchmark_model, 'ax') and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)
            # we only save the max act for each concept to save disk space, otherwise each file will be ~3GB.
            # if you wish to save the raw acts, you can go into predict_latents and modify the output.
            results = benchmark_model.predict_latents(
                all_negative_df, 
                batch_size=args.latent_batch_size, 
                prefix_length=prefix_length
            )
            # save results to disk
            with open(dump_dir / f"{model_name}_latent_results.pkl", "wb") as f:
                pickle.dump(results, f)


def infer_latent_on_train_data(args, rank, world_size, device, logger, training_args, generate_args):
    """This is used for getting threshold for latent and steering."""
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"] if config else 0  # default layer for prompt baselines

    state = load_state(args.dump_dir, "latent_on_train_data", rank)
    last_concept_id_processed = state.get("last_concept_id", None) if state else None
    logger.warning(f"Rank {rank} last concept_id processed: {last_concept_id_processed}")

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Partition concept_ids among ranks sequentially
    assert world_size == 1, "latent_on_train_data only supports world_size = 1"
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx+1:]
        else:
            # If last_concept_id_processed is not in my_concept_ids, process all
            pass

    if len(my_concept_ids) == 0:
        logger.warning(f"Rank {rank} has no concepts to process. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    dump_dir = Path(dump_dir) / "inference_on_train_data"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=1024)
    tokenizer.padding_side = "right"

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16 if args.use_bf16 else None, 
        device_map=device
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, generate_args.dataset_category, None, None, dump_dir,
        use_cache=False, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model, logger=logger, is_inference=True,
        overwrite_inference_data_dir=training_args.overwrite_inference_data_dir
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    has_latent_model = False
    for model_name in args.models:
        # load model on the fly to save memory
        if model_name not in LATENT_EXCLUDE_MODELS:
            has_latent_model = True
            break

    if not has_latent_model:
        logger.warning("No latent model to infer. Exiting.")
        return

    # Now loop over concept_ids and use preloaded models
    cache_df = {}
    all_results = {}
    for model_name in args.models:
        all_results[model_name] = {}
    concept_count = 0
    for concept_id in my_concept_ids:
        current_df = create_data_latent(
            dataset_factory, metadata, concept_id, num_of_examples, args)
        current_df = prepare_df(current_df, tokenizer, is_chat_model, args.model_name)
        if len(current_df) == 0:
            # for cases where the concept_id is not in the dataset, we skip it.
            # we dont increment concept_count in this case.
            continue
        for model_name in args.models:
            logger.warning(f"Inference latent with {model_name} on {device} for concept {concept_id}.")
            # load model on the fly to save memory
            if model_name in LATENT_EXCLUDE_MODELS:
                continue
            model_class = getattr(axbench, model_name)
            logger.warning(f"Loading {model_class} on {device}.")
            benchmark_model = model_class(
                model_instance, tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device
            )
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent"
            )
            benchmark_model.to(device)
            if hasattr(benchmark_model, 'ax') and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)

            results = benchmark_model.predict_latent(
                current_df, batch_size=args.latent_batch_size, prefix_length=prefix_length, 
                return_max_act_only=True, overwrite_concept_id=concept_count
            )
            all_results[model_name][concept_id] = results
            del benchmark_model
            torch.cuda.empty_cache()
        concept_count += 1
        if concept_count % 500 == 0 or concept_id == my_concept_ids[-1]:
            rotation_index = (concept_count-1) // 500
            # save results to disk
            with open(dump_dir / f"rank_{rank}_all_results_{rotation_index}.pkl", "wb") as f:
                pickle.dump(all_results, f)
            # clear all_results
            all_results = {}
            for model_name in args.models:
                all_results[model_name] = {}

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results
    if rank == 0:
        logger.warning("All ranks have finished inference.")


def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "all",
                'help': 'The inference mode.'
            }
        }
    ]
    # Add benchmark CLI args
    benchmark_args = [
        {
            'args': ['--benchmark'],
            'kwargs': {
                'type': str,
                'default': "gsm8k",
                'help': 'Benchmark name (gsm8k, supergpqa, codemmlu)'
            }
        },
        {
            'args': ['--max_questions'],
            'kwargs': {
                'type': int,
                'default': None,
                'help': 'Max number of questions to evaluate (for benchmarking).'
            }
        },
        {
            'args': ['--supergpqa_discipline'],
            'kwargs': {
                'type': str,
                'default': None,
                'help': 'Discipline filter for SuperGPQA benchmark.'
            }
        },
        {
            'args': ['--supergpqa_field'],
            'kwargs': {
                'type': str,
                'default': None,
                'help': 'Field filter for SuperGPQA benchmark.'
            }
        },
    ]
    training_args = TrainingArgs(custom_args=custom_args, section="train", ignore_unknown=True)
    generate_args = DatasetArgs(custom_args=custom_args, section="generate", ignore_unknown=True)
    inference_args = DatasetArgs(custom_args=(custom_args + benchmark_args), section="inference", ignore_unknown=True)

    if training_args.overwrite_metadata_dir is not None and os.path.exists(training_args.overwrite_metadata_dir):
        inference_args.data_dir = training_args.overwrite_metadata_dir # since we only load metadata from this dir
    else:
        inference_args.data_dir = f"{inference_args.dump_dir}/generate"
    inference_args.train_dir = f"{inference_args.dump_dir}/train"
    logger.warning("Inferencing with following configuration:")
    logger.warning(inference_args)
    set_seed(inference_args.seed)

    # ---------------- Distributed setup ----------------
    if inference_args.mode == "benchmark":
        # Benchmark mode: single-process, no torch.distributed
        rank = 0
        world_size = 1
        local_rank = 0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        logger.warning("[Benchmark] Running in single-process (non-distributed) mode.")
    else:
        # AxBench inference modes: require torch.distributed
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=60000),
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt=f'%(asctime)s,%(msecs)03d %(levelname)-8s [Rank {rank}] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )

    # Create a console handler and set its formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    # Optionally, create a file handler per rank
    """
    log_file = f'log_rank_{rank}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    """

    # Add suppress_eval_dir to inference_args if present in command line
    if hasattr(inference_args, 'suppress_eval_dir'):
        suppress_eval_dir = inference_args.suppress_eval_dir
    else:
        suppress_eval_dir = None

    try:
        if inference_args.mode == "latent":
            infer_latent(inference_args, rank, world_size, device, logger, training_args, generate_args)
        elif inference_args.mode == "latent_imbalance":
            infer_latent_imbalance(inference_args, rank, world_size, device, logger, training_args, generate_args)
        elif inference_args.mode == "latent_on_train_data":
            infer_latent_on_train_data(inference_args, rank, world_size, device, logger, training_args, generate_args)
        elif inference_args.mode == "steering":
            infer_steering(inference_args, rank, world_size, device, logger, training_args, generate_args, suppress_eval_dir=suppress_eval_dir)
        elif inference_args.mode == "all":
            infer_latent(inference_args, rank, world_size, device, logger, training_args, generate_args)
            infer_steering(inference_args, rank, world_size, device, logger, training_args, generate_args, suppress_eval_dir=suppress_eval_dir)
        elif inference_args.mode == "benchmark":
            infer_benchmark(
                inference_args, rank, world_size, device, logger, training_args
            )
        elif inference_args.mode == "benchmark_steered":
            infer_benchmark_steered(
                inference_args, rank, world_size, device, logger, training_args
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    # Remove handlers to prevent duplication if the script is run multiple times
    logger.removeHandler(console_handler)
    # If file_handler is used, remove it as well
    # logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()

