"""
Qwen2.5-7B-Instruct LLM judge for borderline entity pairs.

This is the OPTIONAL final check — run ONLY on the top ~1-5% uncertain
pairs where LightGBM and cross-encoder disagree or score near the threshold.

⚠️  This is SLOW (~5-10 pairs/sec) and EXPENSIVE. Never run on the full dataset.

Instance requirements:
  - ml.g4dn.xlarge (T4 16GB) — works with 4-bit quantization (GPTQ/AWQ)
  - ml.g5.xlarge (A10G 24GB) — works with fp16 natively

Usage (on SageMaker notebook):
    python qwen_llm_judge.py --input s3://hackathon-momenta-team/output/borderline_pairs.parquet
"""

import argparse
import gc
import os
import time
from typing import List, Tuple

import boto3
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

S3_BUCKET = "hackathon-momenta-team"

# Model options
MODEL_FP16 = "Qwen/Qwen2.5-7B-Instruct"           # Needs ~14GB VRAM
MODEL_4BIT = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"  # Needs ~7GB VRAM

LOCAL_TMP = "/tmp/llm_judge"

# The prompt template for entity matching
MATCH_PROMPT_TEMPLATE = """You are an expert at business entity resolution. Determine if the two business records below refer to the SAME real-world entity.

Consider: business name similarity, address proximity, and country. Minor spelling differences, abbreviations, and format variations are common and should NOT prevent a match.

Record A:
  Name: {name_a}
  Address: {addr_a}
  Country: {country_a}

Record B:
  Name: {name_b}
  Address: {addr_b}
  Country: {country_b}

Answer with ONLY "MATCH" or "NO_MATCH" followed by a brief one-sentence reason.
"""


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════

def load_qwen_model(use_4bit: bool = True):
    """
    Load Qwen2.5-7B-Instruct with optional 4-bit quantization.

    Parameters
    ----------
    use_4bit : bool
        If True, loads the GPTQ 4-bit quantized version (fits on T4 16GB).
        If False, loads fp16 (needs A10G 24GB or better).

    Returns
    -------
    tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_4BIT if use_4bit else MODEL_FP16

    print(f"📦 Loading model: {model_id}")
    print(f"   Quantization: {'4-bit GPTQ' if use_4bit else 'fp16'}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    model.eval()

    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / (1024**3)
        mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"   GPU Memory: {mem_used:.1f} / {mem_total:.0f} GB")

    print(f"  ✓ Model loaded")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════

def judge_pair(
    model,
    tokenizer,
    name_a: str, addr_a: str, country_a: str,
    name_b: str, addr_b: str, country_b: str,
    max_new_tokens: int = 100,
) -> Tuple[str, str]:
    """
    Use Qwen to judge if two records match.

    Returns
    -------
    tuple of (decision, reason)
        decision: "MATCH" or "NO_MATCH"
        reason: brief explanation
    """
    prompt = MATCH_PROMPT_TEMPLATE.format(
        name_a=name_a, addr_a=addr_a, country_a=country_a,
        name_b=name_b, addr_b=addr_b, country_b=country_b,
    )

    messages = [
        {"role": "system", "content": "You are a precise entity resolution assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,       # Low temp for deterministic output
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip input tokens)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Parse response
    if response.upper().startswith("MATCH"):
        decision = "MATCH"
    elif response.upper().startswith("NO_MATCH"):
        decision = "NO_MATCH"
    else:
        # Try to find MATCH/NO_MATCH anywhere in the response
        if "NO_MATCH" in response.upper() or "NO MATCH" in response.upper():
            decision = "NO_MATCH"
        elif "MATCH" in response.upper():
            decision = "MATCH"
        else:
            decision = "UNCERTAIN"

    return decision, response


def judge_batch(
    model,
    tokenizer,
    pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Judge all pairs in a DataFrame.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Must have columns: name_a, addr_a, country_a, name_b, addr_b, country_b

    Returns
    -------
    pd.DataFrame
        Input df with added columns: llm_decision, llm_reason
    """
    decisions = []
    reasons = []

    print(f"  Judging {len(pairs_df):,} pairs with Qwen...")
    start = time.time()

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        decision, reason = judge_pair(
            model, tokenizer,
            row.get("name_a", ""), row.get("addr_a", ""), row.get("country_a", ""),
            row.get("name_b", ""), row.get("addr_b", ""), row.get("country_b", ""),
        )
        decisions.append(decision)
        reasons.append(reason)

    elapsed = time.time() - start
    speed = len(pairs_df) / elapsed if elapsed > 0 else 0
    print(f"  ✓ Done in {elapsed:.0f}s ({speed:.1f} pairs/sec)")

    pairs_df["llm_decision"] = decisions
    pairs_df["llm_reason"] = reasons

    # Summary
    match_count = sum(1 for d in decisions if d == "MATCH")
    no_match_count = sum(1 for d in decisions if d == "NO_MATCH")
    uncertain_count = sum(1 for d in decisions if d == "UNCERTAIN")
    print(f"  Results: {match_count} MATCH, {no_match_count} NO_MATCH, {uncertain_count} UNCERTAIN")

    return pairs_df


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Qwen LLM judge for borderline pairs")
    parser.add_argument("--input", type=str, required=True, help="S3 path to borderline pairs parquet")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of 4-bit (needs A10G)")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs to judge")
    parser.add_argument("--output", type=str, default=None, help="S3 key for output")
    args = parser.parse_args()

    # Safety check
    print("⚠️  LLM Judge mode — this is slow and costly.")
    print("   Only run on borderline pairs, NOT the full dataset.\n")

    # Load model
    use_4bit = not args.fp16
    model, tokenizer = load_qwen_model(use_4bit=use_4bit)

    # Download pairs
    s3 = boto3.client("s3")
    os.makedirs(LOCAL_TMP, exist_ok=True)
    local_input = os.path.join(LOCAL_TMP, "borderline_pairs.parquet")

    parts = args.input.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    s3.download_file(bucket, key, local_input)
    pairs_df = pd.read_parquet(local_input)
    print(f"  Loaded {len(pairs_df):,} borderline pairs")

    if args.max_pairs and len(pairs_df) > args.max_pairs:
        print(f"  ⚠️  Limiting to {args.max_pairs:,} pairs")
        pairs_df = pairs_df.head(args.max_pairs)

    # Judge
    pairs_df = judge_batch(model, tokenizer, pairs_df)

    # Save
    output_key = args.output or "output/llm_judged_pairs.parquet"
    if output_key.startswith("s3://"):
        output_key = output_key.replace(f"s3://{S3_BUCKET}/", "")

    local_output = os.path.join(LOCAL_TMP, "llm_judged.parquet")
    pairs_df.to_parquet(local_output, index=False)
    s3.upload_file(local_output, S3_BUCKET, output_key)
    print(f"  ✓ Saved to s3://{S3_BUCKET}/{output_key}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
