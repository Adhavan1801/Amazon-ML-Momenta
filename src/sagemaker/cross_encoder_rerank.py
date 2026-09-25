"""
Cross-encoder re-ranking for entity resolution candidate pairs.

Run this on SageMaker after blocking has produced candidate pairs.
The cross-encoder reads both records jointly and predicts a match score.

This is Stage 3 (optional boost) — only use if LightGBM performance plateaus.

Usage (on SageMaker notebook):
    python cross_encoder_rerank.py --input s3://hackathon-momenta-team/output/candidate_pairs.parquet
"""

import argparse
import gc
import os
import time

import boto3
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

S3_BUCKET = "hackathon-momenta-team"

# Cross-encoder model options (pick one):
# Option A: Pre-trained MS-MARCO cross-encoder (English-focused, fast baseline)
CROSS_ENCODER_MODEL_A = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Option B: Multilingual base model (fine-tune on your matched pairs)
CROSS_ENCODER_MODEL_B = "intfloat/multilingual-e5-base"

# Default: start with pre-trained, upgrade to fine-tuned later
DEFAULT_MODEL = CROSS_ENCODER_MODEL_A

LOCAL_TMP = "/tmp/cross_encoder"


# ══════════════════════════════════════════════════════════════════════
# CROSS-ENCODER SCORING
# ══════════════════════════════════════════════════════════════════════

def score_pairs(
    pairs_df: pd.DataFrame,
    model: CrossEncoder,
    text_col_a: str = "text_a",
    text_col_b: str = "text_b",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Score candidate pairs using a cross-encoder.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Must have columns text_col_a and text_col_b with the
        combined text from each entity in the pair.
    model : CrossEncoder
        Loaded cross-encoder model.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    np.ndarray
        Match scores, shape (n_pairs,). Higher = more likely match.
    """
    texts_a = pairs_df[text_col_a].fillna("").tolist()
    texts_b = pairs_df[text_col_b].fillna("").tolist()
    sentence_pairs = list(zip(texts_a, texts_b))

    print(f"  Scoring {len(sentence_pairs):,} pairs (batch_size={batch_size})...")
    start = time.time()

    scores = model.predict(
        sentence_pairs,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    elapsed = time.time() - start
    speed = len(sentence_pairs) / elapsed
    print(f"  ✓ Done in {elapsed:.0f}s ({speed:,.0f} pairs/sec)")

    return scores


def prepare_pair_texts(
    pairs_df: pd.DataFrame,
    source_dfs: dict,
) -> pd.DataFrame:
    """
    Join candidate pairs with source data to create text columns for scoring.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Columns: entity_id_a, entity_id_b, source_a, source_b
    source_dfs : dict
        Mapping of source key → DataFrame (with entity_id, combined_text_for_embedding)

    Returns
    -------
    pd.DataFrame
        pairs_df with added text_a, text_b columns.
    """
    # Build a unified lookup: entity_id → combined_text_for_embedding
    lookup = {}
    for source_key, df in source_dfs.items():
        for eid, text in zip(df["entity_id"], df["combined_text_for_embedding"]):
            lookup[eid] = text

    pairs_df["text_a"] = pairs_df["entity_id_a"].map(lookup).fillna("")
    pairs_df["text_b"] = pairs_df["entity_id_b"].map(lookup).fillna("")

    return pairs_df


# ══════════════════════════════════════════════════════════════════════
# FINE-TUNING (optional — if pre-trained cross-encoder isn't good enough)
# ══════════════════════════════════════════════════════════════════════

def finetune_cross_encoder(
    train_pairs: pd.DataFrame,
    model_id: str = CROSS_ENCODER_MODEL_B,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    output_dir: str = "/tmp/cross_encoder_finetuned",
):
    """
    Fine-tune a cross-encoder on labeled entity pairs.

    Parameters
    ----------
    train_pairs : pd.DataFrame
        Columns: text_a, text_b, label (1=match, 0=no-match)
    model_id : str
        Base model to fine-tune.
    epochs : int
        Number of training epochs.
    """
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder.evaluation import (
        CEBinaryClassificationEvaluator,
    )
    from torch.utils.data import DataLoader

    print(f"\n🔧 Fine-tuning cross-encoder: {model_id}")
    print(f"   Training pairs: {len(train_pairs):,}")
    print(f"   Epochs: {epochs}, LR: {learning_rate}")

    model = CrossEncoder(model_id, num_labels=1, max_length=512)

    # Prepare training examples
    train_examples = [
        InputExample(texts=[row["text_a"], row["text_b"]], label=float(row["label"]))
        for _, row in train_pairs.iterrows()
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    # Train
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        optimizer_params={"lr": learning_rate},
        output_path=output_dir,
        show_progress_bar=True,
    )

    print(f"  ✓ Fine-tuned model saved to: {output_dir}")

    # Upload to S3
    import shutil
    archive_path = shutil.make_archive(
        "/tmp/cross_encoder_model", "zip", output_dir
    )
    s3 = boto3.client("s3")
    s3_key = "models/cross_encoder_finetuned.zip"
    s3.upload_file(archive_path, S3_BUCKET, s3_key)
    print(f"  ✓ Uploaded to s3://{S3_BUCKET}/{s3_key}")

    return model


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-encoder re-ranking")
    parser.add_argument("--input", type=str, required=True, help="S3 path to candidate pairs parquet")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for positive match")
    parser.add_argument("--output", type=str, default=None, help="S3 path for output")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    # Load cross-encoder
    print(f"\n📦 Loading cross-encoder: {args.model}")
    model = CrossEncoder(args.model, max_length=512, device=device)
    print(f"  ✓ Model loaded")

    # Download candidate pairs
    s3 = boto3.client("s3")
    os.makedirs(LOCAL_TMP, exist_ok=True)
    local_input = os.path.join(LOCAL_TMP, "candidate_pairs.parquet")

    # Parse S3 path
    parts = args.input.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    s3.download_file(bucket, key, local_input)
    pairs_df = pd.read_parquet(local_input)
    print(f"  Loaded {len(pairs_df):,} candidate pairs")

    # Score
    scores = score_pairs(pairs_df, model, batch_size=args.batch_size)
    pairs_df["cross_encoder_score"] = scores

    # Filter
    matches = pairs_df[pairs_df["cross_encoder_score"] >= args.threshold]
    print(f"\n  Matches above threshold ({args.threshold}): {len(matches):,} / {len(pairs_df):,}")

    # Save
    output_key = args.output or f"output/cross_encoder_scored_pairs.parquet"
    if output_key.startswith("s3://"):
        output_key = output_key.replace(f"s3://{S3_BUCKET}/", "")

    local_output = os.path.join(LOCAL_TMP, "scored_pairs.parquet")
    pairs_df.to_parquet(local_output, index=False)
    s3.upload_file(local_output, S3_BUCKET, output_key)
    print(f"  ✓ Saved to s3://{S3_BUCKET}/{output_key}")


if __name__ == "__main__":
    main()
