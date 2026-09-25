"""
Embedding generation notebook for SageMaker.

Run this on the SageMaker Notebook Instance (ml.g4dn.xlarge with T4 GPU).
It generates dense embeddings for all source files using multilingual-e5-small,
then saves them back to S3.

This is Stage 1 of the pipeline: Blocking / Candidate Generation.

Usage (on SageMaker notebook):
    %run generate_embeddings.py
    
Or from terminal:
    python generate_embeddings.py --source all --batch-size 512
"""

import argparse
import gc
import os
import time

import boto3
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

S3_BUCKET = "hackathon-momenta-team"
S3_DATA_PREFIX = "data/processed"
S3_EMBEDDINGS_PREFIX = "data/embeddings"

MODEL_ID = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

# E5 models require a prefix for queries/passages
PASSAGE_PREFIX = "passage: "

# Files to process
SOURCE_FILES = {
    "train_s1": "train_source1_clean.parquet",
    "train_s2": "train_source2_clean.parquet",
    "train_s3": "train_source3_clean.parquet",
    "test_s1": "test_source1_clean.parquet",
    "test_s2": "test_source2_clean.parquet",
    "test_s3": "test_source3_clean.parquet",
}

LOCAL_DATA_DIR = "/tmp/data/processed"
LOCAL_EMBEDDINGS_DIR = "/tmp/data/embeddings"


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def download_from_s3(filename: str):
    """Download a parquet file from S3 to local disk."""
    s3 = boto3.client("s3")
    local_path = os.path.join(LOCAL_DATA_DIR, filename)

    if os.path.exists(local_path):
        print(f"  ✓ Already cached: {local_path}")
        return local_path

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    s3_key = f"{S3_DATA_PREFIX}/{filename}"
    print(f"  ↓ Downloading s3://{S3_BUCKET}/{s3_key} ...")
    s3.download_file(S3_BUCKET, s3_key, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  ✓ Downloaded ({size_mb:.0f} MB)")
    return local_path


def upload_to_s3(local_path: str, s3_key: str):
    """Upload a file to S3."""
    s3 = boto3.client("s3")
    print(f"  ↑ Uploading to s3://{S3_BUCKET}/{s3_key} ...")
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"  ✓ Uploaded")


# ══════════════════════════════════════════════════════════════════════
# EMBEDDING GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    text_column: str = "combined_text_for_embedding",
    batch_size: int = 512,
    prefix: str = PASSAGE_PREFIX,
) -> np.ndarray:
    """
    Generate dense embeddings for all rows in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a text column to embed.
    model : SentenceTransformer
        The loaded sentence-transformers model.
    text_column : str
        Column name containing text to embed.
    batch_size : int
        Batch size for encoding.
    prefix : str
        Prefix to prepend (E5 models need "passage: " or "query: ").

    Returns
    -------
    np.ndarray
        Array of shape (n_rows, embedding_dim), dtype float32.
    """
    texts = df[text_column].fillna("").tolist()

    # Add prefix for E5 models
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]

    print(f"  Encoding {len(texts):,} texts (batch_size={batch_size})...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    elapsed = time.time() - start
    speed = len(texts) / elapsed
    print(f"  ✓ Done in {elapsed:.0f}s ({speed:,.0f} records/sec)")

    return embeddings.astype(np.float32)


def process_source(
    source_key: str,
    filename: str,
    model: SentenceTransformer,
    batch_size: int,
):
    """Download data, generate embeddings, save locally and upload to S3."""
    print(f"\n{'='*60}")
    print(f"Processing: {source_key} ({filename})")
    print(f"{'='*60}")

    # Download from S3
    local_path = download_from_s3(filename)

    # Load parquet
    df = pd.read_parquet(local_path)
    print(f"  Loaded {len(df):,} rows")

    # Generate embeddings
    embeddings = generate_embeddings(df, model, batch_size=batch_size)

    # Save embeddings as .npy
    os.makedirs(LOCAL_EMBEDDINGS_DIR, exist_ok=True)
    emb_filename = filename.replace("_clean.parquet", "_embeddings.npy")
    emb_local_path = os.path.join(LOCAL_EMBEDDINGS_DIR, emb_filename)
    np.save(emb_local_path, embeddings)
    size_mb = os.path.getsize(emb_local_path) / (1024 * 1024)
    print(f"  Saved embeddings: {emb_local_path} ({size_mb:.0f} MB)")

    # Also save entity_id mapping (so we can map embeddings back to entities)
    id_filename = filename.replace("_clean.parquet", "_entity_ids.parquet")
    id_local_path = os.path.join(LOCAL_EMBEDDINGS_DIR, id_filename)
    df[["entity_id"]].to_parquet(id_local_path, index=False)

    # Upload to S3
    upload_to_s3(emb_local_path, f"{S3_EMBEDDINGS_PREFIX}/{emb_filename}")
    upload_to_s3(id_local_path, f"{S3_EMBEDDINGS_PREFIX}/{id_filename}")

    # Free memory
    del df, embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  ✓ {source_key} complete!")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings on SageMaker")
    parser.add_argument(
        "--source",
        choices=list(SOURCE_FILES.keys()) + ["all", "train", "test"],
        default="all",
        help="Which source to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Encoding batch size (default 512 for T4 GPU)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})",
    )
    args = parser.parse_args()

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"🖥️  GPU detected: {gpu_name} ({gpu_mem:.0f} GB)")
    else:
        print("⚠️  No GPU detected — running on CPU (will be slow)")
        args.batch_size = min(args.batch_size, 64)

    # Load model
    print(f"\n📦 Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=device)
    print(f"  ✓ Model loaded (embedding dim = {model.get_sentence_embedding_dimension()})")

    # Determine which files to process
    if args.source == "all":
        sources = SOURCE_FILES
    elif args.source == "train":
        sources = {k: v for k, v in SOURCE_FILES.items() if k.startswith("train")}
    elif args.source == "test":
        sources = {k: v for k, v in SOURCE_FILES.items() if k.startswith("test")}
    else:
        sources = {args.source: SOURCE_FILES[args.source]}

    total_start = time.time()

    for key, filename in sources.items():
        process_source(key, filename, model, args.batch_size)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"🎉 All done! Total time: {total_elapsed/60:.1f} minutes")
    print(f"   Embeddings saved to: s3://{S3_BUCKET}/{S3_EMBEDDINGS_PREFIX}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
