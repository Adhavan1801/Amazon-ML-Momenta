"""
Main preprocessing runner.
Loads raw TSV files, applies normalization, and saves as parquet.

Usage:
    python -m src.preprocessing.run_preprocessing --split train
    python -m src.preprocessing.run_preprocessing --split test
    python -m src.preprocessing.run_preprocessing --split all
    python -m src.preprocessing.run_preprocessing --split all --workers 4
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from configs.paths import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    TRAIN_S1_CLEAN, TRAIN_S2_CLEAN, TRAIN_S3_CLEAN,
    TEST_S1_CLEAN, TEST_S2_CLEAN, TEST_S3_CLEAN,
    PROCESSED_DIR,
)
from src.preprocessing.normalize import preprocess_dataframe


def process_source_file(input_path: str, output_path: str, description: str, nrows=None):
    """Load a raw TSV, preprocess it, and save as parquet."""
    print(f"\n{'='*60}")
    print(f"Processing: {description}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    start = time.time()

    # Load raw TSV
    print(f"  Loading TSV...")
    df = pd.read_csv(input_path, sep="\t", dtype=str, nrows=nrows)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Apply preprocessing
    df = preprocess_dataframe(df)

    # Save as parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    elapsed = time.time() - start
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved to {output_path} ({file_size_mb:.1f} MB)")
    print(f"  ✓ Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return description, elapsed


def _process_job(args):
    """Wrapper for ProcessPoolExecutor — unpacks the job tuple."""
    input_path, output_path, description, nrows = args
    return process_source_file(input_path, output_path, description, nrows)


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Which data split to process (default: all)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit rows per file for testing (default: None = all rows)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(num_files, cpu_count/2))",
    )
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    total_start = time.time()

    # Define all jobs
    jobs = []
    if args.split in ("train", "all"):
        jobs.extend([
            (TRAIN_SOURCE1, TRAIN_S1_CLEAN, "Train Source 1"),
            (TRAIN_SOURCE2, TRAIN_S2_CLEAN, "Train Source 2"),
            (TRAIN_SOURCE3, TRAIN_S3_CLEAN, "Train Source 3"),
        ])
    if args.split in ("test", "all"):
        jobs.extend([
            (TEST_SOURCE1, TEST_S1_CLEAN, "Test Source 1"),
            (TEST_SOURCE2, TEST_S2_CLEAN, "Test Source 2"),
            (TEST_SOURCE3, TEST_S3_CLEAN, "Test Source 3"),
        ])

    # Determine worker count — each large file uses ~3-5 GB at peak,
    # so we cap at cpu_count // 2 to avoid memory pressure.
    num_workers = args.workers or min(len(jobs), max(1, os.cpu_count() // 2))
    nrows = args.nrows

    print(f"\n🚀 Preprocessing Pipeline — {len(jobs)} files to process")
    print(f"   Workers: {num_workers} (parallel processes)")
    if nrows:
        print(f"   ⚠️  Limited to {nrows:,} rows per file (test mode)")

    # Build job args — add nrows to each tuple
    job_args = [(inp, out, desc, nrows) for inp, out, desc in jobs]

    if num_workers == 1:
        # Sequential mode
        for ja in job_args:
            _process_job(ja)
    else:
        # Parallel mode
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_job, ja): ja[2]
                for ja in job_args
            }
            for future in as_completed(futures):
                desc = futures[future]
                try:
                    name, elapsed = future.result()
                    print(f"\n  🏁 {name} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
                except Exception as e:
                    print(f"\n  ❌ {desc} FAILED: {e}")
                    raise

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"🎉 All done! Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

