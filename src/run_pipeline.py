"""
Master Pipeline Runner — Pipeline 1: Multi-Stage Funnel
Orchestrates Hybrid Blocking -> Feature Extraction -> GBDT Training -> Hard Vetoes -> Submission Validation.

Usage:
    python src/run_pipeline.py --mode full --sample_size 20000
    python src/run_pipeline.py --mode full (runs full dataset)
"""

import argparse
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import time
import subprocess
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.paths import (
    TRAIN_S1_CLEAN, TRAIN_S2_CLEAN, TRAIN_S3_CLEAN, TRAIN_GROUND_TRUTH,
    TEST_S1_CLEAN, TEST_S2_CLEAN, TEST_S3_CLEAN,
    MATCHING_RESULTS, CANDIDATE_PAIRS, MODELS_DIR
)
from src.blocking.hybrid_blocking import generate_candidate_pairs
from src.matching.feature_extraction import extract_pair_features
from src.matching.train_classifier import train_gbdt_model
from src.matching.predict_submission import generate_submission


def run_full_pipeline(sample_size=None, top_k=10):
    total_start = time.time()
    print("=" * 70)
    print("[INFO] PIPELINE 1: MULTI-STAGE HYBRID FUNNEL ENTITY RESOLUTION")
    print("=" * 70)

    # ----------------------------------------------------
    # STAGE 1: Train Model on Ground Truth
    # ----------------------------------------------------
    print("\n[STEP 1/5] Loading Training Datasets & Ground Truth...")
    train_s1 = pd.read_parquet(TRAIN_S1_CLEAN)
    train_s2 = pd.read_parquet(TRAIN_S2_CLEAN)
    train_s3 = pd.read_parquet(TRAIN_S3_CLEAN)
    
    if "source1_entity_id" not in train_s1.columns:
        train_s1["source1_entity_id"] = train_s1.iloc[:, 0]
    if "target_entity_id" not in train_s2.columns:
        train_s2["target_entity_id"] = train_s2.iloc[:, 0]
    if "target_entity_id" not in train_s3.columns:
        train_s3["target_entity_id"] = train_s3.iloc[:, 0]

    train_s23 = pd.concat([train_s2, train_s3], ignore_index=True)

    # Load Ground Truth Map
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t")
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matches = set(str(row["matched_entity_ids"]).split(",")) if pd.notna(row["matched_entity_ids"]) else set()
        gt_map[s1] = matches

    if sample_size and sample_size < len(train_s1):
        print(f"  [WARN] Subsampling Train S1 to {sample_size:,} records for fast execution.")
        train_s1 = train_s1.head(sample_size)
        
        # Subsample target records: keep all ground truth positives + sample negatives
        relevant_gt_targets = set()
        for s1_id in train_s1["source1_entity_id"]:
            relevant_gt_targets.update(gt_map.get(str(s1_id).strip(), set()))

        gt_target_df = train_s23[train_s23["target_entity_id"].isin(relevant_gt_targets)]
        other_target_df = train_s23[~train_s23["target_entity_id"].isin(relevant_gt_targets)].sample(n=min(50000, len(train_s23) - len(gt_target_df)), random_state=42)
        train_s23 = pd.concat([gt_target_df, other_target_df], ignore_index=True)
        print(f"  [WARN] Subsampled Target S23 to {len(train_s23):,} relevant records for fast test execution.")

    # Generate Candidate Pairs for Train
    print("\n[STEP 2/5] Generating Candidate Pairs for Training...")
    train_cands = generate_candidate_pairs(train_s1, train_s23, top_k=top_k, cache_key=f"train_sub_{sample_size}" if sample_size else "train_full")

    # Assign Ground Truth Labels
    labels = []
    for _, row in train_cands.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        cand = str(row["candidate_entity_id"]).strip()
        labels.append(1 if cand in gt_map.get(s1, set()) else 0)

    print(f"  Total Candidate Pairs: {len(train_cands):,} | Positive Matches: {sum(labels):,}")

    # Extract Features for Train Pairs
    print("\n[STEP 3/5] Extracting Feature Vectors...")
    train_features = extract_pair_features(train_cands, train_s1, train_s23)

    # Train Model
    print("\n[STEP 4/5] Training LightGBM Classifier...")
    model_path = os.path.join(MODELS_DIR, "lgbm_entity_resolution.pkl")
    clf, best_threshold = train_gbdt_model(train_features, labels, model_save_path=model_path)

    # ----------------------------------------------------
    # STAGE 2: Test Dataset Prediction & Submission
    # ----------------------------------------------------
    print("\n[STEP 5/5] Processing Test Set & Generating Final Output...")
    test_s1 = pd.read_parquet(TEST_S1_CLEAN)
    test_s2 = pd.read_parquet(TEST_S2_CLEAN)
    test_s3 = pd.read_parquet(TEST_S3_CLEAN)

    if "source1_entity_id" not in test_s1.columns:
        test_s1["source1_entity_id"] = test_s1.iloc[:, 0]
    if "target_entity_id" not in test_s2.columns:
        test_s2["target_entity_id"] = test_s2.iloc[:, 0]
    if "target_entity_id" not in test_s3.columns:
        test_s3["target_entity_id"] = test_s3.iloc[:, 0]

    test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)

    if sample_size and sample_size < len(test_s1):
        test_s1 = test_s1.head(sample_size)
        test_s23 = test_s23.sample(n=min(100000, len(test_s23)), random_state=42)
        print(f"  [WARN] Subsampled Test set to {len(test_s1):,} query & {len(test_s23):,} target records.")

    test_cands = generate_candidate_pairs(test_s1, test_s23, top_k=top_k, cache_key=f"test_sub_{sample_size}" if sample_size else "test_full")
    test_features = extract_pair_features(test_cands, test_s1, test_s23)

    from src.matching.train_classifier import FEATURE_COLS
    test_probs = clf.predict_proba(test_features[FEATURE_COLS])[:, 1]

    generate_submission(test_cands, test_features, test_probs, threshold=best_threshold)

    # Validate Submission
    print("\n[VALIDATOR] Running Official Submission Validator...")
    val_cmd = [sys.executable, "src/validate_submission.py"]
    subprocess.run(val_cmd)

    total_time = time.time() - total_start
    print(f"\n[SUCCESS] ALL DONE! Pipeline completed in {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Entity Resolution Pipeline 1")
    parser.add_argument("--sample_size", type=int, default=None, help="Limit train samples for fast testing")
    parser.add_argument("--top_k", type=int, default=10, help="Number of candidates per entity")
    args = parser.parse_args()

    run_full_pipeline(sample_size=args.sample_size, top_k=args.top_k)
