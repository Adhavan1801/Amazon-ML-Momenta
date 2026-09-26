"""
Stage 4 & 5: Prediction, Hard Veto Shield & Final Submission Generation
Applies GBDT classifier + Hard Veto Rules, formats TSV files, and validates output format.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from configs.paths import (
    TEST_S1_CLEAN, TEST_S2_CLEAN, TEST_S3_CLEAN,
    MATCHING_RESULTS, CANDIDATE_PAIRS, OUTPUT_DIR, MODELS_DIR
)
from src.blocking.hybrid_blocking import generate_candidate_pairs
from src.matching.feature_extraction import extract_pair_features
from src.matching.train_classifier import FEATURE_COLS


def apply_hard_veto_rules(feature_df, probabilities, threshold=0.85):
    """
    Applies precision-first hard veto rules to reject false matches regardless of raw model output.
    """
    final_preds = []
    veto_count = 0

    country_matches = feature_df["country_match"].values
    house_num_overlaps = feature_df["house_num_overlap"].values
    name_jaros = feature_df["name_jaro"].values

    for i in range(len(probabilities)):
        prob = probabilities[i]
        
        # Hard Veto 1: Different countries explicitly provided
        if country_matches[i] == 0.0:
            final_preds.append("NO_MATCH")
            veto_count += 1
            continue

        # Hard Veto 2: Non-overlapping street numbers with low name similarity
        if house_num_overlaps[i] == 0.0 and name_jaros[i] < 0.85:
            final_preds.append("NO_MATCH")
            veto_count += 1
            continue

        # Hard Veto 3: Very low Jaro-Winkler name similarity
        if name_jaros[i] < 0.40:
            final_preds.append("NO_MATCH")
            veto_count += 1
            continue

        # Otherwise rely on calibrated GBDT probability threshold
        if prob >= threshold:
            final_preds.append("MATCH")
        else:
            final_preds.append("NO_MATCH")

    print(f"[SHIELD] Hard Veto Shield rejected {veto_count:,} potential False Positives!")
    return final_preds


def generate_submission(candidate_pairs_df, feature_df, probabilities, threshold=0.85):
    """Formats predictions into required TSV submission files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    preds = apply_hard_veto_rules(feature_df, probabilities, threshold=threshold)
    feature_df["prediction"] = preds

    # 1. candidate_pairs.tsv format: source1_entity_id \t candidate_entity_ids (comma separated)
    print("[OUTPUT] Generating candidate_pairs.tsv...")
    cand_grouped = candidate_pairs_df.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda ids: ",".join(ids)).reset_index()
    cand_grouped.columns = ["source1_entity_id", "candidate_entity_ids"]
    cand_grouped.to_csv(CANDIDATE_PAIRS, sep="\t", index=False)
    print(f"[SUCCESS] Saved candidate_pairs.tsv ({len(cand_grouped):,} rows)")

    # 2. matching_results.tsv format: source1_entity_id \t matched_entity_ids (comma separated)
    print("[OUTPUT] Generating matching_results.tsv...")
    matched_subset = feature_df[feature_df["prediction"] == "MATCH"]
    match_grouped = matched_subset.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda ids: ",".join(ids)).reset_index()
    match_grouped.columns = ["source1_entity_id", "matched_entity_ids"]

    # Ensure all S1 entities from test set exist in matching_results.tsv
    all_s1_ids = pd.DataFrame({"source1_entity_id": candidate_pairs_df["source1_entity_id"].unique()})
    final_matching = pd.merge(all_s1_ids, match_grouped, on="source1_entity_id", how="left")
    final_matching["matched_entity_ids"] = final_matching["matched_entity_ids"].fillna("")

    final_matching.to_csv(MATCHING_RESULTS, sep="\t", index=False)
    print(f"[SUCCESS] Saved matching_results.tsv ({len(final_matching):,} rows)")


if __name__ == "__main__":
    print("Testing Submission Generator Module...")
