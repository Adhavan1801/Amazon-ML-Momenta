"""
Stage 2: Pairwise Feature Extraction
Computes fine-grained string distances, token overlap, and attribute match features for candidate pairs.
"""

import re
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rapidfuzz import distance, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher


def extract_house_numbers(text):
    """Extract digits/numbers from an address line."""
    if not text or pd.isna(text):
        return set()
    return set(re.findall(r'\b\d+\b', str(text)))


def get_string_similarity(str1, str2):
    """Calculate multiple string similarity metrics between two strings."""
    s1 = str(str1 or "").strip().lower()
    s2 = str(str2 or "").strip().lower()

    if not s1 or not s2:
        return 0.0, 0.0, 0.0, 0.0

    if HAS_RAPIDFUZZ:
        jaro = distance.JaroWinkler.similarity(s1, s2)
        ratio = fuzz.ratio(s1, s2) / 100.0
        token_sort = fuzz.token_sort_ratio(s1, s2) / 100.0
        token_set = fuzz.token_set_ratio(s1, s2) / 100.0
    else:
        matcher = SequenceMatcher(None, s1, s2)
        ratio = matcher.ratio()
        jaro = ratio  # Fallback approximation
        s1_words, s2_words = set(s1.split()), set(s2.split())
        token_sort = len(s1_words & s2_words) / max(len(s1_words | s2_words), 1)
        token_set = token_sort

    return float(jaro), float(ratio), float(token_sort), float(token_set)


def extract_pair_features(candidate_pairs_df, s1_df, s23_df):
    """
    Computes pairwise feature vectors for all pairs in candidate_pairs_df.
    """
    print(f"[INFO] Extracting features for {len(candidate_pairs_df):,} candidate pairs...")
    
    # Prepare lookup dicts for ultra-fast vector access
    s1_dict = s1_df.set_index("source1_entity_id").to_dict(orient="index")
    s23_dict = s23_df.set_index("target_entity_id").to_dict(orient="index")

    features = []

    for _, row in tqdm(candidate_pairs_df.iterrows(), total=len(candidate_pairs_df), desc="Extracting Features"):
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        dense_score = row.get("dense_score", 0.0)

        rec1 = s1_dict.get(s1_id, {})
        rec2 = s23_dict.get(cand_id, {})

        name1 = rec1.get("name_clean", rec1.get("name", ""))
        name2 = rec2.get("name_clean", rec2.get("name", ""))

        addr1 = rec1.get("addr_clean", rec1.get("address", ""))
        addr2 = rec2.get("addr_clean", rec2.get("address", ""))

        country1 = str(rec1.get("country_clean", rec1.get("country", ""))).strip().upper()
        country2 = str(rec2.get("country_clean", rec2.get("country", ""))).strip().upper()

        postal1 = str(rec1.get("postal_code_clean", rec1.get("postal_code", ""))).strip()
        postal2 = str(rec2.get("postal_code_clean", rec2.get("postal_code", ""))).strip()

        # Name similarities
        name_jaro, name_ratio, name_token_sort, name_token_set = get_string_similarity(name1, name2)

        # Address similarities
        addr_jaro, addr_ratio, addr_token_sort, addr_token_set = get_string_similarity(addr1, addr2)

        # Country match flag (1 = match, 0 = mismatch, -1 = missing)
        if country1 and country2:
            country_match = 1.0 if country1 == country2 else 0.0
        else:
            country_match = -1.0

        # Postal code match flag
        if postal1 and postal2 and len(postal1) >= 3 and len(postal2) >= 3:
            postal_match = 1.0 if postal1 == postal2 else 0.0
        else:
            postal_match = -1.0

        # House number overlap
        h1 = extract_house_numbers(addr1)
        h2 = extract_house_numbers(addr2)
        if h1 and h2:
            house_num_overlap = 1.0 if len(h1 & h2) > 0 else 0.0
        else:
            house_num_overlap = -1.0

        features.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "dense_score": float(dense_score),
            "name_jaro": name_jaro,
            "name_ratio": name_ratio,
            "name_token_sort": name_token_sort,
            "name_token_set": name_token_set,
            "addr_jaro": addr_jaro,
            "addr_ratio": addr_ratio,
            "addr_token_sort": addr_token_sort,
            "addr_token_set": addr_token_set,
            "country_match": country_match,
            "postal_match": postal_match,
            "house_num_overlap": house_num_overlap,
        })

    feat_df = pd.DataFrame(features)
    print("[SUCCESS] Feature extraction complete!")
    return feat_df


if __name__ == "__main__":
    print("Testing Feature Extraction Module...")
