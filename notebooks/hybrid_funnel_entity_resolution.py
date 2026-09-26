# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE 1: HYBRID MULTI-STAGE FUNNEL ENTITY RESOLUTION PIPELINE  ║
# ║  (Dense E5 + FAISS GPU -> Pairwise Features -> LightGBM GBDT)       ║
# ╚══════════════════════════════════════════════════════════════════════╝

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import re
import sys
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, fbeta_score

try:
    from rapidfuzz import distance, fuzz
    HAS_RAPIDFUZZ = True
    print("✓ RapidFuzz available for high-speed string metric computation.")
except ImportError:
    HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher

print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Active: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# CELL 2: Configuration & Isolated Directory Paths (No clashes with Qwen)
# ==============================================================================
S3_BUCKET = "hackathon-momenta-team"
PREFIX = "data/processed/"
OUT_PREFIX_P1 = "data/output/pipeline1/"
LOCAL_DATA_DIR = "/home/sagemaker-user/dataset/processed" if os.path.exists("/home/sagemaker-user/dataset/processed") else "dataset/processed"
LOCAL_OUT_DIR = "/tmp/pipeline1_funnel_er_output"

os.makedirs(LOCAL_OUT_DIR, exist_ok=True)

FILES = {
    "train_s1": "train_source1_clean.parquet",
    "train_s2": "train_source2_clean.parquet",
    "train_s3": "train_source3_clean.parquet",
    "train_gt": "train_ground_truth.tsv",
    "test_s1":  "test_source1_clean.parquet",
    "test_s2":  "test_source2_clean.parquet",
    "test_s3":  "test_source3_clean.parquet",
}

def load_dataset(key):
    """Load dataset from local directory if present, else download from S3."""
    filename = FILES[key]
    local_path = os.path.join(LOCAL_DATA_DIR, filename)
    if os.path.exists(local_path):
        print(f"Loading local file: {local_path}")
        return pd.read_parquet(local_path) if filename.endswith(".parquet") else pd.read_csv(local_path, sep="\t")
    
    ws_path = os.path.join("data", "processed", filename)
    if os.path.exists(ws_path):
        print(f"Loading workspace file: {ws_path}")
        return pd.read_parquet(ws_path) if filename.endswith(".parquet") else pd.read_csv(ws_path, sep="\t")

    import boto3
    s3 = boto3.client("s3")
    tmp_path = os.path.join(LOCAL_OUT_DIR, filename)
    print(f"Downloading s3://{S3_BUCKET}/{PREFIX}{filename}...")
    s3.download_file(S3_BUCKET, PREFIX + filename, tmp_path)
    return pd.read_parquet(tmp_path) if filename.endswith(".parquet") else pd.read_csv(tmp_path, sep="\t")


# ==============================================================================
# CELL 3: Hybrid Candidate Generation (Blocking Stage)
# ==============================================================================
def get_id_col(df):
    for col in ["source1_entity_id", "entity_id", "source2_entity_id", "source3_entity_id"]:
        if col in df.columns:
            return col
    return df.columns[0]

def get_text_series(df):
    if "combined_text_for_embedding" in df.columns:
        return df["combined_text_for_embedding"].fillna("").astype(str)
    name_col = next((c for c in ["name_clean", "name", "business_name"] if c in df.columns), df.columns[1])
    addr_col = next((c for c in ["addr_clean", "address", "street_address"] if c in df.columns), None)
    if addr_col:
        return df[name_col].fillna("").astype(str) + " " + df[addr_col].fillna("").astype(str)
    return df[name_col].fillna("").astype(str)


E5_MODEL_NAME = "intfloat/multilingual-e5-small"

def generate_dense_candidates(query_df, target_df, top_k=10, batch_size=1024):
    """Generate dense candidate pairs using E5 + FAISS GPU."""
    print("Loading E5 bi-encoder on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
    model = AutoModel.from_pretrained(E5_MODEL_NAME).to("cuda").half()
    model.eval()

    def encode(texts, prefix="passage: "):
        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Embeddings"):
            batch = [prefix + t for t in texts[i:i + batch_size]]
            inputs = tokenizer(batch, max_length=128, padding=True, truncation=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                pooled = (outputs.last_hidden_state * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
                normed = F.normalize(pooled, p=2, dim=1)
                all_emb.append(normed.cpu().to(torch.float32).numpy())
        return np.vstack(all_emb)

    q_texts = get_text_series(query_df).tolist()
    t_texts = get_text_series(target_df).tolist()

    q_emb = encode(q_texts, prefix="query: ")
    t_emb = encode(t_texts, prefix="passage: ")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Building FAISS IndexFlatIP (dim={q_emb.shape[1]})...")
    d = q_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(t_emb)

    D, I = index.search(q_emb, top_k)

    q_id_col = get_id_col(query_df)
    t_id_col = get_id_col(target_df)

    q_ids = query_df[q_id_col].values
    t_ids = target_df[t_id_col].values

    records = []
    for i in range(len(q_ids)):
        for k in range(top_k):
            match_idx = I[i][k]
            records.append({
                "source1_entity_id": q_ids[i],
                "candidate_entity_id": t_ids[match_idx],
                "dense_score": float(D[i][k])
            })

    return pd.DataFrame(records)


# ==============================================================================
# CELL 4: Pairwise Feature Engineering Engine
# ==============================================================================
def extract_house_numbers(text):
    if not text or pd.isna(text):
        return set()
    return set(re.findall(r'\b\d+\b', str(text)))

def compute_string_metrics(s1, s2):
    str1 = str(s1 or "").strip().lower()
    str2 = str(s2 or "").strip().lower()
    if not str1 or not str2:
        return 0.0, 0.0, 0.0, 0.0
    if HAS_RAPIDFUZZ:
        jaro = distance.JaroWinkler.similarity(str1, str2)
        ratio = fuzz.ratio(str1, str2) / 100.0
        token_sort = fuzz.token_sort_ratio(str1, str2) / 100.0
        token_set = fuzz.token_set_ratio(str1, str2) / 100.0
    else:
        m = SequenceMatcher(None, str1, str2)
        ratio = m.ratio()
        jaro = ratio
        w1, w2 = set(str1.split()), set(str2.split())
        token_sort = len(w1 & w2) / max(len(w1 | w2), 1)
        token_set = token_sort
    return float(jaro), float(ratio), float(token_sort), float(token_set)

def build_features(candidate_pairs_df, query_df, target_df):
    print(f"Building pairwise features for {len(candidate_pairs_df):,} pairs...")
    q_id_col = get_id_col(query_df)
    t_id_col = get_id_col(target_df)

    q_dict = query_df.set_index(q_id_col).to_dict(orient="index")
    t_dict = target_df.set_index(t_id_col).to_dict(orient="index")

    features = []
    for _, row in tqdm(candidate_pairs_df.iterrows(), total=len(candidate_pairs_df), desc="Extracting Features"):
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        dense_score = row.get("dense_score", 0.0)

        rec1 = q_dict.get(s1_id, {})
        rec2 = t_dict.get(cand_id, {})

        name1 = rec1.get("name_clean", rec1.get("name", ""))
        name2 = rec2.get("name_clean", rec2.get("name", ""))

        addr1 = rec1.get("addr_clean", rec1.get("address", ""))
        addr2 = rec2.get("addr_clean", rec2.get("address", ""))

        c1 = str(rec1.get("country_clean", rec1.get("country", ""))).strip().upper()
        c2 = str(rec2.get("country_clean", rec2.get("country", ""))).strip().upper()

        p1 = str(rec1.get("postal_code_clean", rec1.get("postal_code", ""))).strip()
        p2 = str(rec2.get("postal_code_clean", rec2.get("postal_code", ""))).strip()

        name_jaro, name_ratio, name_t_sort, name_t_set = compute_string_metrics(name1, name2)
        addr_jaro, addr_ratio, addr_t_sort, addr_t_set = compute_string_metrics(addr1, addr2)

        country_match = 1.0 if (c1 and c2 and c1 == c2) else (0.0 if (c1 and c2) else -1.0)
        postal_match = 1.0 if (p1 and p2 and p1 == p2 and len(p1) >= 3) else (0.0 if (p1 and p2) else -1.0)

        h1, h2 = extract_house_numbers(addr1), extract_house_numbers(addr2)
        house_num_overlap = 1.0 if (h1 and h2 and len(h1 & h2) > 0) else (0.0 if (h1 and h2) else -1.0)

        features.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "dense_score": float(dense_score),
            "name_jaro": name_jaro,
            "name_ratio": name_ratio,
            "name_token_sort": name_t_sort,
            "name_token_set": name_t_set,
            "addr_jaro": addr_jaro,
            "addr_ratio": addr_ratio,
            "addr_token_sort": addr_t_sort,
            "addr_token_set": addr_t_set,
            "country_match": country_match,
            "postal_match": postal_match,
            "house_num_overlap": house_num_overlap,
        })

    return pd.DataFrame(features)


# ==============================================================================
# CELL 5: Pipeline Execution & Submission Generation
# ==============================================================================
FEATURE_COLS = [
    "dense_score",
    "name_jaro", "name_ratio", "name_token_sort", "name_token_set",
    "addr_jaro", "addr_ratio", "addr_token_sort", "addr_token_set",
    "country_match", "postal_match", "house_num_overlap"
]

def main():
    print("=" * 70)
    print("🚀 PIPELINE 1 EXECUTION — HYBRID MULTI-STAGE FUNNEL")
    print("=" * 70)

    # 1. Load Training Set
    print("\n1. Loading Training Datasets...")
    train_s1 = load_dataset("train_s1")
    train_s2 = load_dataset("train_s2")
    train_s3 = load_dataset("train_s3")
    train_gt = load_dataset("train_gt")

    s2_id_col = get_id_col(train_s2)
    s3_id_col = get_id_col(train_s3)
    train_s2["target_entity_id"] = train_s2[s2_id_col]
    train_s3["target_entity_id"] = train_s3[s3_id_col]

    train_s23 = pd.concat([train_s2, train_s3], ignore_index=True)

    # Generate Train Candidate Pairs
    print("\n2. Generating Train Candidate Pairs with FAISS...")
    train_cands = generate_dense_candidates(train_s1, train_s23, top_k=5)

    # Map Ground Truth Labels
    gt_map = {}
    for _, row in train_gt.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matches = set(str(row["matched_entity_ids"]).split(",")) if pd.notna(row["matched_entity_ids"]) else set()
        gt_map[s1] = matches

    labels = [1 if str(row["candidate_entity_id"]).strip() in gt_map.get(str(row["source1_entity_id"]).strip(), set()) else 0 for _, row in train_cands.iterrows()]

    print(f"Total Train Pairs: {len(train_cands):,} | Positive Matches: {sum(labels):,}")

    # Build Features & Train GBDT Model
    train_feats = build_features(train_cands, train_s1, train_s23)
    
    print("\n3. Training LightGBM Classifier...")
    X = train_feats[FEATURE_COLS]
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    best_thresh, best_f05 = 0.5, 0.0
    for thresh in np.arange(0.3, 0.98, 0.02):
        score = fbeta_score(y_val, (val_probs >= thresh).astype(int), beta=0.5, zero_division=0)
        if score > best_f05:
            best_f05, best_thresh = score, thresh

    print(f"✓ Trained LightGBM! Best F0.5 Score: {best_f05:.4f} at Threshold: {best_thresh:.2f}")

    # 2. Load & Process Test Set
    print("\n4. Loading & Processing Test Datasets...")
    test_s1 = load_dataset("test_s1")
    test_s2 = load_dataset("test_s2")
    test_s3 = load_dataset("test_s3")

    test_s2["target_entity_id"] = test_s2[get_id_col(test_s2)]
    test_s3["target_entity_id"] = test_s3[get_id_col(test_s3)]
    test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)

    test_cands = generate_dense_candidates(test_s1, test_s23, top_k=5)
    test_feats = build_features(test_cands, test_s1, test_s23)

    test_probs = clf.predict_proba(test_feats[FEATURE_COLS])[:, 1]

    # Hard Veto Shield
    print("\n5. Applying Precision-First Hard Veto Rules...")
    preds = []
    for i in range(len(test_probs)):
        prob = test_probs[i]
        c_match = test_feats.iloc[i]["country_match"]
        h_overlap = test_feats.iloc[i]["house_num_overlap"]
        name_jaro = test_feats.iloc[i]["name_jaro"]

        if c_match == 0.0 or (h_overlap == 0.0 and name_jaro < 0.85) or name_jaro < 0.40:
            preds.append("NO_MATCH")
        else:
            preds.append("MATCH" if prob >= best_thresh else "NO_MATCH")

    test_feats["prediction"] = preds

    # Output Submission TSVs
    print("\n6. Generating Submission TSV files...")
    cand_path = os.path.join(LOCAL_OUT_DIR, "candidate_pairs_p1.tsv")
    match_path = os.path.join(LOCAL_OUT_DIR, "matching_results_p1.tsv")
    cand_root_path = os.path.join(LOCAL_OUT_DIR, "candidate_pairs.tsv")
    match_root_path = os.path.join(LOCAL_OUT_DIR, "matching_results.tsv")

    cand_grouped = test_cands.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda ids: ",".join(ids)).reset_index()
    cand_grouped.columns = ["source1_entity_id", "candidate_entity_ids"]
    cand_grouped.to_csv(cand_path, sep="\t", index=False)
    cand_grouped.to_csv(cand_root_path, sep="\t", index=False)

    matched_subset = test_feats[test_feats["prediction"] == "MATCH"]
    match_grouped = matched_subset.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda ids: ",".join(ids)).reset_index()
    match_grouped.columns = ["source1_entity_id", "matched_entity_ids"]

    all_s1 = pd.DataFrame({"source1_entity_id": test_s1[get_id_col(test_s1)].unique()})
    final_matching = pd.merge(all_s1, match_grouped, on="source1_entity_id", how="left").fillna("")
    final_matching.to_csv(match_path, sep="\t", index=False)
    final_matching.to_csv(match_root_path, sep="\t", index=False)

    print(f"✓ Saved candidate_pairs_p1.tsv and matching_results_p1.tsv to {LOCAL_OUT_DIR}")

    # Upload to isolated S3 output prefix
    import boto3
    s3 = boto3.client("s3")
    print(f"Uploading files to s3://{S3_BUCKET}/{OUT_PREFIX_P1}...")
    s3.upload_file(cand_path, S3_BUCKET, OUT_PREFIX_P1 + "candidate_pairs.tsv")
    s3.upload_file(match_path, S3_BUCKET, OUT_PREFIX_P1 + "matching_results.tsv")
    print("🎉 Pipeline 1 uploaded cleanly to S3 without clashing!")

if __name__ == "__main__":
    main()
