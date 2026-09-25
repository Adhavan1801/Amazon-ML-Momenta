# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ENTITY RESOLUTION PIPELINE — SAGEMAKER A10G NOTEBOOK              ║
# ║  Full pipeline: Embed → Block → Features → Train → Predict         ║
# ╚══════════════════════════════════════════════════════════════════════╝

# =============================================================================
# CELL 1 — Install (run once, then restart kernel)
# =============================================================================
# !pip install -q sentence-transformers faiss-cpu rapidfuzz scikit-learn "pyarrow>=14,<18"

# =============================================================================
# CELL 2 — Environment & Imports
# =============================================================================
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc, time, warnings
import numpy as np
import pandas as pd
import torch
import faiss
import lightgbm as lgb
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# =============================================================================
# CELL 3 — Configuration
# =============================================================================
DATA_DIR  = "/home/sagemaker-user/dataset/processed"
CACHE_DIR = "/home/sagemaker-user/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

BUCKET     = "hackathon-momenta-team"
OUT_PREFIX = "data/output/"

# Embedding config
EMBED_MODEL_ID  = "intfloat/multilingual-e5-small"
EMBED_DIM       = 384
EMBED_BATCH     = 512
PASSAGE_PREFIX  = "passage: "

# Blocking config
TOP_K   = 20
NLIST   = 4096
NPROBE  = 128

# Dev mode
DEV_MODE   = True
DEV_SAMPLE = 20_000

# Columns to load
FEAT_COLS  = ["entity_id", "name_clean", "name_tokens", "addr_clean",
              "street_number", "postal_code", "state_region", "country"]
EMBED_COLS = ["entity_id", "combined_text_for_embedding"]

print(f"DEV_MODE: {DEV_MODE}  (S1 sample: {DEV_SAMPLE:,})")
print(f"TOP_K:    {TOP_K} candidates per source")


# =============================================================================
# CELL 4 — Helper Functions
# =============================================================================
def load_parquet(filename, columns=None, sample_n=None):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_parquet(path, columns=columns)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"  Loaded {filename}: {len(df):>10,} rows  ({df.memory_usage(deep=True).sum()/1e6:.0f} MB)")
    return df


def encode_texts(model, texts, batch_size=EMBED_BATCH, prefix=PASSAGE_PREFIX):
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=DEVICE,
    )
    return embeddings.astype(np.float32)


def get_embeddings(model, df, filename_tag, text_col="combined_text_for_embedding"):
    cache_path = os.path.join(CACHE_DIR, f"{filename_tag}_embeddings.npy")
    ids_path   = os.path.join(CACHE_DIR, f"{filename_tag}_ids.npy")

    if os.path.exists(cache_path) and os.path.exists(ids_path):
        emb = np.load(cache_path)
        ids = np.load(ids_path, allow_pickle=True)
        if len(ids) == len(df) and np.array_equal(ids, df["entity_id"].values):
            print(f"  [cache hit] {cache_path}")
            return emb

    print(f"  Encoding {len(df):,} texts...")
    texts = df[text_col].fillna("").tolist()
    emb = encode_texts(model, texts)

    np.save(cache_path, emb)
    np.save(ids_path, df["entity_id"].values)
    print(f"  Cached: {cache_path} ({emb.nbytes/1e6:.0f} MB)")
    return emb


def build_faiss_index(embeddings):
    n, d = embeddings.shape
    nlist = min(NLIST, max(1, n // 40))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    train_n = min(n, nlist * 64)
    index.train(embeddings[:train_n])
    index.add(embeddings)
    index.nprobe = NPROBE
    print(f"  FAISS: {n:,} vectors, {nlist} clusters, nprobe={NPROBE}")
    return index


def search_candidates(query_emb, index, top_k=TOP_K):
    t0 = time.time()
    scores, indices = index.search(query_emb, top_k)
    print(f"  Search: {len(query_emb):,} queries in {time.time()-t0:.1f}s")
    return scores, indices


def build_lookup(df):
    return df.set_index("entity_id").to_dict("index")


def free(*args):
    for obj in args:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


FEATURE_NAMES = [
    "cosine_sim",
    "name_levenshtein", "name_partial", "name_token_sort", "name_jaccard",
    "addr_levenshtein", "addr_partial",
    "country_match", "postal_match", "street_match",
    "name_len_diff", "addr_len_diff",
]


def compute_features(s1_id, s2s3_id, cosine_sim, source, s1_lk, s2_lk, s3_lk):
    a = s1_lk.get(s1_id, {})
    b = (s2_lk if source == "S2" else s3_lk).get(s2s3_id, {})

    na, nb = a.get("name_clean", ""), b.get("name_clean", "")
    aa, ab = a.get("addr_clean", ""), b.get("addr_clean", "")
    ta, tb = a.get("name_tokens", ""), b.get("name_tokens", "")

    set_a = set(ta.split()) if ta else set()
    set_b = set(tb.split()) if tb else set()

    return [
        cosine_sim,
        fuzz.ratio(na, nb) / 100.0,
        fuzz.partial_ratio(na, nb) / 100.0,
        fuzz.token_sort_ratio(na, nb) / 100.0,
        len(set_a & set_b) / max(len(set_a | set_b), 1),
        fuzz.ratio(aa, ab) / 100.0,
        fuzz.partial_ratio(aa, ab) / 100.0,
        float(a.get("country", "").lower() == b.get("country", "").lower()) if a.get("country") and b.get("country") else 0.0,
        float(a.get("postal_code", "") == b.get("postal_code", "")) if a.get("postal_code") and b.get("postal_code") else 0.0,
        float(a.get("street_number", "") == b.get("street_number", "")) if a.get("street_number") and b.get("street_number") else 0.0,
        abs(len(na) - len(nb)) / max(len(na), len(nb), 1),
        abs(len(aa) - len(ab)) / max(len(aa), len(ab), 1),
    ]


def fbeta(y_true, y_pred, beta=0.5):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0: return 0.0
    return (1 + beta**2) * (p * r) / (beta**2 * p + r)


# =============================================================================
# CELL 5 — Load Embedding Model
# =============================================================================
print(f"Loading: {EMBED_MODEL_ID}")
embed_model = SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)
print(f"Loaded (dim={embed_model.get_sentence_embedding_dimension()})")


# =============================================================================
# CELL 6 — TRAIN: Embed & Block
# =============================================================================
print("\n" + "="*70)
print("PHASE 1: EMBEDDING & BLOCKING (TRAIN)")
print("="*70)

s1_n = DEV_SAMPLE if DEV_MODE else None
tag = "_dev" if DEV_MODE else ""

train_s1_df = load_parquet("train_source1_clean.parquet", columns=EMBED_COLS, sample_n=s1_n)
train_s1_emb = get_embeddings(embed_model, train_s1_df, f"train_s1{tag}")
train_s1_ids = train_s1_df["entity_id"].values
free(train_s1_df)

print("\n--- S2 ---")
train_s2_df = load_parquet("train_source2_clean.parquet", columns=EMBED_COLS)
train_s2_emb = get_embeddings(embed_model, train_s2_df, "train_s2")
train_s2_ids = train_s2_df["entity_id"].values
free(train_s2_df)

s2_ix = build_faiss_index(train_s2_emb)
s2_sc, s2_idx = search_candidates(train_s1_emb, s2_ix)
free(s2_ix, train_s2_emb)

print("\n--- S3 ---")
train_s3_df = load_parquet("train_source3_clean.parquet", columns=EMBED_COLS)
train_s3_emb = get_embeddings(embed_model, train_s3_df, "train_s3")
train_s3_ids = train_s3_df["entity_id"].values
free(train_s3_df)

s3_ix = build_faiss_index(train_s3_emb)
s3_sc, s3_idx = search_candidates(train_s1_emb, s3_ix)
free(s3_ix, train_s3_emb, train_s1_emb)

# Build candidate pairs
print("\n--- Candidate pairs ---")
pairs = []
for i in range(len(train_s1_ids)):
    sid = train_s1_ids[i]
    for j in range(TOP_K):
        if s2_idx[i, j] >= 0:
            pairs.append((sid, train_s2_ids[s2_idx[i, j]], float(s2_sc[i, j]), "S2"))
        if s3_idx[i, j] >= 0:
            pairs.append((sid, train_s3_ids[s3_idx[i, j]], float(s3_sc[i, j]), "S3"))

train_cand = pd.DataFrame(pairs, columns=["s1_id", "s2s3_id", "cosine_sim", "source"])
free(pairs, s2_sc, s2_idx, s3_sc, s3_idx)
print(f"Train candidates: {len(train_cand):,} pairs")


# =============================================================================
# CELL 7 — Load Features & Ground Truth
# =============================================================================
print("\n" + "="*70)
print("PHASE 2: FEATURE ENGINEERING")
print("="*70)

s1_lk = build_lookup(load_parquet("train_source1_clean.parquet", columns=FEAT_COLS))
s2_lk = build_lookup(load_parquet("train_source2_clean.parquet", columns=FEAT_COLS))
s3_lk = build_lookup(load_parquet("train_source3_clean.parquet", columns=FEAT_COLS))

gt_df = pd.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str)
gt_map = {}
for _, row in gt_df.iterrows():
    s1 = row["source1_entity_id"]
    gt_map[s1] = set(m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip())
free(gt_df)
print(f"Ground truth: {len(gt_map):,} S1 entities")


# =============================================================================
# CELL 8 — Compute Features & Labels
# =============================================================================
print(f"\nComputing {len(train_cand):,} features...")
t0 = time.time()

feats, labels = [], []
for _, r in tqdm(train_cand.iterrows(), total=len(train_cand)):
    feats.append(compute_features(r["s1_id"], r["s2s3_id"], r["cosine_sim"], r["source"], s1_lk, s2_lk, s3_lk))
    labels.append(1 if r["s2s3_id"] in gt_map.get(r["s1_id"], set()) else 0)

X = np.array(feats, dtype=np.float32)
y = np.array(labels, dtype=np.int32)
free(feats, labels)

n_pos = y.sum()
print(f"Done in {time.time()-t0:.0f}s | Pos: {n_pos:,} ({100*n_pos/len(y):.2f}%) | Neg: {len(y)-n_pos:,}")


# =============================================================================
# CELL 9 — Train LightGBM
# =============================================================================
print("\n" + "="*70)
print("PHASE 3: LIGHTGBM")
print("="*70)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

ds_tr = lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES)
ds_va = lgb.Dataset(X_va, y_va, feature_name=FEATURE_NAMES, reference=ds_tr)

params = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "scale_pos_weight": (len(y)-n_pos) / max(n_pos, 1),
    "verbose": -1, "n_jobs": 4, "seed": 42,
}

model = lgb.train(params, ds_tr, num_boost_round=1000, valid_sets=[ds_va], valid_names=["val"],
                  callbacks=[lgb.log_evaluation(50), lgb.early_stopping(50)])

print(f"\nBest iter: {model.best_iteration}")
for n, v in sorted(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain")), key=lambda x: -x[1]):
    print(f"  {n:25s} {v:>10.0f}")


# =============================================================================
# CELL 10 — Tune Threshold (F0.5)
# =============================================================================
print("\n" + "="*70)
print("PHASE 4: THRESHOLD TUNING")
print("="*70)

va_pred = model.predict(X_va)
best_t, best_f = 0.5, 0.0
for t in np.arange(0.10, 0.95, 0.01):
    f = fbeta(y_va, (va_pred >= t).astype(int))
    if f > best_f: best_t, best_f = t, f

yh = (va_pred >= best_t).astype(int)
print(f"Threshold: {best_t:.2f}")
print(f"Precision: {precision_score(y_va, yh, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_va, yh, zero_division=0):.4f}")
print(f"F0.5:      {best_f:.4f}")


# =============================================================================
# CELL 11 — TEST: Embed, Block, Feature, Predict
# =============================================================================
print("\n" + "="*70)
print("PHASE 5: TEST INFERENCE")
print("="*70)

test_s1_df = load_parquet("test_source1_clean.parquet", columns=EMBED_COLS)
test_s1_emb = get_embeddings(embed_model, test_s1_df, "test_s1")
test_s1_ids = test_s1_df["entity_id"].values
free(test_s1_df)

print("\n--- Test S2 ---")
test_s2_df = load_parquet("test_source2_clean.parquet", columns=EMBED_COLS)
test_s2_emb = get_embeddings(embed_model, test_s2_df, "test_s2")
test_s2_ids = test_s2_df["entity_id"].values
free(test_s2_df)
idx2 = build_faiss_index(test_s2_emb)
sc2, ix2 = search_candidates(test_s1_emb, idx2)
free(idx2, test_s2_emb)

print("\n--- Test S3 ---")
test_s3_df = load_parquet("test_source3_clean.parquet", columns=EMBED_COLS)
test_s3_emb = get_embeddings(embed_model, test_s3_df, "test_s3")
test_s3_ids = test_s3_df["entity_id"].values
free(test_s3_df)
idx3 = build_faiss_index(test_s3_emb)
sc3, ix3 = search_candidates(test_s1_emb, idx3)
free(idx3, test_s3_emb, test_s1_emb)

free(embed_model)
print("Embedding model freed")

# Build test candidates
tp = []
for i in range(len(test_s1_ids)):
    sid = test_s1_ids[i]
    for j in range(TOP_K):
        if ix2[i, j] >= 0: tp.append((sid, test_s2_ids[ix2[i, j]], float(sc2[i, j]), "S2"))
        if ix3[i, j] >= 0: tp.append((sid, test_s3_ids[ix3[i, j]], float(sc3[i, j]), "S3"))

test_cand = pd.DataFrame(tp, columns=["s1_id", "s2s3_id", "cosine_sim", "source"])
free(tp, sc2, ix2, sc3, ix3)
print(f"Test candidates: {len(test_cand):,}")

# Features
ts1_lk = build_lookup(load_parquet("test_source1_clean.parquet", columns=FEAT_COLS))
ts2_lk = build_lookup(load_parquet("test_source2_clean.parquet", columns=FEAT_COLS))
ts3_lk = build_lookup(load_parquet("test_source3_clean.parquet", columns=FEAT_COLS))

print(f"Computing {len(test_cand):,} test features...")
t0 = time.time()
tf = []
for _, r in tqdm(test_cand.iterrows(), total=len(test_cand)):
    tf.append(compute_features(r["s1_id"], r["s2s3_id"], r["cosine_sim"], r["source"], ts1_lk, ts2_lk, ts3_lk))

X_test = np.array(tf, dtype=np.float32)
free(tf)
print(f"Done in {time.time()-t0:.0f}s")

# Predict
test_cand["score"] = model.predict(X_test)
test_cand["match"] = (test_cand["score"] >= best_t).astype(int)
print(f"Predicted matches: {test_cand['match'].sum():,} / {len(test_cand):,}")


# =============================================================================
# CELL 12 — Generate Submission Files
# =============================================================================
print("\n" + "="*70)
print("PHASE 6: SUBMISSION")
print("="*70)

matched = test_cand[test_cand["match"] == 1]
match_grp = matched.groupby("s1_id")["s2s3_id"].apply(lambda x: ",".join(x)).to_dict()
cand_grp  = test_cand.groupby("s1_id")["s2s3_id"].apply(lambda x: ",".join(x)).to_dict()

all_s1 = pd.read_parquet(os.path.join(DATA_DIR, "test_source1_clean.parquet"), columns=["entity_id"])["entity_id"].values

with open("/home/sagemaker-user/matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1 in all_s1:
        f.write(f"{s1}\t{match_grp.get(s1, '')}\n")

with open("/home/sagemaker-user/candidate_pairs.tsv", "w") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1 in all_s1:
        f.write(f"{s1}\t{cand_grp.get(s1, '')}\n")

import boto3
s3c = boto3.client("s3", region_name="ap-southeast-2")
s3c.upload_file("/home/sagemaker-user/matching_results.tsv", BUCKET, f"{OUT_PREFIX}matching_results.tsv")
s3c.upload_file("/home/sagemaker-user/candidate_pairs.tsv", BUCKET, f"{OUT_PREFIX}candidate_pairs.tsv")

n_match = sum(1 for s1 in all_s1 if s1 in match_grp)
print(f"matching_results.tsv:  {len(all_s1):,} rows ({n_match:,} with matches)")
print(f"candidate_pairs.tsv:   {len(all_s1):,} rows")
print(f"Uploaded to s3://{BUCKET}/{OUT_PREFIX}")


# =============================================================================
# CELL 13 — Save Model
# =============================================================================
import pickle
with open("/home/sagemaker-user/lgb_model.pkl", "wb") as f:
    pickle.dump({"model": model, "threshold": best_t, "features": FEATURE_NAMES,
                 "config": {"DEV_MODE": DEV_MODE, "TOP_K": TOP_K, "EMBED_MODEL": EMBED_MODEL_ID}}, f)
s3c.upload_file("/home/sagemaker-user/lgb_model.pkl", BUCKET, "models/lgb_model.pkl")
print(f"Model saved to s3://{BUCKET}/models/lgb_model.pkl")
print("\nPIPELINE COMPLETE!")
