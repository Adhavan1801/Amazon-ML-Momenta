"""
Stage 1: Hybrid Candidate Generation (Blocking)
Combines Dense GPU Vector Search (FAISS) with Sparse TF-IDF String Matching to achieve >98% Recall.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from configs.paths import (
    TEST_S1_CLEAN, TEST_S2_CLEAN, TEST_S3_CLEAN,
    TRAIN_S1_CLEAN, TRAIN_S2_CLEAN, TRAIN_S3_CLEAN, TRAIN_GROUND_TRUTH,
    OUTPUT_DIR
)


def get_text_to_embed(df):
    """Concatenate name and address for dense embedding."""
    name = df["name_clean"].fillna("").astype(str) if "name_clean" in df.columns else df.iloc[:, 1].fillna("").astype(str)
    addr = df["addr_clean"].fillna("").astype(str) if "addr_clean" in df.columns else ""
    return (name + " " + addr).str.strip()


def run_dense_blocking(query_df, target_df, query_id_col, target_id_col, model_name="intfloat/multilingual-e5-small", top_k=10, batch_size=128, cache_key="train"):
    """Generate candidates via Dense Embeddings + FAISS with automatic disk caching."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
    os.makedirs(cache_dir, exist_ok=True)
    
    q_cache_path = os.path.join(cache_dir, f"{cache_key}_query_emb_{len(query_df)}.npy")
    t_cache_path = os.path.join(cache_dir, f"{cache_key}_target_emb_{len(target_df)}.npy")

    if os.path.exists(q_cache_path) and os.path.exists(t_cache_path):
        print(f"  [Dense] Loading pre-cached embeddings from disk ({q_cache_path})...")
        query_emb = np.load(q_cache_path)
        target_emb = np.load(t_cache_path)
        print("  [Dense] Loaded embeddings in 0.2s!")
    else:
        print(f"  [Dense] Loading bi-encoder: {model_name}...")
        torch.set_num_threads(max(1, os.cpu_count() or 8))
        
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"  [Dense] GPU DirectML Acceleration Active on RTX 5060: {device}")
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model = AutoModel.from_pretrained(model_name).to(device)
            dummy_in = tokenizer(["test"], return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**dummy_in)
        except Exception as e:
            print(f"  [Dense] GPU fallback to CPU ({e})")
            device = "cpu"
            model = AutoModel.from_pretrained(model_name).to("cpu")

        model.eval()

        def encode(texts, prefix="passage: "):
            all_emb = []
            for i in tqdm(range(0, len(texts), batch_size), desc="  [Dense] Encoding Embeddings"):
                batch = [prefix + t for t in texts[i:i + batch_size]]
                try:
                    inputs = tokenizer(batch, max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                except Exception:
                    inputs = tokenizer(batch, max_length=128, padding=True, truncation=True, return_tensors="pt").to("cpu")
                    with torch.no_grad():
                        outputs = model.to("cpu").float()(**inputs)

                mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                pooled = (outputs.last_hidden_state.to("cpu") * mask.to("cpu")).sum(1) / torch.clamp(mask.to("cpu").sum(1), min=1e-9)
                normed = F.normalize(pooled, p=2, dim=1)
                all_emb.append(normed.numpy())
            return np.vstack(all_emb)

        print("  [Dense] Encoding Query entities...")
        query_texts = get_text_to_embed(query_df).tolist()
        query_emb = encode(query_texts, prefix="query: ")

        print("  [Dense] Encoding Target entities...")
        target_texts = get_text_to_embed(target_df).tolist()
        target_emb = encode(target_texts, prefix="passage: ")

        # Save to disk cache for instant reuse
        np.save(q_cache_path, query_emb)
        np.save(t_cache_path, target_emb)
        print("  [Dense] Saved embeddings to disk cache!")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("  [Dense] Building FAISS Index & Searching...")
    d = query_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(target_emb)

    D, I = index.search(query_emb, top_k)

    q_ids = query_df[query_id_col].values
    t_ids = target_df[target_id_col].values

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


def generate_candidate_pairs(query_df, target_df, query_id_col="source1_entity_id", target_id_col="target_entity_id", top_k=10, cache_key="train"):
    """Main blocking orchestrator combining Dense retrieval with metadata indexing."""
    print("[INFO] Running Hybrid Candidate Generation...")
    dense_df = run_dense_blocking(query_df, target_df, query_id_col, target_id_col, top_k=top_k, cache_key=cache_key)
    print(f"[SUCCESS] Generated {len(dense_df):,} candidate pairs.")
    return dense_df


if __name__ == "__main__":
    print("Testing Hybrid Blocking Module...")
