# ╔══════════════════════════════════════════════════════════════════════╗
# ║  QWEN 2.5 7B INSTRUCT — HIGH PERFORMANCE PIPELINE (ml.g5.8xlarge)   ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ==============================================================================
# CELL 1: Environment Setup & Hardware Check
# ==============================================================================
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import sys
import torch
import faiss
import numpy as np
import pandas as pd
import boto3
from tqdm import tqdm
import psutil

ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
print(f"System RAM: {ram_gb:.1f} GB")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Active: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
else:
    print("❌ WARNING: No GPU detected! Qwen 7B requires GPU runtime.")


# ==============================================================================
# CELL 2: Configuration & S3 / Local Paths
# ==============================================================================
S3_BUCKET = "hackathon-momenta-team"
PREFIX = "data/processed/"
OUT_PREFIX = "data/output/"
LOCAL_DATA_DIR = "/home/sagemaker-user/dataset/processed" if os.path.exists("/home/sagemaker-user/dataset/processed") else "dataset/processed"
LOCAL_OUT_DIR = "/tmp/qwen_er_output"

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
        if filename.endswith(".parquet"):
            return pd.read_parquet(local_path)
        else:
            return pd.read_csv(local_path, sep="\t")
    else:
        print(f"Downloading from S3: s3://{S3_BUCKET}/{PREFIX}{filename}")
        s3 = boto3.client("s3")
        tmp_path = os.path.join(LOCAL_OUT_DIR, filename)
        s3.download_file(S3_BUCKET, PREFIX + filename, tmp_path)
        if filename.endswith(".parquet"):
            return pd.read_parquet(tmp_path)
        else:
            return pd.read_csv(tmp_path, sep="\t")

print("Ready to load dataset files...")


# ==============================================================================
# CELL 3: Load Qwen 2.5 7B Instruct Model in FP16
# ==============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading native FP16 model: {MODEL_ID} on GPU...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()

vram_used = torch.cuda.memory_allocated() / (1024**3)
print(f"✓ Qwen 2.5 7B loaded successfully! VRAM Allocated: {vram_used:.2f} GB")


# ==============================================================================
# CELL 4: High-Speed FAISS Candidate Generation (Blocking Stage)
# ==============================================================================
import torch.nn.functional as F
from transformers import AutoModel

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

print("1. Loading test datasets via load_dataset()...")
test_s1 = load_dataset("test_s1")
test_s2 = load_dataset("test_s2")
test_s3 = load_dataset("test_s3")

s1_id_col = get_id_col(test_s1)
s2_id_col = get_id_col(test_s2)
s3_id_col = get_id_col(test_s3)

test_s1["source1_entity_id"] = test_s1[s1_id_col]
test_s2["target_entity_id"] = test_s2[s2_id_col]
test_s3["target_entity_id"] = test_s3[s3_id_col]

test_s1["text_to_embed"] = get_text_series(test_s1)
test_s2["text_to_embed"] = get_text_series(test_s2)
test_s3["text_to_embed"] = get_text_series(test_s3)

test_s2_s3 = pd.concat([test_s2[["target_entity_id", "text_to_embed"]], test_s3[["target_entity_id", "text_to_embed"]]], ignore_index=True)
del test_s2, test_s3
gc.collect()

print(f"Loaded S1 ({len(test_s1):,} rows) | Target S2/S3 ({len(test_s2_s3):,} rows)")

print("2. Encoding embeddings with multilingual-e5-small on GPU...")
E5_MODEL_NAME = "intfloat/multilingual-e5-small"
e5_tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
e5_model = AutoModel.from_pretrained(E5_MODEL_NAME).to("cuda").half()
e5_model.eval()

def encode_embeddings(texts, batch_size=1024):
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = ["passage: " + t for t in texts[i:i + batch_size]]
        inputs = e5_tokenizer(batch, max_length=128, padding=True, truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = e5_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
            normed = F.normalize(pooled, p=2, dim=1)
            all_emb.append(normed.cpu().to(torch.float32).numpy())
    return np.vstack(all_emb)

emb_s1 = encode_embeddings(test_s1["text_to_embed"].tolist(), batch_size=1024)
emb_s23 = encode_embeddings(test_s2_s3["text_to_embed"].tolist(), batch_size=1024)

del e5_model, e5_tokenizer
gc.collect()
torch.cuda.empty_cache()

print("3. Searching top-5 candidates per S1 entity with FAISS...")
d = emb_s1.shape[1]
index = faiss.IndexFlatIP(d)
index.add(emb_s23)

TOP_K = 5
D, I = index.search(emb_s1, TOP_K)

s1_ids = test_s1["source1_entity_id"].values
target_ids = test_s2_s3["target_entity_id"].values

records = []
for i in range(len(s1_ids)):
    s1_id = s1_ids[i]
    for k in range(TOP_K):
        match_idx = I[i][k]
        records.append((s1_id, target_ids[match_idx], float(D[i][k])))

candidate_pairs_df = pd.DataFrame(records, columns=["source1_entity_id", "candidate_entity_id", "similarity_score"])
print(f"✓ Generated {len(candidate_pairs_df):,} candidate pairs!")


# ==============================================================================
# CELL 5: Qwen 2.5 7B LLM Pair Judging
# ==============================================================================
MATCH_PROMPT = """You are an expert at business entity resolution. Determine if the two business records below refer to the SAME real-world entity.

Consider business name similarity, address proximity, brand names, and country code. Minor spelling differences, legal entity suffix variations (e.g. Inc vs LLC), abbreviations, and formatting changes are common and should NOT prevent a match.

Record A:
  Name: {name_a}
  Address: {addr_a}
  Country: {country_a}

Record B:
  Name: {name_b}
  Address: {addr_b}
  Country: {country_b}

Answer with ONLY ONE WORD: "MATCH" or "NO_MATCH".
"""

def judge_pair(name_a, addr_a, country_a, name_b, addr_b, country_b):
    prompt = MATCH_PROMPT.format(
        name_a=str(name_a or "").strip(),
        addr_a=str(addr_a or "").strip(),
        country_a=str(country_a or "").strip(),
        name_b=str(name_b or "").strip(),
        addr_b=str(addr_b or "").strip(),
        country_b=str(country_b or "").strip(),
    )
    messages = [
        {"role": "system", "content": "You are a precise entity resolution judge. Respond with only MATCH or NO_MATCH."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.01, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()
    return "MATCH" if "MATCH" in response and "NO_MATCH" not in response and "NO MATCH" not in response else "NO_MATCH"


# ==============================================================================
# CELL 6: Process Candidate Pairs & Upload Results to S3
# ==============================================================================
def evaluate_and_format_submission(test_s1_df, candidate_pairs_df, max_pairs=None):
    subset = candidate_pairs_df.head(max_pairs) if max_pairs else candidate_pairs_df
    print(f"Evaluating {len(subset):,} candidate pairs with Qwen 2.5 7B...")
    
    s1_map = test_s1_df.set_index("source1_entity_id").to_dict(orient="index")
    s23_map = test_s2_s3.set_index("target_entity_id").to_dict(orient="index")
    
    matches_dict = {s1: [] for s1 in test_s1_df["source1_entity_id"].unique()}
    candidates_dict = {s1: [] for s1 in test_s1_df["source1_entity_id"].unique()}
    
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Judging pairs"):
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        candidates_dict[s1_id].append(cand_id)
        
        rec_a = s1_map.get(s1_id, {})
        rec_b = s23_map.get(cand_id, {})
        
        decision = judge_pair(
            rec_a.get("name_clean", rec_a.get("name", "")),
            rec_a.get("addr_clean", rec_a.get("address", "")),
            rec_a.get("country", ""),
            rec_b.get("name_clean", rec_b.get("name", "")),
            rec_b.get("addr_clean", rec_b.get("address", ""))
        )
        if decision == "MATCH":
            matches_dict[s1_id].append(cand_id)
            
    matching_rows = [{"source1_entity_id": s1, "matched_entity_ids": ",".join(m)} for s1, m in matches_dict.items()]
    candidate_rows = [{"source1_entity_id": s1, "candidate_entity_ids": ",".join(c)} for s1, c in candidates_dict.items()]
    
    matching_path = os.path.join(LOCAL_OUT_DIR, "matching_results.tsv")
    candidate_path = os.path.join(LOCAL_OUT_DIR, "candidate_pairs.tsv")
    
    pd.DataFrame(matching_rows).to_csv(matching_path, sep="\t", index=False)
    pd.DataFrame(candidate_rows).to_csv(candidate_path, sep="\t", index=False)
    
    print(f"✓ Local matching_results.tsv created: {matching_path}")
    print(f"✓ Local candidate_pairs.tsv created: {candidate_path}")
    
    s3 = boto3.client("s3")
    s3.upload_file(matching_path, S3_BUCKET, OUT_PREFIX + "matching_results.tsv")
    s3.upload_file(candidate_path, S3_BUCKET, OUT_PREFIX + "candidate_pairs.tsv")
    print(f"✓ Uploaded results to s3://{S3_BUCKET}/{OUT_PREFIX}")
    return matching_path, candidate_path

print("Qwen 2.5 7B script updated successfully with custom Cell 2 logic!")
