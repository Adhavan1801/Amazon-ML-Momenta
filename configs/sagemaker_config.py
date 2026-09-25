"""
SageMaker configuration for entity resolution pipeline.

This file contains all instance types, model identifiers, S3 paths,
and cost estimates. Import from here instead of hardcoding values.

Models deployed on SageMaker (GPU):
  1. multilingual-e5-small  — embedding model for blocking/shortlisting
  2. cross-encoder (multilingual-e5-base / XLM-R) — optional re-ranker
  3. Qwen2.5-7B-Instruct — optional LLM judge for borderline pairs

LightGBM runs locally (no GPU needed).
"""

# ══════════════════════════════════════════════════════════════════════
# AWS / REGION
# ══════════════════════════════════════════════════════════════════════
AWS_REGION = "ap-southeast-2"           # Sydney — same as S3 bucket
AWS_ACCOUNT_ID = "866923471207"
S3_BUCKET = "hackathon-momenta-team"
S3_DATA_PREFIX = "data/processed"
S3_MODEL_PREFIX = "models"
S3_OUTPUT_PREFIX = "output"

# ══════════════════════════════════════════════════════════════════════
# IAM
# ══════════════════════════════════════════════════════════════════════
SAGEMAKER_ROLE_NAME = "SageMakerExecutionRole-Momenta"
# This will be set after role creation:
SAGEMAKER_ROLE_ARN = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{SAGEMAKER_ROLE_NAME}"

# ══════════════════════════════════════════════════════════════════════
# NOTEBOOK INSTANCE
# ══════════════════════════════════════════════════════════════════════
NOTEBOOK_INSTANCE_NAME = "momenta-entity-resolution"

# Instance for embedding + cross-encoder (T4 GPU, 16GB VRAM, 16GB RAM)
# Cost: ~$0.736/hr in ap-southeast-2
NOTEBOOK_INSTANCE_TYPE_GPU = "ml.g4dn.xlarge"

# Fallback CPU instance for debugging (no GPU)
# Cost: ~$0.269/hr
NOTEBOOK_INSTANCE_TYPE_CPU = "ml.m5.xlarge"

# Volume size in GB — needs to hold models + data
NOTEBOOK_VOLUME_SIZE_GB = 50

# ══════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════

# Model 1: Embedding model for blocking/shortlisting
EMBEDDING_MODEL = {
    "name": "multilingual-e5-small",
    "hf_model_id": "intfloat/multilingual-e5-small",
    "params": "~118M",
    "vram_needed": "~1 GB",
    "embedding_dim": 384,
    "max_seq_length": 512,
    "batch_size_gpu": 512,       # T4 can handle this easily
    "batch_size_cpu": 64,
    "query_prefix": "query: ",   # Required by E5 models
    "passage_prefix": "passage: ",
    "instance_type": "ml.g4dn.xlarge",
    "estimated_speed": "~5,000 records/sec on T4",
    "cost_per_hour": 0.736,
}

# Alternative embedding model (slightly larger, potentially better)
EMBEDDING_MODEL_ALT = {
    "name": "paraphrase-multilingual-MiniLM-L12-v2",
    "hf_model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "params": "~118M",
    "vram_needed": "~1 GB",
    "embedding_dim": 384,
    "max_seq_length": 128,
    "batch_size_gpu": 512,
    "batch_size_cpu": 64,
    "query_prefix": "",          # No prefix needed
    "passage_prefix": "",
    "instance_type": "ml.g4dn.xlarge",
    "estimated_speed": "~5,000 records/sec on T4",
    "cost_per_hour": 0.736,
}

# Model 2: Cross-encoder for re-ranking candidate pairs
CROSS_ENCODER_MODEL = {
    "name": "multilingual-e5-base-cross-encoder",
    "hf_model_id": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Start with this, fine-tune later
    "alt_hf_model_id": "intfloat/multilingual-e5-base",       # Can fine-tune as cross-encoder
    "params": "~110M",
    "vram_needed": "~2 GB",
    "max_seq_length": 512,
    "batch_size_gpu": 256,
    "batch_size_cpu": 32,
    "instance_type": "ml.g4dn.xlarge",
    "estimated_speed": "~2,000 pairs/sec on T4",
    "cost_per_hour": 0.736,
}

# Model 3: LLM judge for borderline pairs (optional)
LLM_JUDGE_MODEL = {
    "name": "Qwen2.5-7B-Instruct",
    "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
    "params": "7B",
    "vram_needed": "~14 GB (fp16) / ~7 GB (GPTQ-4bit)",
    "max_seq_length": 2048,
    "batch_size_gpu": 1,         # LLM generation is sequential
    "instance_type": "ml.g4dn.xlarge",   # T4 16GB — tight but works with 4-bit quant
    "alt_instance_type": "ml.g5.xlarge",  # A10G 24GB — comfortable for fp16
    "estimated_speed": "~5-10 pairs/sec (generation is slow)",
    "cost_per_hour_g4dn": 0.736,
    "cost_per_hour_g5": 1.408,
    "note": "Only use on borderline pairs (top ~1-5% uncertain). NOT for full dataset.",
}

# ══════════════════════════════════════════════════════════════════════
# COST ESTIMATES (ap-southeast-2 on-demand pricing)
# ══════════════════════════════════════════════════════════════════════
COST_ESTIMATES = {
    "ml.g4dn.xlarge": {
        "per_hour": 0.736,
        "gpu": "1x NVIDIA T4 (16 GB)",
        "vcpu": 4,
        "ram_gb": 16,
    },
    "ml.g5.xlarge": {
        "per_hour": 1.408,
        "gpu": "1x NVIDIA A10G (24 GB)",
        "vcpu": 4,
        "ram_gb": 16,
    },
    "ml.m5.xlarge": {
        "per_hour": 0.269,
        "gpu": "None (CPU only)",
        "vcpu": 4,
        "ram_gb": 16,
    },
}
