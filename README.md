# Entity Resolution Pipeline — Multi-Stage Hybrid Architecture

> A high-performance, scalable entity resolution framework designed for multi-source business record deduplication and entity matching.

---

## Architecture & System Design

The system employs a multi-stage funnel architecture designed to reduce candidate search complexity while maximizing precision under the $F_{0.5}$ metric:

```
[ Raw Source Datasets (S1, S2, S3) ]
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 0: Data Preprocessing & Normalization             │
│ • Legal Suffix Removal (Inc, LLC, Ltd, Pvt, etc.)       │
│ • Address Standardization (Road, Street, Avenue)        │
│ • Extraction of Postal Codes, Street Numbers & States   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Candidate Generation (Blocking)                │
│ • Dense Retrieval: Bi-Encoder (multilingual-e5-small)   │
│ • Vector Indexing: GPU FAISS Approximate Search (Top-K) │
│ • Sparse Retrieval: TF-IDF & Character N-gram Indexing  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Pairwise Feature Engineering                   │
│ • String Distance Metrics: Jaro-Winkler, Levenshtein    │
│ • Token Overlap: Token-Sort & Token-Set Similarity      │
│ • Structured Attribute Flags: Country, Postal, House #  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Supervised Classification & Reranking           │
│ • Model: Gradient Boosted Decision Trees (LightGBM)     │
│ • Optimization: Continuous Match Probability Scoring    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Precision Filtering & Threshold Calibration     │
│ • Decision Threshold Tuning for F0.5 Score              │
│ • Rule-Based Constraint Shields (Country/House Mismatch)│
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
[ Output: matching_results.tsv & candidate_pairs.tsv ]
```

---

## Mathematical Objective ($F_{0.5}$ Metric)

Evaluation is performed using the $F_{0.5}$ score, which places a higher weighting on Precision relative to Recall:

$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

To optimize for this objective, the post-processing pipeline enforces strict precision constraints via rule-based veto filters and calibrated probability boundaries.

---

## Directory Structure

```
.
├── configs/                             # Configuration and path definitions
│   ├── paths.py
│   └── sagemaker_config.py
├── src/                                 # Core source code
│   ├── preprocessing/                   # Data normalization and feature cleaning
│   │   ├── normalize.py
│   │   ├── constants.py
│   │   └── run_preprocessing.py
│   ├── blocking/                        # Candidate generation and index search
│   │   └── hybrid_blocking.py
│   ├── matching/                        # Pairwise feature extraction & modeling
│   │   ├── feature_extraction.py
│   │   ├── train_classifier.py
│   │   └── predict_submission.py
│   ├── run_pipeline.py                  # Pipeline execution CLI
│   └── validate_submission.py           # Submission format validator
├── notebooks/                           # Execution notebooks and cloud scripts
│   ├── hybrid_funnel_entity_resolution.ipynb
│   └── hybrid_funnel_entity_resolution.py
├── requirements.txt                     # Project dependencies
└── README.md                            # Documentation
```

---

## Usage & Execution

### Local Command Line Execution

```powershell
# Subsampled run (for fast validation)
.\momenta\Scripts\python.exe src/run_pipeline.py --sample_size 10000 --top_k 5

# Full dataset execution
.\momenta\Scripts\python.exe src/run_pipeline.py --top_k 5
```

### Cloud GPU Execution (AWS SageMaker)

Execute `notebooks/hybrid_funnel_entity_resolution.ipynb` or run the standalone script:

```bash
python notebooks/hybrid_funnel_entity_resolution.py
```

### Submission Validation

Verify output schema compliance:

```powershell
python src/validate_submission.py
```
