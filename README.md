# Entity Resolution Pipeline — Project Structure

```
E:\hackathon\
├── .gitignore
├── README.md                          # This file
├── requirements.txt                   # Pinned dependencies
│
├── configs/                           # Configuration files (goes to git)
│   └── paths.py                       # Central path definitions
│
├── src/                               # All source code (goes to git)
│   ├── preprocessing/                 # Stage 0: Data cleaning & normalization
│   │   ├── __init__.py
│   │   ├── normalize.py               # Core normalization functions
│   │   ├── run_preprocessing.py       # Main script to run preprocessing
│   │   └── constants.py               # Abbreviation maps, stopwords, etc.
│   │
│   ├── blocking/                      # Stage 1-2: Candidate generation
│   │   └── __init__.py
│   │
│   ├── matching/                      # Stage 3-4: Feature eng + classifier
│   │   └── __init__.py
│   │
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       └── evaluation.py              # F_0.5 scoring, validation
│
├── notebooks/                         # Jupyter notebooks for EDA (goes to git)
│
├── Dataset/                           # Original data (GITIGNORED — too large)
│   └── student_resource/
│       └── dataset/{train,test}/*.tsv
│
├── data/                              # Intermediate data
│   ├── raw/                           # Symlinks or copies of original TSVs
│   └── processed/                     # Cleaned parquet files (GITIGNORED)
│
├── models/                            # Saved model checkpoints (GITIGNORED)
├── output/                            # Final submission files (GITIGNORED)
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── momenta/                           # Virtual environment (GITIGNORED)
└── private/                           # Private notebooks/scratch (GITIGNORED)
```

## What goes to Git vs What doesn't

| Goes to Git ✅ | Gitignored ❌ |
|---|---|
| `src/` (all code) | `Dataset/` (raw TSVs, ~1.5GB) |
| `configs/` | `data/processed/` (parquet files) |
| `notebooks/` | `momenta/` (venv) |
| `requirements.txt` | `models/` (checkpoints) |
| `.gitignore` | `output/` (submission TSVs) |
| `README.md` | `private/` |
