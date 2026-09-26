# Business Entity Resolution — High Performance Pipeline & System Architecture

Pipeline: **clean → shortlist (blocking) → prune → features → 2-stage LightGBM + XGBoost → decide**.
Everything is CPU-only and uses only the challenge data (no external lookups). All libraries are MIT, BSD or Apache-2.0.

---

## 🏆 Model Leaderboard & Code Versions

| Validation $F_{0.5}$ | Git Release Tag | Key Model Changes / Config | Experiment Folder | Output TSV |
| :--- | :--- | :--- | :--- | :--- |
| **`0.9807`** (98.07%) | [`f05-0.9807`](https://github.com/Adhavan1801/Amazon-ML-Momenta/releases/tag/f05-0.9807) | Cross-Encoder (`xlm-roberta-base`) GPU Re-Ranking on Borderline Pairs (1:1 Blend) | [`experiments/f05_0.9807_ce/`](experiments/f05_0.9807_ce/run_info.json) | [`output/matching_results.tsv`](output/matching_results.tsv) |
| `0.9777` (97.77%) | [`f05-0.9777`](https://github.com/Adhavan1801/Amazon-ML-Momenta/releases/tag/f05-0.9777) | +7 Domain Features (name/addr token Jaccard & min overlap, prefix match, len delta, zip5) | [`experiments/f05_0.9777/`](experiments/f05_0.9777/run_info.json) | [`output/matching_results.tsv`](output/matching_results.tsv) |
| `0.9776` (97.76%) | [`f05-0.9776`](https://github.com/Adhavan1801/Amazon-ML-Momenta/releases/tag/f05-0.9776) | Baseline XGBoost + LightGBM 2-Stage Cascade (40% S1, 900 rounds) | [`experiments/f05_0.9776/`](experiments/f05_0.9776/run_info.json) | [`output/matching_results.tsv`](output/matching_results.tsv) |

> **Tip to checkout exact code for a score:**
> ```bash
> git checkout f05-0.9777
> ```

---

## 1. Setup (once)

```bash
# Python 3.11
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Run Everything (One Command)

```bash
python src/run_all.py \
  --data  <...>/student_resource/dataset \
  --work  <scratch folder, needs ~40 GB free> \
  --out   <folder for the two output .tsv files> \
  --mem-gb 32 --threads 20 \
  --validate <...>/student_resource/utils/validate_submission.py
```

Example on Windows PC:

```powershell
python src\run_all.py --data D:\ml\student_resource\dataset `
  --work D:\ml\work --out D:\ml\output --mem-gb 32 --threads 20 `
  --validate D:\ml\student_resource\utils\validate_submission.py
```

- The default settings reproduce our best submission: XGBoost+LightGBM stage 1 on 40% of S1, 900 rounds; stage 2 with top-30 stage-1 features; pruner keeps 20 per S1 plus 3 per record ($F_{0.5} = 0.9776$).
- Hyperparameters can be adjusted via `--no-xgb --s1-pct --s1-rounds --s2-topk --prune-k --prune-krev --prune-pmin --prune-load-k`.
- `--mem-gb`: Set to machine RAM. With $\ge 24$, the pipeline uses bigger, faster chunks.
- `--threads`: CPU threads to use (default: all cores).
- **Resume Capability:** Each finished step leaves a marker in `<work>/_done/`. If a run stops or restarts, re-running the command resumes where it left off.

**Output:** `<out>/matching_results.tsv` and `<out>/candidate_pairs.tsv`.

---

## 3. Pipeline Steps (`run_all.py` Execution Flow)

| Step | Command (`python src/cli.py …`) | Reads → Writes | Typical Time |
|---|---|---|---|
| convert | `convert` | dataset `.tsv` → `work/data/*.parquet` | ~1 min |
| dict | `dict` | train pairs → `artifacts/indic_dict.json` (Indic→Latin words) | ~1 min |
| prep_train / prep_test | `prep train` / `prep test` | normalized names and addresses → `work/{split}_recs.parquet` | ~2 min |
| gtidx | `gtidx` | ground truth → `work/train_gt_idx.parquet` | <1 min |
| tokens_* | `tokens train` / `tokens test` | typed tokens + idf index → `work/{split}_tok/` | ~3 min |
| block_*_<country>, finish_* | `block train india` … | candidate pairs → `work/{split}_pairs/pairs_<country>.parquet` | ~10 min |
| prune_train / prune_test | `prune train` / `prune test` | pruner LightGBM, ~5.5 candidates per S1 → `work/{split}_pruned/` | ~4 min |
| recall | `recall` | prints candidate recall ceiling after blocking and pruning | <1 min |
| recfeat_*, feats_*_<country> | `recfeat train`, `feats train india` … | ~76 pair features → `work/{split}_feats/` | ~10 min |
| ctx_* | `ctx train` / `ctx test` | rank/gap context features → `work/{split}_ctx/` | ~2 min |
| train1, pred1_train | `train1`, `pred1 train` | stage-1 LightGBM (2 folds) + out-of-fold p1 | ~5 min |
| train2, eval | `train2`, `eval` | stage-2 LightGBM + threshold tuned on held-out macro F0.5 | ~5 min |
| pred1_test, predict | `pred1 test`, `predict` | final `matching_results.tsv` + `candidate_pairs.tsv` | ~5 min |

---

## 4. Partial Re-execution

```bash
python src/run_all.py ... --list                 # shows steps and status
python src/run_all.py ... --from feats_train_india   # re-run from a step onward
python src/run_all.py ... --only eval            # re-run single step
```

| Modified File | Re-run starting from |
|---|---|
| `normalize.py` (cleaning rules) | `prep_train` |
| `block.py` (candidate shortlisting) | `tokens_train` |
| `prune.py` (pruner model) | `prune_train` (delete `work/artifacts/pruner.txt`) |
| `feat.py` (pair similarity features) | `recfeat_train` |
| `model.py` (models, thresholding) | `train1` |

---

## 5. Codebase Overview (`src/`)

| File | Functionality |
|---|---|
| `config.py` | Environment configurations and resource memory knobs |
| `run_all.py` / `cli.py` | Pipeline orchestrator and CLI entry points |
| `data.py` | Data format conversion (TSV $\leftrightarrow$ Parquet) |
| `normalize.py` | Text normalization: Indic→Latin transliteration, legal form cleaning, address standardization |
| `learn_dict.py` | Learns Indic-script to Latin dictionary mappings from training ground truth |
| `prep.py` | Parallel dataset preparation and text cleaning |
| `block.py` | Token blocking using typed hashing and inverted indices |
| `prune.py` | LightGBM candidate pair pruning model |
| `feat.py` | High-dimensional string/token distance features (RapidFuzz) |
| `stages.py` | Memory-bounded chunk processing stages |
| `model.py` | Stage-1 & Stage-2 LightGBM model training, re-ranking, and threshold tuning |
| `metrics.py` | Macro $F_{0.5}$ evaluation metric calculator |
