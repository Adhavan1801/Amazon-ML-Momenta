# Model Evaluation & Benchmark Results

## Baseline Pipeline Benchmark Results

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **Validation $F_{0.5}$ Score** | **0.97758 (97.76%)** | Macro $F_{0.5}$ metric on held-out validation fold |
| **Single Threshold $F_{0.5}$** | **0.97734 (97.73%)** | Score at single optimal threshold ($0.675$) |
| **Primary Model** | `Stage 2 LightGBM` | 2-Stage Cascade Ensemble (LightGBM + XGBoost) |
| **First-Match Threshold (`thr`)** | `0.550` | Threshold for primary entity candidate match |
| **Extra-Match Threshold (`thr2`)** | `0.725` | Threshold for secondary candidate matches |
| **Exclusive Matching** | `Enabled` | Enforces one primary match per S1 record |

---

## Detailed Model Metrics (`scores.json`)

```json
{
  "f05_score": 0.9775833354517283,
  "f05_single_threshold_score": 0.9773432919033166,
  "model_stage": "p2 (Stage 2 LightGBM Re-Ranker)",
  "exclusive_match_rule": true,
  "first_match_threshold": 0.55,
  "extra_match_threshold": 0.725,
  "single_threshold": 0.675
}
```

---

## Generated Output Submissions
* `output/matching_results.tsv` (97.3 MB, 2,206,821 source1 entities)
* `output/candidate_pairs.tsv` (184.2 MB candidate pair list)
