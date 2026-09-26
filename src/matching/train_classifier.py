"""
Stage 3: GBDT Classifier Training
Trains a LightGBM / XGBoost model on ground truth pair features to output precise matching probabilities.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, fbeta_score

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import GradientBoostingClassifier

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from configs.paths import MODELS_DIR


FEATURE_COLS = [
    "dense_score",
    "name_jaro", "name_ratio", "name_token_sort", "name_token_set",
    "addr_jaro", "addr_ratio", "addr_token_sort", "addr_token_set",
    "country_match", "postal_match", "house_num_overlap"
]


def train_gbdt_model(feature_df, labels, model_save_path=None):
    """
    Train a LightGBM classifier with precision & F0.5 optimization.
    """
    print(f"[MODEL] Training LightGBM Classifier on {len(feature_df):,} labeled candidate pairs...")
    X = feature_df[FEATURE_COLS]
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train Set: {len(X_train):,} samples (Positives: {sum(y_train):,})")
    print(f"  Val Set:   {len(X_val):,} samples (Positives: {sum(y_val):,})")

    if HAS_LGB:
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    else:
        clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
        clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]

    # Threshold optimization for F_0.5
    best_thresh = 0.5
    best_f05 = 0.0
    for thresh in np.arange(0.3, 0.98, 0.02):
        preds = (val_probs >= thresh).astype(int)
        score = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        if score > best_f05:
            best_f05 = score
            best_thresh = thresh

    print(f"\n[VALIDATION] Validation Complete!")
    print(f"  [RESULT] Best F_0.5 Score: {best_f05:.4f} at Decision Threshold: {best_thresh:.2f}")

    val_preds = (val_probs >= best_thresh).astype(int)
    print("\n" + classification_report(y_val, val_preds, target_names=["NO_MATCH", "MATCH"], digits=4))

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, "wb") as f:
            pickle.dump({"model": clf, "optimal_threshold": best_thresh, "feature_cols": FEATURE_COLS}, f)
        print(f"[SUCCESS] Model saved to {model_save_path}")

    return clf, best_thresh


if __name__ == "__main__":
    print("Testing Train Classifier Module...")
