import numpy as np


def auc(y, s):
    """ROC AUC via rank statistic (no sklearn dependency)."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
