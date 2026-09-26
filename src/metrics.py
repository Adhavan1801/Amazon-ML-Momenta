"""Official metric: per-Source-1 F0.5, macro-averaged over ALL Source-1 entities (singletons included)."""
import numpy as np
import polars as pl


def f_beta(pred: set, true: set, beta=0.5) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def _key(df):
    return df.with_columns((pl.col("i").cast(pl.UInt64) * 4294967296 + pl.col("j").cast(pl.UInt64)).alias("k"))


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame) -> float:
    """pred / truth: DataFrames with columns i, j. s1: DataFrame with column i (all S1 to average over)."""
    tp = _key(pred).join(_key(truth).select("k"), on="k", how="semi").group_by("i").len().rename({"len": "tp"})
    npred = pred.group_by("i").len().rename({"len": "np"})
    ntrue = truth.group_by("i").len().rename({"len": "nt"})
    A = s1.join(npred, on="i", how="left").join(ntrue, on="i", how="left").join(tp, on="i", how="left").fill_null(0)
    p = (A["tp"] / A["np"].clip(1)).to_numpy()
    r = (A["tp"] / A["nt"].clip(1)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.nan_to_num(1.25 * p * r / (0.25 * p + r))
    f = np.where((A["np"].to_numpy() == 0) & (A["nt"].to_numpy() == 0), 1.0, f)
    return float(f.mean())
