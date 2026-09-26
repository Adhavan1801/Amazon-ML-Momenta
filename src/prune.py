"""Stage 0: cheap pruner on blocking features -> keeps a short candidate list per S1."""
import numpy as np
import polars as pl
import lightgbm as lgb

GROUP_TYPES = {"name": ["n", "k", "p", "f"], "addr": ["a", "b"], "num": ["d", "h"], "cross": ["x"], "combo": ["m", "c", "s"]}
PRUNE_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=50,
                    feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=1,
                    num_threads=__import__('config').THREADS)


def block_features(P: pl.DataFrame, tot: pl.DataFrame) -> pl.DataFrame:
    g = tot.select("idx", *[pl.sum_horizontal([c for c in ts if c in tot.columns]).alias(f"t_{n}")
                            for n, ts in GROUP_TYPES.items()])
    gi = g.rename({c: c + "_i" for c in g.columns if c != "idx"}).rename({"idx": "i"})
    gj = g.rename({c: c + "_j" for c in g.columns if c != "idx"}).rename({"idx": "j"})
    P = P.join(gi, on="i", how="left").join(gj, on="j", how="left")
    ex = []
    for n in GROUP_TYPES:
        ex += [(pl.col(f"bs_{n}") / (pl.col(f"t_{n}_i") + 1e-3)).cast(pl.Float32).alias(f"ov_{n}_i"),
               (pl.col(f"bs_{n}") / (pl.col(f"t_{n}_j") + 1e-3)).cast(pl.Float32).alias(f"ov_{n}_j")]
    P = P.with_columns(ex).with_columns(
        (pl.col("ov_name_i") + pl.col("ov_name_j")).alias("ov_name_s"),
        (pl.col("ov_addr_i") + pl.col("ov_addr_j") + pl.col("ov_num_i") + pl.col("ov_num_j")).alias("ov_an_s"),
    )
    ctx = []
    for c in ["bs", "ov_name_s", "ov_an_s"]:
        ctx += [(pl.col(c).max().over("i") - pl.col(c)).cast(pl.Float32).alias(f"{c}_gap_i"),
                (pl.col(c).max().over("j") - pl.col(c)).cast(pl.Float32).alias(f"{c}_gap_j")]
    ctx += [pl.len().over("i").cast(pl.UInt16).alias("nc_i"), pl.len().over("j").cast(pl.UInt16).alias("nc_j")]
    return P.with_columns(ctx)


def prune_feature_cols(P):
    drop = {"i", "j", "y"}
    return [c for c in P.columns if c not in drop]


def train_pruner(P: pl.DataFrame, rounds=300):
    cols = prune_feature_cols(P)
    ds = lgb.Dataset(P.select(cols).to_numpy(), P["y"].to_numpy(), feature_name=cols, free_raw_data=True)
    return lgb.train(PRUNE_PARAMS, ds, rounds), cols


def apply_pruner(P: pl.DataFrame, model, cols, k_keep=15, k_rev=2, p_min=0.005):
    p0 = np.zeros(P.height, np.float32)
    step = 2_000_000
    for s in range(0, P.height, step):
        p0[s:s + step] = model.predict(P.slice(s, step).select(cols).to_numpy())
    P = P.with_columns(pl.Series("p0", p0))
    P = P.with_columns(pl.col("p0").rank("ordinal", descending=True).over("i").cast(pl.UInt16).alias("p0_rk_i"),
                       pl.col("p0").rank("ordinal", descending=True).over("j").cast(pl.UInt16).alias("p0_rk_j"))
    keep = ((pl.col("p0_rk_i") <= k_keep) | (pl.col("p0_rk_j") <= k_rev)) & (pl.col("p0") >= p_min)
    return P.filter(keep)
