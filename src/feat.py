"""Vectorised pair features (no country-specific inputs)."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from normalize import skeleton


def add_record_cols(recs: pl.DataFrame) -> pl.DataFrame:
    """Derived per-record columns used by the features."""
    w = recs.select("idx", pl.col("name_core").str.split(" ").alias("w")).explode("w")
    vocab = w.select("w").unique().with_columns(
        pl.col("w").map_elements(skeleton, return_dtype=pl.Utf8).alias("sk"))
    sk = (w.join(vocab, on="w", how="left").group_by("idx", maintain_order=True)
          .agg(pl.col("sk").str.join(" ").alias("name_sk")))
    recs = recs.join(sk, on="idx", how="left").with_columns(
        pl.col("name_sk").fill_null(""),
        pl.col("name_core").str.replace_all(" ", "").alias("name_ns"),
        pl.col("addr_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("nums_l"),
        pl.col("name_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("nnums_l"),
        pl.col("legal").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("legal_l"),
        pl.col("name_core").str.split(" ").list.len().alias("n_ntok"),
        pl.col("addr_core").str.split(" ").list.len().alias("a_ntok"),
    ).sort("idx")
    return recs


_SCORERS_NAME = [("n_ratio", fuzz.ratio), ("n_tset", fuzz.token_set_ratio),
                 ("n_tsort", fuzz.token_sort_ratio), ("n_partial", fuzz.partial_ratio),
                 ("n_jw", JaroWinkler.normalized_similarity)]
_SCORERS_ADDR = [("a_ratio", fuzz.ratio), ("a_tset", fuzz.token_set_ratio),
                 ("a_tsort", fuzz.token_sort_ratio), ("a_partial", fuzz.partial_ratio)]


def pair_features(P: pl.DataFrame, recs: pl.DataFrame, pos=None) -> pl.DataFrame:
    """P: i, j + blocking columns. recs: output of add_record_cols; pos maps idx -> row of recs."""
    I = P["i"].to_numpy()
    J = P["j"].to_numpy()
    if pos is not None:
        I, J = pos[I], pos[J]
    cols = {}

    def g(col, idx):
        return recs[col].gather(idx)

    for fld, scorers in (("name_core", _SCORERS_NAME), ("addr_core", _SCORERS_ADDR)):
        a = g(fld, I).to_list()
        b = g(fld, J).to_list()
        for name, sc in scorers:
            cols[name] = cpdist(a, b, scorer=sc, workers=-1, dtype=np.float32)
    a, b = g("name_ns", I).to_list(), g("name_ns", J).to_list()
    cols["ns_ratio"] = cpdist(a, b, scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    cols["ns_partial"] = cpdist(a, b, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32)
    cols["ns_lev"] = cpdist(a, b, scorer=Levenshtein.distance, workers=-1, dtype=np.float32)
    na = np.array([len(x) for x in a], np.float32)
    nb = np.array([len(x) for x in b], np.float32)
    cols["ns_len_i"], cols["ns_len_j"] = na, nb
    cols["ns_contains"] = np.array([(len(x) > 3 and len(y) > 3 and (x in y or y in x)) for x, y in zip(a, b)], np.int8)
    a, b = g("name_sk", I).to_list(), g("name_sk", J).to_list()
    cols["sk_ratio"] = cpdist(a, b, scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    cols["sk_tset"] = cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    a, b = g("name_full", I).to_list(), g("name_full", J).to_list()
    cols["nf_tset"] = cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    F = pl.DataFrame(cols)

    X = pl.DataFrame({
        "nums_i": g("nums_l", I), "nums_j": g("nums_l", J),
        "nn_i": g("nnums_l", I), "nn_j": g("nnums_l", J),
        "lg_i": g("legal_l", I), "lg_j": g("legal_l", J),
        "st_i": g("state", I), "st_j": g("state", J),
    })
    X = X.select(
        pl.col("nums_i").list.len().alias("nnum_i"), pl.col("nums_j").list.len().alias("nnum_j"),
        pl.col("nums_i").list.set_intersection("nums_j").list.len().alias("num_inter"),
        (pl.col("nums_i").list.first() == pl.col("nums_j").list.first()).cast(pl.Int8).fill_null(-1).alias("num_first_eq"),
        pl.col("nums_j").list.first().is_in(pl.col("nums_i")).cast(pl.Int8).fill_null(-1).alias("num_j1_in_i"),
        pl.when((pl.col("nn_i").list.len() > 0) & (pl.col("nn_j").list.len() > 0))
          .then((pl.col("nn_i").list.set_intersection("nn_j").list.len() > 0).cast(pl.Int8) * 2 - 1)
          .otherwise(0).alias("name_num_match"),
        pl.when((pl.col("lg_i").list.len() > 0) & (pl.col("lg_j").list.len() > 0))
          .then((pl.col("lg_i").list.set_intersection("lg_j").list.len() > 0).cast(pl.Int8) * 2 - 1)
          .otherwise(0).alias("legal_match"),
        (pl.col("lg_i").list.len() > 0).cast(pl.Int8).alias("legal_i"),
        (pl.col("lg_j").list.len() > 0).cast(pl.Int8).alias("legal_j"),
        pl.when((pl.col("st_i") != "") & (pl.col("st_j") != ""))
          .then((pl.col("st_i") == pl.col("st_j")).cast(pl.Int8) * 2 - 1).otherwise(0).alias("state_match"),
    ).with_columns(
        (pl.col("num_inter") / pl.max_horizontal(pl.col("nnum_i"), pl.col("nnum_j"), pl.lit(1))).alias("num_jac"))

    R = pl.DataFrame({
        "src_j": g("src", J), "is_dom_j": g("is_dom", J), "is_indic_j": g("is_indic", J),
        "n_ntok_i": g("n_ntok", I), "n_ntok_j": g("n_ntok", J),
        "a_ntok_i": g("a_ntok", I), "a_ntok_j": g("a_ntok", J),
        "a_empty_j": (g("addr_core", J) == "").cast(pl.Int8),
        "nf_s1_i": g("nf_s1", I), "nf_s1_j": g("nf_s1", J), "nf_all_j": g("nf_all", J),
    })
    B = P
    out = pl.concat([B, F, X, R], how="horizontal")
    out = out.with_columns((pl.col("n_tset") * pl.col("a_tset") / 1e4).alias("na_prod"),
                           ((pl.col("n_tset") + pl.col("a_tset")) / 2).alias("na_mean"))
    return out


CTX_COLS = ["na_mean", "n_tset", "a_tset", "sk_tset", "ns_ratio", "p0"]


def add_context(F: pl.DataFrame, cols, prefix="") -> pl.DataFrame:
    exprs = []
    for c in cols:
        exprs += [
            pl.col(c).rank("min", descending=True).over("i").cast(pl.Float32).alias(f"{prefix}{c}_rk_i"),
            (pl.col(c).max().over("i") - pl.col(c)).cast(pl.Float32).alias(f"{prefix}{c}_gap_i"),
            pl.col(c).rank("min", descending=True).over("j").cast(pl.Float32).alias(f"{prefix}{c}_rk_j"),
            (pl.col(c).max().over("j") - pl.col(c)).cast(pl.Float32).alias(f"{prefix}{c}_gap_j"),
        ]
    exprs += [pl.len().over("i").cast(pl.Float32).alias(f"{prefix}ncand_i"), pl.len().over("j").cast(pl.Float32).alias(f"{prefix}ncand_j")]
    return F.with_columns(exprs)
