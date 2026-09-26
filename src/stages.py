"""Chunked, memory-bounded pipeline stages (per country, per S1-index chunk).

stage prune   : blocking pairs -> pruner features -> p0 -> short list      (pruned_<c>.parquet)
stage feats   : short list -> full pair features (parts)                   (feats/<c>_<k>.parquet)
stage ctx     : global i/j context features over all parts                 (ctx/<c>_<k>.parquet)
"""
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl

import config as C


def pk(df):
    """Add a single u64 pair key (much cheaper joins than on [i, j])."""
    return df.with_columns((pl.col("i").cast(pl.UInt64) * 4294967296 + pl.col("j").cast(pl.UInt64)).alias("k"))

from prune import block_features, prune_feature_cols, train_pruner, apply_pruner
from feat import add_record_cols, pair_features, add_context, CTX_COLS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def maxrss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except ImportError:          # Windows
        return float("nan")


def countries(pairs_dir):
    return sorted(os.path.basename(p)[6:-8] for p in glob.glob(os.path.join(pairs_dir, "pairs_*.parquet")))


def load_pairs(pairs_dir, c, k_i=30, k_j=3):
    return (pl.read_parquet(os.path.join(pairs_dir, f"pairs_{c}.parquet"))
            .filter((pl.col("b_rank_i") <= k_i) | (pl.col("b_rank_j") <= k_j)))


def i_chunks(P, n_chunks):
    ids = P["i"].unique().sort()
    edges = [ids[int(len(ids) * k / n_chunks)] for k in range(n_chunks)] + [ids[-1] + 1]
    for a, b in zip(edges[:-1], edges[1:]):
        yield P.filter((pl.col("i") >= a) & (pl.col("i") < b))


def block_feats_chunk(Pc, tot, jagg):
    F = block_features(Pc, tot)
    # j-side context must be global -> replace chunk-local values
    F = F.drop(["bs_gap_j", "ov_name_s_gap_j", "ov_an_s_gap_j", "nc_j"]).join(jagg, on="j", how="left").with_columns(
        (pl.col("bs_max_j") - pl.col("bs")).cast(pl.Float32).alias("bs_gap_j")).drop("bs_max_j")
    return F


def label(P, gt):
    g = pk(gt).select("k", pl.lit(1, pl.Int8).alias("y"))
    return pk(P).join(g, on="k", how="left").with_columns(pl.col("y").fill_null(0)).drop("k")


# ------------------------------------------------------------------ prune
PRUNE_K = int(os.environ.get("BER_PRUNE_K", "15"))          # keep top-K candidates per S1
PRUNE_KREV = int(os.environ.get("BER_PRUNE_KREV", "2"))     # ... plus each record's top-K S1
PRUNE_PMIN = float(os.environ.get("BER_PRUNE_PMIN", "0.003"))
PRUNE_LOAD_K = int(os.environ.get("BER_PRUNE_LOAD_K", "40" if C.BIG else "20"))   # blocking candidates read per S1
def stage_prune(work, split, model_dir, train_frac=0.12, n_chunks=None):
    n_chunks = n_chunks or C.PRUNE_CHUNKS
    import lightgbm as lgb
    pairs_dir = f"{work}/{split}_pairs"
    tot = pl.read_parquet(f"{work}/{split}_tok/tot.parquet")
    out_dir = f"{work}/{split}_pruned"
    os.makedirs(out_dir, exist_ok=True)
    gt = pl.read_parquet(f"{work}/train_gt_idx.parquet") if split == "train" else None
    mpath = f"{model_dir}/pruner.txt"
    if split == "train" and not os.path.exists(mpath):
        samples = []
        for c in countries(pairs_dir):
            P = load_pairs(pairs_dir, c)
            jagg = P.group_by("j").agg(pl.col("bs").max().alias("bs_max_j"), pl.len().cast(pl.UInt16).alias("nc_j"))
            S = P.filter((pl.col("i").hash(seed=3) % 1000) < int(train_frac * 1000))
            samples.append(label(block_feats_chunk(S, tot, jagg), gt))
            del P, S, jagg
            gc.collect()
        S = pl.concat(samples, how="diagonal").fill_null(0)
        log(f"[prune] training pruner on {S.height} pairs, pos={S['y'].sum()}")
        model, cols = train_pruner(S)
        model.save_model(mpath)
        json.dump(cols, open(f"{model_dir}/pruner_cols.json", "w"))
        del S, samples
        gc.collect()
    model = lgb.Booster(model_file=mpath)
    cols = json.load(open(f"{model_dir}/pruner_cols.json"))
    for c in countries(pairs_dir):
        if os.path.exists(f"{out_dir}/pruned_{c}.parquet"):
            continue
        P = load_pairs(pairs_dir, c, k_i=PRUNE_LOAD_K)
        jagg = P.group_by("j").agg(pl.col("bs").max().alias("bs_max_j"), pl.len().cast(pl.UInt16).alias("nc_j"))
        kept = []
        for Pc in i_chunks(P, n_chunks):
            F = block_feats_chunk(Pc, tot, jagg)
            for col in cols:
                if col not in F.columns:
                    F = F.with_columns(pl.lit(0.0).alias(col))
            p0 = model.predict(F.select(cols).to_numpy())
            F = F.with_columns(pl.Series("p0", p0.astype(np.float32)))
            F = F.with_columns(pl.col("p0").rank("ordinal", descending=True).over("i").cast(pl.UInt16).alias("p0_rk_i"))
            kept.append(F.filter((pl.col("p0_rk_i") <= max(20, PRUNE_K)) & (pl.col("p0") >= PRUNE_PMIN)))
            del F, Pc
            gc.collect()
        del P
        K = pl.concat(kept)
        del kept
        R = pk(K.select("i", "j", "p0")).with_columns(
            pl.col("p0").rank("ordinal", descending=True).over("j").cast(pl.UInt16).alias("p0_rk_j")).select("k", "p0_rk_j")
        K = pk(K).join(R, on="k").drop("k")
        del R
        K = K.filter(((pl.col("p0_rk_i") <= PRUNE_K) | (pl.col("p0_rk_j") <= PRUNE_KREV)) & (pl.col("p0") >= PRUNE_PMIN))
        K.write_parquet(f"{out_dir}/pruned_{c}.parquet")
        msg = f"[prune] {split}/{c}: kept {K.height} pairs ({K.height / K['i'].n_unique():.1f}/S1)"
        if gt is not None:
            n_all = gt.join(pl.read_parquet(f"{work}/train_recs.parquet", columns=["idx", "country_n"])
                            .filter(pl.col("country_n") == c).select(pl.col("idx").alias("i")), on="i", how="semi").height
            hit = pk(K.select("i", "j")).join(pk(gt).select("k"), on="k", how="semi").height
            msg += f" recall={hit / n_all:.4f}"
        log(msg)
        del K
        gc.collect()


# ------------------------------------------------------------------ features
def stage_recfeat(work, split, chunk=1_000_000):
    """Per-record derived columns (skeletons, number lists...) computed once, in slices."""
    cols = ["idx", "src", "country_n", "name_core", "name_full", "legal", "name_nums",
            "is_dom", "is_indic", "addr_core", "state", "addr_nums"]
    out_dir = f"{work}/{split}_recfeat"
    os.makedirs(out_dir, exist_ok=True)
    lf = pl.scan_parquet(f"{work}/{split}_recs.parquet").select(cols)
    n = lf.select(pl.len()).collect().item()
    # how common is each exact core name: among S1 (reference) and among all records, per country
    nf = lf.select("country_n", "name_core", "src").collect()
    f_s1 = nf.filter(pl.col("src") == 1).group_by("country_n", "name_core").len().rename({"len": "nf_s1"})
    f_all = nf.group_by("country_n", "name_core").len().rename({"len": "nf_all"})
    del nf
    for k, s in enumerate(range(0, n, chunk)):
        r = lf.slice(s, chunk).collect()
        r = r.join(f_s1, on=["country_n", "name_core"], how="left").join(f_all, on=["country_n", "name_core"], how="left")
        r = add_record_cols(r.with_columns(pl.col("nf_s1").fill_null(0).cast(pl.Float32), pl.col("nf_all").cast(pl.Float32)))
        r.write_parquet(f"{out_dir}/part_{k:03d}.parquet")
        del r
        gc.collect()
    log(f"[recfeat] {split}: {n} records")


FEAT_REC_COLS = ["idx", "src", "name_core", "name_full", "name_ns", "name_sk", "addr_core", "state",
                 "nums_l", "nnums_l", "legal_l", "is_dom", "is_indic", "n_ntok", "a_ntok", "nf_s1", "nf_all"]


def stage_feats(work, split, n_chunks=None, only=None):
    n_chunks = n_chunks or C.FEAT_CHUNKS
    out_dir = f"{work}/{split}_feats"
    os.makedirs(out_dir, exist_ok=True)
    n_total = pl.scan_parquet(f"{work}/{split}_recs.parquet").select(pl.len()).collect().item()
    for c in countries(f"{work}/{split}_pairs"):
        if only and c != only:
            continue
        kp = f"{work}/{split}_pruned/pruned_{c}.parquet"
        ij = pl.read_parquet(kp, columns=["i", "j"])
        ids = ij["i"].unique().sort()
        del ij
        src_lf = pl.scan_parquet(f"{work}/{split}_recfeat/part_*.parquet").select(FEAT_REC_COLS)
        edges = [ids[int(len(ids) * k / n_chunks)] for k in range(n_chunks)] + [ids[-1] + 1]
        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if os.path.exists(f"{out_dir}/{c}_{k:03d}.parquet"):
                continue
            t = time.time()
            Kc = pl.scan_parquet(kp).filter((pl.col("i") >= lo) & (pl.col("i") < hi)).collect()
            need = pl.DataFrame({"idx": pl.concat([Kc["i"], Kc["j"]]).unique()})
            recs = src_lf.join(need.lazy(), on="idx", how="semi").collect(engine="streaming")
            pos = np.full(n_total, -1, np.int64)
            pos[recs["idx"].to_numpy()] = np.arange(recs.height)
            F = pair_features(Kc, recs, pos)
            F.write_parquet(f"{out_dir}/{c}_{k:03d}.parquet")
            log(f"[feats] {split}/{c} chunk {k}: {F.height} pairs {time.time() - t:.0f}s maxrss={maxrss_gb():.2f}GB")
            del F, Kc, recs, need, pos
            gc.collect()


def stage_ctx(work, split):
    fdir = f"{work}/{split}_feats"
    out_dir = f"{work}/{split}_ctx"
    os.makedirs(out_dir, exist_ok=True)
    for c in countries(f"{work}/{split}_pairs"):
        files = sorted(glob.glob(f"{fdir}/{c}_*.parquet"))
        S = pl.concat([pl.read_parquet(f, columns=["i", "j"] + CTX_COLS).with_columns(pl.lit(k, pl.UInt16).alias("part"))
                       for k, f in enumerate(files)])
        S = add_context(S, CTX_COLS).drop(CTX_COLS)
        for k, f in enumerate(files):
            S.filter(pl.col("part") == k).drop("part").write_parquet(f"{out_dir}/{os.path.basename(f)}")
        log(f"[ctx] {split}/{c}: {S.height} rows")
        del S
        gc.collect()


if __name__ == "__main__":
    stage, split = sys.argv[1], sys.argv[2]
    W = C.WORK
    M = C.ART
    if stage == "prune":
        stage_prune(W, split, M)
    elif stage == "recfeat":
        stage_recfeat(W, split)
    elif stage == "feats":
        stage_feats(W, split, only=sys.argv[3] if len(sys.argv) > 3 else None)
    elif stage == "ctx":
        stage_ctx(W, split)
