"""Stage-1 / stage-2 LightGBM, decision rule, evaluation, submission writing."""
import gc
import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl


def pk(df):
    """Add a single u64 pair key (much cheaper joins than on [i, j])."""
    return df.with_columns((pl.col("i").cast(pl.UInt64) * 4294967296 + pl.col("j").cast(pl.UInt64)).alias("k"))

import config as C

W = C.WORK
M = C.ART
DROP = {"i", "j", "y", "p0_rk_j", "fold"}
P1 = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_child_samples=40,
          feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
          verbose=-1, seed=7, num_threads=C.THREADS, max_bin=127)
P2 = dict(P1, num_leaves=63, learning_rate=0.05)
XP = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=10, eta=0.08,
          subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, max_bin=127,
          nthread=C.THREADS, seed=7)
XGB_ROUNDS = 500


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parts(split):
    return sorted(glob.glob(f"{W}/{split}_feats/*.parquet"))


def load_part(split, f, cols=None):
    F = pl.read_parquet(f, columns=cols)
    C = pl.read_parquet(f"{W}/{split}_ctx/{os.path.basename(f)}").drop(["i", "j"])
    C = C.select([c for c in C.columns if c not in F.columns])
    return pl.concat([F, C], how="horizontal")


def fold_of(i_col):
    return (i_col.hash(seed=11) % 2).cast(pl.Int8)


def gt_pairs():
    return pk(pl.read_parquet(f"{W}/train_gt_idx.parquet")).select("k", pl.lit(1, pl.Int8).alias("y"))


# ------------------------------------------------------------------ stage 1
def train_stage1(sample_pct=None, rounds=None):
    """Stage-1 LightGBM (+ optional XGBoost), one model per fold of S1 entities.
    Arrays are built fold by fold, part by part, so peak memory stays near the size of the sample."""
    sample_pct = sample_pct or int(os.environ.get("BER_S1_PCT", "25"))
    rounds = rounds or int(os.environ.get("BER_S1_ROUNDS", "700"))
    gt = gt_pairs()
    cols = None
    X = {0: [], 1: []}
    Y = {0: [], 1: []}
    for f in parts("train"):
        D = load_part("train", f)
        D = D.filter((pl.col("i").hash(seed=5) % 100) < sample_pct)
        D = pk(D).join(gt, on="k", how="left").drop("k").with_columns(pl.col("y").fill_null(0), fold_of(pl.col("i")).alias("fold"))
        if cols is None:
            cols = [c for c in D.columns if c not in DROP]
        for fold in (0, 1):
            S = D.filter(pl.col("fold") == fold)
            X[fold].append(S.select(cols).cast(pl.Float32).to_numpy())
            Y[fold].append(S["y"].to_numpy())
        del D, S
        gc.collect()
    json.dump(cols, open(f"{M}/s1_cols.json", "w"))
    n = sum(len(y) for v in Y.values() for y in v)
    log(f"[s1] sample {sample_pct}% rows={n} features={len(cols)} rounds={rounds}")
    for fold in (0, 1):
        Xf = np.concatenate(X[fold]); X[fold] = None
        Yf = np.concatenate(Y[fold]); Y[fold] = None
        gc.collect()
        ds = lgb.Dataset(Xf, Yf, feature_name=cols, free_raw_data=True)
        m = lgb.train(P1, ds, rounds)
        m.save_model(f"{M}/s1_fold{fold}.txt")
        imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)
        log(f"[s1] fold{fold} trained; top: {[c for _, c in imp[:12]]}")
        del ds, m
        gc.collect()
        if C.XGB:
            import xgboost as xgb
            dm = xgb.QuantileDMatrix(Xf, label=Yf, feature_names=cols, max_bin=XP["max_bin"])
            xm = xgb.train(XP, dm, XGB_ROUNDS)
            xm.save_model(f"{M}/s1x_fold{fold}.json")
            log(f"[s1] xgboost fold{fold} trained")
            del dm, xm
        del Xf, Yf
        gc.collect()


def predict_stage1(split):
    cols = json.load(open(f"{M}/s1_cols.json"))
    ms = [lgb.Booster(model_file=f"{M}/s1_fold{k}.txt") for k in (0, 1)]
    xs = None
    if C.XGB and all(os.path.exists(f"{M}/s1x_fold{k}.json") for k in (0, 1)):
        import xgboost as xgb
        xs = []
        for k in (0, 1):
            b = xgb.Booster()
            b.load_model(f"{M}/s1x_fold{k}.json")
            xs.append(b)

    def pr(k, X):
        p = ms[k].predict(X)
        if xs is not None:
            import xgboost as xgb
            p = (p + xs[k].predict(xgb.DMatrix(X, feature_names=cols))) / 2
        return p
    out_dir = f"{W}/{split}_p1"
    os.makedirs(out_dir, exist_ok=True)
    for f in parts(split):
        if os.path.exists(f"{out_dir}/{os.path.basename(f)}"):
            continue
        D = load_part(split, f)
        X = D.select(cols).cast(pl.Float32).to_numpy()
        if split == "train":   # out-of-fold: fold-0 rows scored by the fold-1 model and vice versa
            fold = D.select(fold_of(pl.col("i")))["i"].to_numpy()
            p = np.empty(len(fold), np.float64)
            for fd in (0, 1):
                m_ = fold == fd
                if m_.any():
                    p[m_] = pr(1 - fd, X[m_])
        else:
            p = (pr(0, X) + pr(1, X)) / 2
        D.select("i", "j").with_columns(pl.Series("p1", p.astype(np.float32))).write_parquet(f"{out_dir}/{os.path.basename(f)}")
        log(f"[s1] predicted {split}/{os.path.basename(f)} rows={D.height}")
        del D, X
        gc.collect()


# ------------------------------------------------------------------ stage 2
S2_KEEP = ["p0", "n_tset", "a_tset", "na_mean", "ns_ratio", "sk_tset", "num_first_eq", "num_jac",
           "a_empty_j", "src_j", "legal_match", "state_match", "bs", "ov_name_s", "ov_an_s", "is_indic_j", "is_dom_j",
           "nf_s1_i", "nf_s1_j", "nf_all_j"]


def s2_raw_cols():
    """Raw features passed to stage 2: the fixed list + (optionally) the top-K stage-1 features by gain."""
    k = int(os.environ.get("BER_S2_TOPK", "0"))
    cols = list(S2_KEEP)
    if k:
        m = lgb.Booster(model_file=f"{M}/s1_fold0.txt")
        names = m.feature_name()
        top = [n for _, n in sorted(zip(m.feature_importance("gain"), names), reverse=True)]
        feat_cols = set(pl.read_parquet_schema(parts("train")[0]).keys())
        for n in top:
            if len(cols) >= len(S2_KEEP) + k:
                break
            if n not in cols and n in feat_cols:
                cols.append(n)
    return cols


def stage2_frame(split, country_files):
    """All pairs of one country: i, j, p1 + p1 context + raw features."""
    base = pl.concat([pl.read_parquet(f"{W}/{split}_p1/{os.path.basename(f)}") for f in country_files])
    rc = json.load(open(f"{M}/s2_raw_cols.json")) if split == "test" and os.path.exists(f"{M}/s2_raw_cols.json") else s2_raw_cols()
    raw = pl.concat([pl.read_parquet(f, columns=rc) for f in country_files]).cast(
        {c: pl.Float32 for c in rc})
    D = pl.concat([base, raw], how="horizontal")
    D = D.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("i").cast(pl.Float32).alias("p1_rk_i"),
        pl.col("p1").rank("ordinal", descending=True).over("j").cast(pl.Float32).alias("p1_rk_j"),
        (pl.col("p1").max().over("i") - pl.col("p1")).alias("p1_gap_i"),
        (pl.col("p1").max().over("j") - pl.col("p1")).alias("p1_gap_j"),
        pl.col("p1").sum().over("i").alias("p1_sum_i"),
        pl.col("p1").sum().over("j").alias("p1_sum_j"),
        (pl.col("p1") > 0.5).sum().over("i").cast(pl.Float32).alias("p1_n05_i"),
        (pl.col("p1") > 0.5).sum().over("j").cast(pl.Float32).alias("p1_n05_j"),
        pl.len().over("i").cast(pl.Float32).alias("n_i"),
    )
    # best competing S1 for this record, and best competitor record for this S1
    D = D.with_columns(
        pl.when(pl.col("p1_rk_j") == 1).then(pl.col("p1").filter(pl.col("p1_rk_j") == 2).max().over("j"))
        .otherwise(pl.col("p1").max().over("j")).fill_null(0).alias("p1_other_j"),
        pl.col("p1").filter(pl.col("p1_rk_i") == 2).max().over("i").fill_null(0).alias("p1_second_i"),
    ).with_columns((pl.col("p1") - pl.col("p1_other_j")).alias("p1_margin_j"))
    return D


def country_files(split):
    fs = parts(split)
    cs = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in fs})
    return {c: [f for f in fs if os.path.basename(f).rsplit("_", 1)[0] == c] for c in cs}


def train_stage2(rounds=500):
    gt = gt_pairs()
    json.dump(s2_raw_cols(), open(f"{M}/s2_raw_cols.json", "w"))
    frames = []
    for c, fs in country_files("train").items():
        D = pk(stage2_frame("train", fs)).join(gt, on="k", how="left").drop("k").with_columns(
            pl.col("y").fill_null(0), fold_of(pl.col("i")).alias("fold"))
        D.write_parquet(f"{W}/train_s2_{c}.parquet")
        frames.append(D.filter(pl.col("fold") == 0))
        del D
        gc.collect()
    S = pl.concat(frames)
    cols = [c for c in S.columns if c not in DROP]
    json.dump(cols, open(f"{M}/s2_cols.json", "w"))
    m = lgb.train(P2, lgb.Dataset(S.select(cols).cast(pl.Float32).to_numpy(), S["y"].to_numpy(), feature_name=cols), rounds)
    m.save_model(f"{M}/s2.txt")
    log(f"[s2] trained on {S.height} rows; top: {[c for _, c in sorted(zip(m.feature_importance('gain'), cols), reverse=True)[:10]]}")


# ------------------------------------------------------------------ decision + eval
def decide(D, pcol, thr, exclusive=True, thr2=None):
    """Pairs kept as matches. exclusive: each S2/S3 record goes to at most one S1 (its best).
    thr applies to each S1's best remaining candidate, thr2 (default = thr) to its further candidates."""
    thr2 = thr if thr2 is None else thr2
    d = D.filter(pl.col(pcol) >= min(thr, thr2))
    if exclusive:
        d = d.sort(pcol, descending=True).unique("j", keep="first")
    if thr2 != thr:
        d = d.with_columns(pl.col(pcol).rank("ordinal", descending=True).over("i").alias("_r"))
        d = d.filter(((pl.col("_r") == 1) & (pl.col(pcol) >= thr)) | ((pl.col("_r") > 1) & (pl.col(pcol) >= thr2)))
    return d.select("i", "j")


def macro_f05(pred, truth, s1_ids):
    """Vectorised per-S1 F0.5 (singletons included), macro-averaged over s1_ids."""
    from metrics import macro_f05 as mf
    return mf(pred, truth, pl.DataFrame({"i": pl.Series(s1_ids, dtype=pl.UInt32)}))


def evaluate_train(thresholds=np.arange(0.3, 0.96, 0.025)):
    m = lgb.Booster(model_file=f"{M}/s2.txt")
    cols = json.load(open(f"{M}/s2_cols.json"))
    recs = pl.read_parquet(f"{W}/train_recs.parquet", columns=["idx", "src"]).filter(pl.col("src") == 1)
    s1_val = recs.filter(fold_of(pl.col("idx")) == 1)["idx"]
    gt = pl.read_parquet(f"{W}/train_gt_idx.parquet")
    gt_val = gt.join(pl.DataFrame({"i": s1_val}), on="i", how="semi")
    V = []
    for f in sorted(glob.glob(f"{W}/train_s2_*.parquet")):
        D = pl.read_parquet(f)
        D = D.with_columns(pl.Series("p2", m.predict(D.select(cols).cast(pl.Float32).to_numpy()).astype(np.float32)))
        V.append(D.select("i", "j", "p1", "p2", "fold", "y"))
    V = pl.concat(V)
    V.write_parquet(f"{W}/val_scores.parquet")        # used by the borderline export / GPU re-rankers
    # the exclusivity step needs competitors from both folds, so decide on all rows then score fold 1
    res = {}
    for pcol in ("p1", "p2"):
        for excl in (True, False):
            for t in thresholds:
                pred = decide(V, pcol, t, excl).join(pl.DataFrame({"i": s1_val}), on="i", how="semi")
                res[(pcol, excl, round(float(t), 3))] = macro_f05(pred, gt_val, s1_val)
    best = max(res, key=res.get)
    alt = (best[0], True, best[2])          # prefer the one-S1-per-record rule on (near) ties
    if not best[1] and res.get(alt, -1) >= res[best] - 1e-4:
        best = alt
    # two thresholds: first match of an S1 vs its extra matches
    pc, ex, t0 = best
    best2 = (res[best], t0, t0)
    for t1 in np.arange(max(0.3, t0 - 0.2), t0 + 0.001, 0.025):
        for t2 in np.arange(t0, min(0.97, t0 + 0.15), 0.025):
            pred = decide(V, pc, t1, ex, t2).join(pl.DataFrame({"i": s1_val}), on="i", how="semi")
            v = macro_f05(pred, gt_val, s1_val)
            if v > best2[0]:
                best2 = (v, round(float(t1), 3), round(float(t2), 3))
    log(f"[eval] two thresholds: F0.5={best2[0]:.4f} first-match thr={best2[1]} extra-match thr={best2[2]}")
    for pcol in ("p1", "p2"):
        for excl in (True, False):
            b = max((k for k in res if k[0] == pcol and k[1] == excl), key=res.get)
            log(f"[eval] {pcol} exclusive={excl}: best F0.5={res[b]:.4f} at thr={b[2]}")
    json.dump({"pcol": best[0], "exclusive": best[1], "thr": best2[1], "thr2": best2[2], "f05": best2[0],
               "f05_single_thr": res[best], "single_thr": best[2]}, open(f"{M}/decision.json", "w"))
    log(f"[eval] BEST single {best} -> {res[best]:.4f} | with two thresholds -> {best2[0]:.4f}")
    return res


def predict_test(out_dir):
    dec = json.load(open(f"{M}/decision.json"))
    m = lgb.Booster(model_file=f"{M}/s2.txt")
    cols = json.load(open(f"{M}/s2_cols.json"))
    recs = pl.read_parquet(f"{W}/test_recs.parquet", columns=["idx", "entity_id", "src"])
    ids = recs["entity_id"]
    sel, cand = [], []
    for c, fs in country_files("test").items():
        D = stage2_frame("test", fs)
        D = D.with_columns(pl.Series("p2", m.predict(D.select(cols).cast(pl.Float32).to_numpy()).astype(np.float32)))
        sel.append(decide(D, dec["pcol"], dec["thr"], dec["exclusive"], dec.get("thr2")))
        cand.append(D.select("i", "j", "p1", "p2"))
        del D
        gc.collect()
    cand = pl.concat(cand)
    cand.write_parquet(f"{W}/test_scores.parquet")    # used by the borderline export / GPU re-rankers
    write_tsv(pl.concat(sel), recs, f"{out_dir}/matching_results.tsv", "matched_entity_ids")
    write_tsv(cand.select("i", "j"), recs, f"{out_dir}/candidate_pairs.tsv", "candidate_entity_ids")


def write_tsv(pairs, recs, path, col):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    eid = recs.select("idx", "entity_id")
    x = (pairs.join(eid.rename({"idx": "j", "entity_id": "je"}), on="j")
         .sort(["i", "je"]).group_by("i", maintain_order=True).agg(pl.col("je").str.join(",").alias(col)))
    s1 = recs.filter(pl.col("src") == 1).select(pl.col("idx").alias("i"), pl.col("entity_id").alias("source1_entity_id"))
    out = s1.join(x, on="i", how="left").with_columns(pl.col(col).fill_null("")).select("source1_entity_id", col)
    out.write_csv(path, separator="\t", quote_style="never")
    log(f"[write] {path}: {out.height} rows, non-empty={(out[col] != '').sum()}")


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "train1":
        train_stage1()
    elif step == "pred1":
        predict_stage1(sys.argv[2])
    elif step == "train2":
        train_stage2()
    elif step == "eval":
        evaluate_train()
    elif step == "test":
        predict_test(C.OUT)
