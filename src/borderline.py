"""Stages 4.2-4.4 glue: export borderline pairs for GPU re-rankers, then combine their scores.

  python cli.py export_borderline          -> <WORK>/borderline/{train,test}.parquet
      train.parquet: held-out (fold-1) pairs with lo <= p2 < hi, with raw texts and label y
      test.parquet : test pairs with lo <= p2 < hi, with raw texts
  (run sagemaker/gpu/cross_encoder.py and/or sagemaker/gpu/qwen_judge.py on a GPU; they write
   <WORK>/borderline/{ce,llm}_{train,test}.parquet with a column `score`)
  python cli.py combine                     -> tunes the blend on train, writes <OUT>/matching_results.tsv
"""
import json
import os

import numpy as np
import polars as pl

import config as C
from metrics import macro_f05

BAND = (0.15, 0.95)       # p2 range treated as "borderline"


def _texts(split):
    return pl.read_parquet(os.path.join(C.WORK, f"{split}_recs.parquet"),
                           columns=["idx", "entity_id", "country", "name_raw", "addr_raw"])


def _attach(P, split):
    R = _texts(split)
    return (P.join(R.rename({c: c + "_1" for c in R.columns if c != "idx"}), left_on="i", right_on="idx")
             .join(R.rename({c: c + "_2" for c in R.columns if c != "idx"}), left_on="j", right_on="idx"))


def export(lo=BAND[0], hi=BAND[1]):
    out = os.path.join(C.WORK, "borderline")
    os.makedirs(out, exist_ok=True)
    V = pl.read_parquet(os.path.join(C.WORK, "val_scores.parquet")).filter(pl.col("fold") == 1)
    Bt = V.filter((pl.col("p2") >= lo) & (pl.col("p2") < hi))
    _attach(Bt, "train").write_parquet(os.path.join(out, "train.parquet"))
    T = pl.read_parquet(os.path.join(C.WORK, "test_scores.parquet"))
    Bs = T.filter((pl.col("p2") >= lo) & (pl.col("p2") < hi))
    _attach(Bs, "test").write_parquet(os.path.join(out, "test.parquet"))
    print(f"[borderline] train pairs={Bt.height} (pos={int(Bt['y'].sum())}) of {V.height}; "
          f"test pairs={Bs.height} of {T.height}; band={lo}-{hi}", flush=True)


def _blend(df, has_ce, has_llm, w):
    s = w[0] * pl.col("p2")
    if has_ce:
        s = s + w[1] * pl.col("ce").fill_null(pl.col("p2"))
    if has_llm:
        s = s + w[2] * pl.col("llm").fill_null(pl.col("p2"))
    return df.with_columns((s / sum(w[k] for k, h in ((0, True), (1, has_ce), (2, has_llm)) if h)).alias("pf"))


def _decide(D, thr):
    d = D.filter(pl.col("pf") >= thr).sort("pf", descending=True).unique("j", keep="first")
    return d.select("i", "j")


def combine():
    b = os.path.join(C.WORK, "borderline")
    extra = {}
    for name in ("ce", "llm"):
        for split in ("train", "test"):
            p = os.path.join(b, f"{name}_{split}.parquet")
            if os.path.exists(p):
                extra.setdefault(name, {})[split] = pl.read_parquet(p, columns=["i", "j", "score"]).rename({"score": name})
    if not extra:
        raise SystemExit("no ce_*/llm_* score files found in " + b)
    has_ce, has_llm = "ce" in extra, "llm" in extra

    # ---- tune weights + threshold on held-out fold-1 (all pairs; borderline ones get the extra scores)
    V = pl.read_parquet(os.path.join(C.WORK, "val_scores.parquet"))
    for name, d in extra.items():
        V = V.join(d["train"], on=["i", "j"], how="left")
    recs = pl.read_parquet(os.path.join(C.WORK, "train_recs.parquet"), columns=["idx", "src"]).filter(pl.col("src") == 1)
    s1 = recs.filter((pl.col("idx").hash(seed=11) % 2) == 1).select(pl.col("idx").alias("i"))
    gt = pl.read_parquet(os.path.join(C.WORK, "train_gt_idx.parquet")).join(s1, on="i", how="semi")
    base_thr = json.load(open(os.path.join(C.ART, "decision.json")))["thr"]
    grid_w = [(1, a, c) for a in ([0, 0.5, 1, 2, 4] if has_ce else [0]) for c in ([0, 0.5, 1, 2] if has_llm else [0])]
    best = (-1, None, None)
    for w in grid_w:
        Vb = _blend(V, has_ce, has_llm, w)
        for thr in np.arange(0.4, 0.91, 0.025):
            f = macro_f05(_decide(Vb, thr).join(s1, on="i", how="semi"), gt, s1)
            if f > best[0]:
                best = (f, w, float(thr))
    f0 = macro_f05(_decide(V.with_columns(pl.col("p2").alias("pf")), base_thr).join(s1, on="i", how="semi"), gt, s1)
    print(f"[combine] held-out F0.5 without re-ranker={f0:.4f}  with={best[0]:.4f}  weights={best[1]} thr={best[2]:.3f}", flush=True)
    json.dump({"f05_base": f0, "f05": best[0], "weights": best[1], "thr": best[2]},
              open(os.path.join(C.ART, "combine.json"), "w"))

    # ---- apply to test
    T = pl.read_parquet(os.path.join(C.WORK, "test_scores.parquet"))
    for name, d in extra.items():
        if "test" in d:
            T = T.join(d["test"], on=["i", "j"], how="left")
    T = _blend(T, has_ce and "test" in extra["ce"], has_llm and "test" in extra.get("llm", {}), best[1])
    from model import write_tsv
    R = pl.read_parquet(os.path.join(C.WORK, "test_recs.parquet"), columns=["idx", "entity_id", "src"])
    write_tsv(_decide(T, best[2]), R, os.path.join(C.OUT, "matching_results.tsv"), "matched_entity_ids")
    write_tsv(T.select("i", "j"), R, os.path.join(C.OUT, "candidate_pairs.tsv"), "candidate_entity_ids")
