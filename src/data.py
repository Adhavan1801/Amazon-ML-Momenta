"""Data loading helpers: TSV -> parquet, ground truth -> record-index pairs, text pairs for the Indic dictionary."""
import os

import polars as pl

import config as C


def _read_tsv(path):
    # quote_char=None: addresses may contain quotes; empty cells stay "" (not null)
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def convert():
    """<DATA>/{train,test}/*_sourceN.tsv -> <WORK>/data/{split}/sN.parquet (+ gt.parquet)."""
    for split in ("train", "test"):
        out = os.path.join(C.WORK, "data", split)
        os.makedirs(out, exist_ok=True)
        for s in (1, 2, 3):
            df = _read_tsv(os.path.join(C.DATA, split, f"{split}_source{s}.tsv"))
            df.write_parquet(os.path.join(out, f"s{s}.parquet"))
            print(f"[convert] {split} source{s}: {df.shape}", flush=True)
    gt = _read_tsv(os.path.join(C.DATA, "train", "train_ground_truth.tsv"))
    gt.write_parquet(os.path.join(C.WORK, "data", "train", "gt.parquet"))
    print(f"[convert] ground truth: {gt.shape}", flush=True)


def gt_long():
    gt = pl.read_parquet(os.path.join(C.WORK, "data", "train", "gt.parquet"))
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "s1", "matched_entity_ids": "t"}))


def text_pairs(path):
    """(target name/address, S1 name/address) for every ground-truth pair -> used by learn_dict."""
    d = os.path.join(C.WORK, "data", "train")
    s1 = pl.read_parquet(f"{d}/s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                                                   pl.col("business_address").alias("a1"))
    T = pl.concat([pl.read_parquet(f"{d}/s{s}.parquet") for s in (2, 3)]).select(
        pl.col("entity_id").alias("t"), pl.col("business_name").alias("nt"), pl.col("business_address").alias("at"))
    gt_long().join(T, on="t").join(s1, on="s1").select("t", "s1", "nt", "n1", "at", "a1").write_parquet(path)


def gt_idx():
    """Ground truth as (i, j) record indices of the prepared train records."""
    recs = pl.read_parquet(os.path.join(C.WORK, "train_recs.parquet"), columns=["idx", "entity_id"])
    g = (gt_long().join(recs.rename({"idx": "i", "entity_id": "s1"}), on="s1")
         .join(recs.rename({"idx": "j", "entity_id": "t"}), on="t").select("i", "j"))
    g.write_parquet(os.path.join(C.WORK, "train_gt_idx.parquet"))
    print(f"[gt_idx] {g.height} true pairs", flush=True)
