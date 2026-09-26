"""Normalise every record of a split into one parquet table."""
import os
import sys
import time
from multiprocessing import Pool

import polars as pl

import config as C
import normalize as N

DICT_PATH = C.DICT_PATH


def _init():
    N.load_dicts(DICT_PATH)


def _work(rows):
    out = []
    for name, addr, country in rows:
        n = N.norm_name(name or "")
        a = N.norm_addr(addr or "", country or "")
        out.append((*n, *a))
    return out


def prep_split(data_dir, split, out_path, workers=None, chunk=10000, block=None):
    workers = workers or C.THREADS
    block = block or C.PREP_BLOCK
    t0 = time.time()
    parts = []
    cols = ["name_core", "name_full", "legal", "name_nums", "is_dom", "is_indic",
            "addr_core", "state", "addr_nums"]
    tmp_dir = out_path + ".parts"
    os.makedirs(tmp_dir, exist_ok=True)
    with Pool(workers, initializer=_init, maxtasksperchild=50) as pool:
        for s in (1, 2, 3):
            src = pl.read_parquet(os.path.join(data_dir, split, f"s{s}.parquet")).fill_null("")
            for b in range(0, src.height, block):
                d = src.slice(b, block)
                rows = list(zip(d["business_name"].to_list(), d["business_address"].to_list(),
                                d["country"].to_list()))
                res = []
                for r in pool.imap(_work, [rows[i:i + chunk] for i in range(0, len(rows), chunk)]):
                    res.extend(r)
                feat = pl.DataFrame(res, schema=cols, orient="row")
                part = pl.concat([d.select("entity_id", pl.lit(s).cast(pl.Int8).alias("src"), "country",
                                           pl.col("business_name").alias("name_raw"),
                                           pl.col("business_address").alias("addr_raw")), feat],
                                 how="horizontal")
                pp = os.path.join(tmp_dir, f"{s}_{b:09d}.parquet")
                part.write_parquet(pp)
                parts.append(pp)
                del rows, res, feat, part
                print(f"  {split} s{s}: {b + d.height}/{src.height}  {time.time() - t0:.0f}s", flush=True)
            del src
    out = pl.concat([pl.read_parquet(p) for p in parts]).with_columns(
        pl.col("country").str.to_lowercase().str.strip_chars().alias("country_n"),
        pl.col("is_dom").cast(pl.Int8), pl.col("is_indic").cast(pl.Int8),
    ).with_row_index("idx")
    out.write_parquet(out_path)
    for p in parts:
        os.remove(p)
    print(f"{split}: {out.height} records in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    split = sys.argv[1]
    prep_split(os.path.join(C.WORK, "data"), split, os.path.join(C.WORK, f"{split}_recs.parquet"))
