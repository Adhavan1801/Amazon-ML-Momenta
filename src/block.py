"""Scalable blocking by weighted rare-token overlap (inverted index joins).

Each record emits typed tokens:
  n:<name word>      k:<skeleton of name word>     p:<first 5 chars of name w/o spaces>
  a:<address word>   d:<address number>            h:<number>_<skeleton(next word)[:3]>
Token weight = idf inside the country group; tokens with df above a cap are
ignored for retrieval (they still count in features). For every S1 record we
sum the weights of shared tokens per candidate, keep the top-K, and also keep
the top-R S1 records of every S2/S3 record (reverse view).
The per-type weight sums are returned as features.
"""
import math
import time

import numpy as np
import polars as pl

from normalize import skeleton

TYPES = ["n", "k", "p", "a", "d", "h", "x", "b", "f", "m", "c", "s"]


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


TYP_CODE = {t: k for k, t in enumerate(TYPES)}


def _tok_chunk(recs: pl.DataFrame) -> pl.DataFrame:
    """recs: idx, name_core, addr_core -> (idx u32, tid u64, typ u8), unique."""
    name_w = (recs.select("idx", pl.col("name_core").str.split(" ").alias("w"))
              .explode("w").filter(pl.col("w").str.len_chars() >= 2))
    vocab = name_w.select("w").unique().with_columns(
        pl.col("w").map_elements(skeleton, return_dtype=pl.Utf8).alias("sk"))
    name_w = name_w.join(vocab, on="w", how="left")
    parts = [
        name_w.select("idx", pl.col("w").alias("t"), pl.lit(0, pl.UInt8).alias("typ")),
        name_w.filter(pl.col("sk").str.len_chars() >= 3).select("idx", pl.col("sk").alias("t"), pl.lit(1, pl.UInt8).alias("typ")),
        recs.select("idx", pl.col("name_core").str.replace_all(" ", "").alias("t"))
            .filter(pl.col("t").str.len_chars() >= 5)
            .select("idx", pl.col("t").str.slice(0, 5), pl.lit(2, pl.UInt8).alias("typ")),
    ]
    addr_w = (recs.select("idx", pl.col("addr_core").str.split(" ").alias("w"))
              .with_columns(pl.col("w").list.shift(-1).alias("nxt"))
              .explode(["w", "nxt"]).filter(pl.col("w") != ""))
    is_num = pl.col("w").str.contains(r"^\d+$")
    parts.append(addr_w.filter(~is_num & (pl.col("w").str.len_chars() >= 3)).select("idx", pl.col("w").alias("t"), pl.lit(3, pl.UInt8).alias("typ")))
    parts.append(addr_w.filter(is_num).select("idx", pl.col("w").alias("t"), pl.lit(4, pl.UInt8).alias("typ")))
    hv = addr_w.filter(is_num & pl.col("nxt").is_not_null() & ~pl.col("nxt").str.contains(r"^\d+$"))
    nv = hv.select("nxt").unique().with_columns(
        pl.col("nxt").map_elements(lambda x: skeleton(x)[:3], return_dtype=pl.Utf8).alias("nk"))
    hv = hv.join(nv, on="nxt", how="left")
    parts.append(hv.select("idx", (pl.col("w") + "_" + pl.col("nk")).alias("t"), pl.lit(5, pl.UInt8).alias("typ")))
    # exact full core name
    parts.append(recs.select("idx", pl.col("name_core").str.replace_all(" ", "").alias("t"))
                 .filter(pl.col("t").str.len_chars() >= 3).select("idx", "t", pl.lit(8, pl.UInt8).alias("typ")))
    # address bigrams (consecutive words, numbers included)
    parts.append(addr_w.filter(pl.col("nxt").is_not_null() & (pl.col("nxt") != ""))
                 .select("idx", (pl.col("w") + "_" + pl.col("nxt")).alias("t"), pl.lit(7, pl.UInt8).alias("typ")))
    # name word x address word combinations
    nw = (recs.select("idx", pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.head(5).alias("nw"))
          .explode("nw").drop_nulls())
    aw = (recs.select("idx", pl.col("addr_core").str.split(" ").list.eval(pl.element().filter((pl.element().str.len_chars() >= 4) & ~pl.element().str.contains(r"^\d+$"))).list.head(8).alias("aw"))
          .explode("aw").drop_nulls())
    x = nw.join(aw, on="idx")
    parts.append(x.select("idx", (pl.col("nw") + "|" + pl.col("aw")).alias("t"), pl.lit(6, pl.UInt8).alias("typ")))
    # full name x address combos: survive truncated addresses and common names
    ns = recs.select("idx", pl.col("name_core").str.replace_all(" ", "").alias("ns")).filter(pl.col("ns").str.len_chars() >= 3)
    anum = addr_w.filter(is_num).select("idx", pl.col("w").alias("num")).unique()
    aword = addr_w.filter(~is_num & (pl.col("w").str.len_chars() >= 4)).select("idx", pl.col("w").alias("aw")).unique()
    parts.append(ns.join(anum, on="idx").select("idx", (pl.col("ns") + "|" + pl.col("num")).alias("t"), pl.lit(9, pl.UInt8).alias("typ")))
    parts.append(ns.join(aword, on="idx").select("idx", (pl.col("ns") + "|" + pl.col("aw")).alias("t"), pl.lit(10, pl.UInt8).alias("typ")))
    sk = (name_w.filter(pl.col("sk").str.len_chars() >= 2).group_by("idx", maintain_order=True)
          .agg(pl.col("sk").str.join("")).rename({"sk": "sks"}))
    parts.append(sk.join(anum, on="idx").select("idx", (pl.col("sks") + "|" + pl.col("num")).alias("t"), pl.lit(11, pl.UInt8).alias("typ")))
    T = pl.concat(parts)
    T = T.select(pl.col("idx").cast(pl.UInt32), (pl.col("t") + pl.col("typ").cast(pl.Utf8)).hash(seed=1).alias("tid"), "typ")
    return T.unique()


def build_tokens(recs_path, out_dir, chunk=250_000):
    """Tokenise all records of a split into part files: idx, tid, typ, src, cc."""
    import gc, json, os
    os.makedirs(out_dir, exist_ok=True)
    lf = pl.scan_parquet(recs_path)
    meta = lf.select("idx", "src", "country_n").collect()
    ccode = {c: k for k, c in enumerate(meta["country_n"].unique().sort().to_list())}
    meta = meta.select(pl.col("idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8),
                       pl.col("country_n").replace_strict(ccode, return_dtype=pl.UInt8).alias("cc"))
    n = meta.height
    for k, s in enumerate(range(0, n, chunk)):
        r = lf.slice(s, chunk).select("idx", "name_core", "addr_core").collect()
        T = _tok_chunk(r).join(meta.slice(s, chunk), on="idx")
        T.write_parquet(os.path.join(out_dir, f"part_{k:04d}.parquet"))
        del r, T
        gc.collect()
    json.dump(ccode, open(os.path.join(out_dir, "countries.json"), "w"))
    _log(f"[tokens] {out_dir}: {n} records, countries={ccode}")
    return ccode


def index_tokens(tok_dir, df_cap=2000, n_buckets=32):
    """Hash-bucketed: idf per country, per-record weight totals, retrieval files. Streams."""
    import gc, glob as G, json, os, shutil
    import numpy as np
    parts = sorted(G.glob(os.path.join(tok_dir, "part_*.parquet")))
    bdir = os.path.join(tok_dir, "buckets")
    os.makedirs(bdir, exist_ok=True)
    metas = []
    for k, p in enumerate(parts):
        if os.path.exists(os.path.join(bdir, "done")):
            break
        t = pl.read_parquet(p)
        metas.append(t.select("idx", "cc").unique())
        t = t.with_columns((pl.col("tid") % n_buckets).cast(pl.UInt8).alias("bk"))
        for (bk,), g in t.partition_by("bk", as_dict=True).items():
            g.drop("bk").write_parquet(os.path.join(bdir, f"b{bk:02d}_p{k:04d}.parquet"))
        del t
        gc.collect()
    open(os.path.join(bdir, "done"), "w").close()
    meta = pl.concat(metas) if metas else pl.concat([pl.read_parquet(p, columns=["idx", "cc"]).unique() for p in parts])
    n_idx = int(meta["idx"].max()) + 1
    n_rec = dict(meta.group_by("cc").len().iter_rows())
    del meta
    tot = np.zeros((n_idx, len(TYPES)), np.float32)
    rdir = os.path.join(tok_dir, "retr")
    os.makedirs(rdir, exist_ok=True)
    logn = pl.DataFrame({"cc": list(n_rec.keys()), "logn": [math.log(v) for v in n_rec.values()]},
                        schema={"cc": pl.UInt8, "logn": pl.Float32})
    for bk in range(n_buckets):
        t = pl.read_parquet(os.path.join(bdir, f"b{bk:02d}_p*.parquet"))
        df = t.group_by(["cc", "tid"]).agg(pl.len().cast(pl.UInt32).alias("df"))
        t = t.join(df, on=["cc", "tid"]).join(logn, on="cc").with_columns(
            (pl.col("logn") - pl.col("df").cast(pl.Float32).log()).cast(pl.Float32).alias("w")).drop("logn")
        agg = t.group_by(["idx", "typ"]).agg(pl.col("w").sum())
        np.add.at(tot, (agg["idx"].to_numpy(), agg["typ"].to_numpy().astype(np.int64)), agg["w"].to_numpy())
        t.filter(pl.col("df") <= df_cap).drop("df").write_parquet(os.path.join(rdir, f"part_b{bk:02d}.parquet"))
        del t, df, agg
        gc.collect()
    shutil.rmtree(bdir)
    T = pl.DataFrame({"idx": np.arange(n_idx, dtype=np.uint32), **{t: tot[:, k] for k, t in enumerate(TYPES)}})
    T.write_parquet(os.path.join(tok_dir, "tot.parquet"))
    _log(f"[index] {tok_dir}: records={n_idx}")


GROUPS = {"name": [0, 1, 2, 8], "addr": [3, 7], "num": [4, 5], "cross": [6], "combo": [9, 10, 11]}


def generate(tok_dir, k_top=40, k_rev=3, s1_chunk=5000, query_idx=None, out_dir=None, q_df_cap=None, k_keep_chunk=50, only_chunks=False, only_country=None, j_parts=1):
    """Candidate pairs + blocking features via a sorted inverted index (numpy)."""
    import gc, json, os
    import numpy as np
    glob = os.path.join(tok_dir, "retr", "part_*.parquet")
    ccode = json.load(open(os.path.join(tok_dir, "countries.json")))
    grp_of = np.zeros(len(TYPES), np.int8)
    gnames = list(GROUPS)
    for g, (name, ts) in enumerate(GROUPS.items()):
        grp_of[ts] = g
    out = []
    for cname, cc in sorted(ccode.items(), key=lambda x: x[1]):
        if only_country and cname != only_country:
            continue
        t0 = time.time()
        lf = pl.scan_parquet(glob).filter(pl.col("cc") == cc)
        n_rec_c = int(lf.select(pl.col("idx").n_unique()).collect(engine="streaming").item())
        q_lf = lf.filter(pl.col("src") == 1)
        if query_idx is not None:
            q_lf = q_lf.filter(pl.col("idx").is_in(query_idx.implode()))
        Q = q_lf.select("idx", "tid", "typ", "w").collect(engine="streaming").sort("idx")
        if q_df_cap:
            Q = Q.filter(pl.col("w") >= float(np.log(n_rec_c / q_df_cap)) - 0.5)
        q_idx = Q["idx"].to_numpy()
        q_tid = Q["tid"].to_numpy()
        q_grp = grp_of[Q["typ"].to_numpy()]
        q_w = Q["w"].to_numpy()
        del Q
        gc.collect()
        uq, starts = np.unique(q_idx, return_index=True)
        bounds = np.append(starts, len(q_idx))
        tmp = os.path.join(out_dir or "/tmp", f"_tmp_{cname}")
        os.makedirs(tmp, exist_ok=True)
        n_parts = 0
        # candidate records are split into j-partitions: a pair's score only uses its own j's tokens,
        # so partitions are independent and the merge in finish() is exact.
        for jp in range(j_parts):
            T = (lf.filter((pl.col("src") != 1) & ((pl.col("idx") % j_parts) == jp))
                 .select("tid", "idx").collect(engine="streaming").sort("tid"))
            tt = T["tid"].to_numpy()
            tj = T["idx"].to_numpy()
            del T
            gc.collect()
            lo_all = np.searchsorted(tt, q_tid, "left")
            cnt_all = np.searchsorted(tt, q_tid, "right") - lo_all
            _log(f"[block] {cname} part {jp + 1}/{j_parts}: queries={len(uq)} q_tokens={len(q_idx)} "
                 f"t_postings={len(tt)} rows={int(cnt_all.sum())} {time.time() - t0:.0f}s")
            for c in range(0, len(uq), s1_chunk):
                a, b = bounds[c], bounds[min(c + s1_chunk, len(uq))]
                cnt = cnt_all[a:b]
                tot = int(cnt.sum())
                if tot == 0:
                    continue
                rows = np.repeat(np.arange(a, b), cnt)
                offs = np.repeat(lo_all[a:b] - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                pos = offs + np.arange(tot)
                w = q_w[rows]
                g = q_grp[rows]
                D = pl.DataFrame({"i": q_idx[rows], "j": tj[pos], "w": w,
                                  **{f"bs_{n}": np.where(g == k, w, np.float32(0)) for k, n in enumerate(gnames)}})
                agg = D.group_by(["i", "j"]).agg(pl.col("w").sum().alias("bs"), pl.len().cast(pl.UInt16).alias("b_ntok"),
                                                 *[pl.col(f"bs_{n}").sum() for n in gnames])
                agg = agg.with_columns(pl.col("bs").rank("ordinal", descending=True).over("i").cast(pl.UInt16).alias("b_rank_i"))
                agg.filter(pl.col("b_rank_i") <= k_keep_chunk).write_parquet(os.path.join(tmp, f"c{n_parts:05d}.parquet"))
                n_parts += 1
                del D, agg, rows, offs, pos, w, g
            del tt, tj, lo_all, cnt_all
            gc.collect()
        del q_idx, q_tid, q_grp, q_w
        gc.collect()
        _log(f"[block] {cname}: chunks written ({n_parts}) {time.time() - t0:.0f}s")
        if only_chunks:
            continue
        finish(tmp, os.path.join(out_dir, f"pairs_{cname}.parquet"), k_top, k_rev)
    if out_dir:
        return None, None
    P = pl.concat(out)
    Tt = pl.read_parquet(os.path.join(tok_dir, "tot.parquet"))
    return P, Tt


def finish(tmp, out_path, k_top=40, k_rev=3, i_batch=150_000):
    """Merge chunk files: keep each S1's top-k_top plus each record's top-k_rev S1.

    Memory-bounded: the per-record top-k_rev table is a streaming group-by over (j, i, bs);
    the per-S1 ranking is done batch by batch over S1 index ranges (all of an S1's rows are in its batch)."""
    import shutil, os
    t0 = time.time()
    src = os.path.join(tmp, "c*.parquet")
    Rj = (pl.scan_parquet(src).select("i", "j", "bs")
          .group_by("j").agg(pl.col("i").top_k_by("bs", k_rev).alias("i"))
          .with_columns(pl.int_ranges(1, pl.col("i").list.len() + 1, dtype=pl.UInt16).alias("b_rank_j"))
          .explode(["i", "b_rank_j"]).collect(engine="streaming"))
    Rj = Rj.with_columns((pl.col("i").cast(pl.UInt64) * 4294967296 + pl.col("j").cast(pl.UInt64)).alias("k")).select("k", "i", "b_rank_j")
    ids = pl.scan_parquet(src).select(pl.col("i").unique()).collect(engine="streaming")["i"].sort()
    parts = []
    for s in range(0, len(ids), i_batch):
        lo, hi = ids[s], ids[min(s + i_batch, len(ids)) - 1]
        B = pl.scan_parquet(src).filter((pl.col("i") >= lo) & (pl.col("i") <= hi)).drop("b_rank_i").collect()
        B = B.with_columns(pl.col("bs").rank("ordinal", descending=True).over("i").cast(pl.UInt16).alias("b_rank_i"),
                           (pl.col("i").cast(pl.UInt64) * 4294967296 + pl.col("j").cast(pl.UInt64)).alias("k"))
        rj = Rj.filter((pl.col("i") >= lo) & (pl.col("i") <= hi)).select("k", "b_rank_j")
        B = B.join(rj, on="k", how="left").filter((pl.col("b_rank_i") <= k_top) | pl.col("b_rank_j").is_not_null())
        B = B.with_columns(pl.col("b_rank_i").clip(upper_bound=k_top + 1), pl.col("b_rank_j").fill_null(k_rev + 1)).drop("k")
        p = os.path.join(tmp, f"merged_{s:09d}.parquet")
        B.write_parquet(p)
        parts.append(p)
        del B
    del Rj
    P = pl.concat([pl.read_parquet(p) for p in parts])
    P.write_parquet(out_path)
    shutil.rmtree(tmp)
    _log(f"[finish] {out_path}: pairs={P.height} ({P.height / P['i'].n_unique():.1f}/S1) {time.time() - t0:.0f}s")


def recall(P: pl.DataFrame, recs: pl.DataFrame, gt_pairs: pl.DataFrame, query_idx=None):
    """gt_pairs: columns i, j (record idx)."""
    g = gt_pairs if query_idx is None else gt_pairs.filter(pl.col("i").is_in(query_idx))
    hit = g.join(P.select("i", "j"), on=["i", "j"], how="semi").height
    return hit / max(g.height, 1), g.height
