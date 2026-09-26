"""Single entry point for every pipeline step:  python cli.py <step> [args]

Normally you run src/run_all.py, which calls these steps in order in fresh processes.
Run a single step by hand when you change the code behind it, e.g.
    python cli.py feats train india
"""
import json
import os
import sys

import config as C


def main(argv):
    step, args = argv[0], argv[1:]
    if step == "convert":
        from data import convert
        convert()
    elif step == "dict":
        import learn_dict  # noqa: F401  (module main)
        from data import text_pairs
        from learn_dict import learn
        p = os.path.join(C.WORK, "pairs_text.parquet")
        text_pairs(p)
        learn(p, C.DICT_PATH)
    elif step == "prep":
        from prep import prep_split
        split = args[0]
        prep_split(os.path.join(C.WORK, "data"), split, os.path.join(C.WORK, f"{split}_recs.parquet"))
    elif step == "gtidx":
        from data import gt_idx
        gt_idx()
    elif step == "tokens":
        from block import build_tokens, index_tokens
        split = args[0]
        tok = os.path.join(C.WORK, f"{split}_tok")
        if not os.path.exists(os.path.join(tok, "countries.json")):
            build_tokens(os.path.join(C.WORK, f"{split}_recs.parquet"), tok)
        index_tokens(tok)
        import glob as _g
        for p in _g.glob(os.path.join(tok, "part_*.parquet")):     # raw tokens no longer needed (disk)
            os.remove(p)
    elif step == "block":        # block <split> <country>: chunk loop, then merge (same process)
        from block import generate, finish
        split, country = args
        tok = os.path.join(C.WORK, f"{split}_tok")
        out = os.path.join(C.WORK, f"{split}_pairs")
        os.makedirs(out, exist_ok=True)
        tmp = os.path.join(out, f"_tmp_{country}")
        if not os.path.exists(os.path.join(tmp, "DONE")):
            generate(tok, k_top=40, k_rev=3, q_df_cap=300, out_dir=out, only_chunks=True,
                     only_country=country, s1_chunk=C.BLOCK_S1_CHUNK, j_parts=1 if C.BIG else 3,
                     k_keep_chunk=50 if C.BIG else 30)
            open(os.path.join(tmp, "DONE"), "w").close()
    elif step == "finish":
        from block import finish
        split, country = args
        out = os.path.join(C.WORK, f"{split}_pairs")
        finish(os.path.join(out, f"_tmp_{country}"), os.path.join(out, f"pairs_{country}.parquet"))
    elif step == "prune":
        import shutil
        from stages import stage_prune
        shutil.rmtree(os.path.join(C.WORK, f"{args[0]}_tok", "retr"), ignore_errors=True)   # blocking done (disk)
        stage_prune(C.WORK, args[0], C.ART)
    elif step == "recfeat":
        from stages import stage_recfeat
        stage_recfeat(C.WORK, args[0])
    elif step == "feats":
        from stages import stage_feats
        stage_feats(C.WORK, args[0], only=args[1] if len(args) > 1 else None)
    elif step == "ctx":
        from stages import stage_ctx
        stage_ctx(C.WORK, args[0])
    elif step in ("train1", "pred1", "train2", "eval"):
        import model
        {"train1": model.train_stage1, "train2": model.train_stage2, "eval": model.evaluate_train}.get(
            step, lambda: model.predict_stage1(args[0]))()
    elif step == "predict":
        import model
        model.predict_test(C.OUT)
    elif step == "recall":
        report_recall()
    elif step == "export_borderline":
        import borderline
        borderline.export()
    elif step == "combine":
        import borderline
        borderline.combine()
    else:
        raise SystemExit(f"unknown step {step}")


def report_recall():
    """Share of true train matches that survive blocking and pruning (the recall ceiling)."""
    import polars as pl
    key = pl.col("i").cast(pl.UInt64) * 4294967296 + pl.col("j").cast(pl.UInt64)
    gt = pl.read_parquet(os.path.join(C.WORK, "train_gt_idx.parquet"))
    r = pl.read_parquet(os.path.join(C.WORK, "train_recs.parquet"), columns=["idx", "src", "country_n"]).filter(pl.col("src") == 1)
    res = {}
    for c in sorted(r["country_n"].unique().to_list()):
        g = gt.join(r.filter(pl.col("country_n") == c).select(pl.col("idx").alias("i")), on="i", how="semi").select(key.alias("k"))
        row = {"true_pairs": g.height}
        for stage, path in (("blocking", f"train_pairs/pairs_{c}.parquet"), ("pruned", f"train_pruned/pruned_{c}.parquet")):
            p = os.path.join(C.WORK, path)
            if os.path.exists(p):
                P = pl.read_parquet(p, columns=["i", "j"]).select(key.alias("k"))
                row[stage] = round(g.join(P, on="k", how="semi").height / max(g.height, 1), 4)
        res[c] = row
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
