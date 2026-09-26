"""Run the whole pipeline end to end:  data -> blocking -> matching -> output.

    python src/run_all.py --data <dataset dir> --work <scratch dir> --out <output dir>

Every step runs in its own Python process (keeps peak memory low) and writes a marker in
<work>/_done/, so re-running the command resumes where it stopped.

    --list            print the steps and exit
    --from STEP       start at STEP (earlier steps must be done already); later markers are cleared
    --only STEP ...   run only these steps (markers ignored for them)
    --validate PATH   run the official utils/validate_submission.py at the end
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def steps(work):
    """Ordered list of (name, cli-args). Country lists are read from the token index when available."""
    def countries(split):
        p = os.path.join(work, f"{split}_tok", "countries.json")
        return sorted(json.load(open(p))) if os.path.exists(p) else None

    S = [("convert", ["convert"]), ("dict", ["dict"]),
         ("prep_train", ["prep", "train"]), ("prep_test", ["prep", "test"]), ("gtidx", ["gtidx"]),
         ("tokens_train", ["tokens", "train"]), ("tokens_test", ["tokens", "test"])]
    for split in ("train", "test"):
        cs = countries(split)
        if cs is None:
            S.append((f"block_{split}", None))          # expanded once the tokens exist
            continue
        for c in cs:
            S += [(f"block_{split}_{c}", ["block", split, c]), (f"finish_{split}_{c}", ["finish", split, c])]
    S += [("prune_train", ["prune", "train"]), ("prune_test", ["prune", "test"]), ("recall", ["recall"]),
          ("recfeat_train", ["recfeat", "train"]), ("recfeat_test", ["recfeat", "test"])]
    for split in ("train", "test"):
        cs = countries(split)
        if cs is None:
            S.append((f"feats_{split}", None))
            continue
        S += [(f"feats_{split}_{c}", ["feats", split, c]) for c in cs]
    S += [("ctx_train", ["ctx", "train"]), ("ctx_test", ["ctx", "test"]),
          ("train1", ["train1"]), ("pred1_train", ["pred1", "train"]), ("train2", ["train2"]), ("eval", ["eval"]),
          ("pred1_test", ["pred1", "test"]), ("predict", ["predict"]), ("export_borderline", ["export_borderline"])]
# "combine" is run on demand after the GPU re-rankers:  run_all.py ... --only combine
    return S


def run(name, args, env, done_dir, force=False):
    marker = os.path.join(done_dir, name)
    if os.path.exists(marker) and not force:
        print(f"[skip] {name} (done)", flush=True)
        return
    print(f"\n===== {name}: python cli.py {' '.join(args)}", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "cli.py"), *args], env=env, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"step {name} failed (exit {r.returncode}). Fix and re-run the same command to resume.")
    open(marker, "w").close()
    print(f"===== {name} done in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/ and test/ .tsv files")
    ap.add_argument("--work", required=True, help="scratch folder (~40 GB free)")
    ap.add_argument("--out", required=True, help="output folder for the two .tsv files")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--mem-gb", type=float, default=6, help="RAM available; >=24 uses bigger/faster chunks")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--validate", help="path to utils/validate_submission.py")
    ap.add_argument("--no-xgb", dest="xgb", action="store_false", help="disable the XGBoost part of the stage-1 ensemble")
    ap.add_argument("--s1-pct", type=int, default=40, help="%% of S1 entities used to train stage 1")
    ap.add_argument("--s1-rounds", type=int, default=900, help="boosting rounds for stage-1 LightGBM")
    ap.add_argument("--s2-topk", type=int, default=30, help="also give stage 2 the top-K stage-1 features")
    ap.add_argument("--prune-k", type=int, default=20, help="candidates kept per S1 after pruning")
    ap.add_argument("--prune-krev", type=int, default=3, help="S1s kept per S2/S3 record after pruning")
    ap.add_argument("--prune-pmin", type=float, default=0.002, help="minimum pruner probability")
    ap.add_argument("--prune-load-k", type=int, default=40, help="blocking candidates per S1 fed to the pruner")
    a = ap.parse_args()

    work, out = os.path.abspath(a.work), os.path.abspath(a.out)
    env = dict(os.environ, BER_DATA=os.path.abspath(a.data), BER_WORK=work, BER_OUT=out,
               BER_ART=os.path.join(work, "artifacts"), BER_THREADS=str(a.threads), BER_MEM_GB=str(a.mem_gb),
               POLARS_MAX_THREADS=str(a.threads), BER_XGB="1" if a.xgb else "0",
               BER_S1_PCT=str(a.s1_pct), BER_S1_ROUNDS=str(a.s1_rounds), BER_S2_TOPK=str(a.s2_topk),
               BER_PRUNE_K=str(a.prune_k), BER_PRUNE_KREV=str(a.prune_krev), BER_PRUNE_PMIN=str(a.prune_pmin),
               **({"BER_PRUNE_LOAD_K": str(a.prune_load_k)} if a.prune_load_k else {}))
    done = os.path.join(work, "_done")
    os.makedirs(done, exist_ok=True)

    if a.list:
        for n, _ in steps(work):
            print(("[done] " if os.path.exists(os.path.join(done, n)) else "       ") + n)
        return
    if a.only:
        extra = {"combine": ["combine"], "export_borderline": ["export_borderline"]}
        known = dict((n, args) for n, args in steps(work) if args)
        for n in a.only:
            args = known.get(n) or extra.get(n)
            if args:
                run(n, args, env, done, force=True)
            else:
                print(f"[warn] unknown step {n}")
        return

    started = a.start is None
    while True:                       # re-expand after tokens exist (country-specific steps)
        progressed = False
        for n, args in steps(work):
            if not started:
                if n == a.start:
                    started = True
                    for m, _ in steps(work):     # clear markers from here on
                        if m == n or started_after(m, n, work):
                            p = os.path.join(done, m)
                            if os.path.exists(p):
                                os.remove(p)
                else:
                    continue
            if args is None:
                progressed = True
                break
            run(n, args, env, done)
        if not progressed:
            break
    if a.validate:
        subprocess.run([sys.executable, a.validate, "--matching", os.path.join(out, "matching_results.tsv"),
                        "--candidate", os.path.join(out, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(os.path.abspath(a.data), "test")])
    print("\nALL DONE ->", out)


def started_after(m, n, work):
    names = [x for x, _ in steps(work)]
    return m in names and n in names and names.index(m) > names.index(n)


if __name__ == "__main__":
    main()
