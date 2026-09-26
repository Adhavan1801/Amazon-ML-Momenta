"""Learn Indic-script -> Latin dictionaries from TRAIN ground-truth pairs only.

token  : word-level, from positionally aligned names with equal token counts
segment: address comma-segment level (e.g. state names written in Indic script)
"""
import json
import re
import sys
from collections import Counter, defaultdict

import polars as pl

from normalize import basic, _INDIC_RE

PUNCT = ".,;:()[]{}-_/\\'\"*#<>|!?"


def tok_split(s):
    return [t.strip(PUNCT) for t in s.split() if t.strip(PUNCT)]


def learn(pairs_path, out_path, min_count=3, min_conf=0.4):
    p = pl.read_parquet(pairs_path)
    tok_cnt = defaultdict(Counter)
    for nt, n1 in p.filter(pl.col("nt").str.contains(r"[ऀ-ൿ]")).select("nt", "n1").iter_rows():
        a, b = tok_split(nt), tok_split(n1)
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            if _INDIC_RE.search(x):
                y = basic(y)
                if y:
                    tok_cnt[x][y] += 1
    token = {}
    for x, c in tok_cnt.items():
        y, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_count and n / tot >= min_conf:
            token[x] = y
    seg_cnt = defaultdict(Counter)
    for at, a1 in p.filter(pl.col("at").str.contains(r"[ऀ-ൿ]")).select("at", "a1").iter_rows():
        segs1 = {s.strip() for s in a1.split(",") if s.strip()}
        for s in at.split(","):
            s = s.strip()
            if _INDIC_RE.search(s):
                for y in segs1:
                    seg_cnt[s][y] += 1
    seg_tot = Counter()
    for at in p.filter(pl.col("at").str.contains(r"[ऀ-ൿ]"))["at"]:
        for s in at.split(","):
            s = s.strip()
            if _INDIC_RE.search(s):
                seg_tot[s] += 1
    segment = {}
    for x, c in seg_cnt.items():
        y, n = c.most_common(1)[0]
        if n >= min_count and n / seg_tot[x] >= min_conf:
            segment[x] = y
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"token": token, "segment": segment}, f, ensure_ascii=False)
    print(f"tokens learned: {len(token)}  segments learned: {len(segment)}")
    return token, segment


if __name__ == "__main__":
    import os
    import config as C
    from data import text_pairs
    p = os.path.join(C.WORK, "pairs_text.parquet")
    text_pairs(p)
    learn(p, C.DICT_PATH)
