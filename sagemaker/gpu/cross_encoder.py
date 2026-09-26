"""Stage 4.3 — fine-tune a multilingual cross-encoder on borderline pairs and score them (GPU).

Input : <borderline_dir>/train.parquet (labelled held-out borderline pairs), test.parquet
Output: <borderline_dir>/ce_train.parquet (out-of-fold scores), ce_test.parquet  — columns i, j, score

    python cross_encoder.py --dir <WORK>/borderline [--model xlm-roberta-base] [--epochs 2]

Models (<= 8B, MIT): xlm-roberta-base (default), microsoft/mdeberta-v3-base, xlm-roberta-large.
Cross-fitting: 2 folds over the train pairs -> every train pair is scored by a model that never saw it
(needed to tune the blend fairly); test = average of the two fold models.
"""
import argparse
import os
import time

import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def rec_text(df, side):
    return (df[f"name_raw_{side}"].fill_null("") + " | " + df[f"addr_raw_{side}"].fill_null("") + " | "
            + df[f"country_{side}"].fill_null("")).to_list()


def batches(a, b, y, tok, bs, max_len, shuffle):
    idx = np.random.permutation(len(a)) if shuffle else np.arange(len(a))
    for s in range(0, len(a), bs):
        k = idx[s:s + bs]
        enc = tok([a[i] for i in k], [b[i] for i in k], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        yield enc, (torch.tensor(y[k], dtype=torch.float32) if y is not None else None)


def train_one(a, b, y, args, dev):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(a) + args.bs - 1) // args.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t = time.time()
    for ep in range(args.epochs):
        for n, (enc, yb) in enumerate(batches(a, b, y, tok, args.bs, args.max_len, True)):
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = lossf(model(**enc).logits.squeeze(-1), yb.to(dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            if n % 200 == 0:
                print(f"  epoch {ep} step {n} loss {loss.item():.4f} {time.time() - t:.0f}s", flush=True)
    return tok, model


@torch.no_grad()
def predict(tok, model, a, b, args, dev):
    model.eval()
    out = []
    for enc, _ in batches(a, b, None, tok, args.bs * 2, args.max_len, False):
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=400000, help="cap on train pairs (speed)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    tr = pl.read_parquet(os.path.join(args.dir, "train.parquet"))
    if tr.height > args.max_train:
        tr = tr.sample(args.max_train, seed=0)
    te = pl.read_parquet(os.path.join(args.dir, "test.parquet"))
    fold = (tr["i"].hash(seed=21) % 2).to_numpy()
    a_tr, b_tr, y_tr = rec_text(tr, 1), rec_text(tr, 2), tr["y"].to_numpy().astype(np.float32)
    a_te, b_te = rec_text(te, 1), rec_text(te, 2)
    oof = np.zeros(tr.height, np.float32)
    test_scores = np.zeros(te.height, np.float32)
    for f in (0, 1):
        trn, val = np.where(fold != f)[0], np.where(fold == f)[0]
        print(f"fold {f}: train {len(trn)} / score {len(val)}", flush=True)
        tok, model = train_one([a_tr[i] for i in trn], [b_tr[i] for i in trn], y_tr[trn], args, dev)
        oof[val] = predict(tok, model, [a_tr[i] for i in val], [b_tr[i] for i in val], args, dev)
        test_scores += predict(tok, model, a_te, b_te, args, dev) / 2
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None
    from sklearn_free_auc import auc
    print(f"OOF AUC on borderline train pairs: {auc(y_tr, oof):.4f}", flush=True)
    tr.select("i", "j").with_columns(pl.Series("score", oof)).write_parquet(os.path.join(args.dir, "ce_train.parquet"))
    te.select("i", "j").with_columns(pl.Series("score", test_scores)).write_parquet(os.path.join(args.dir, "ce_test.parquet"))
    print("wrote ce_train.parquet / ce_test.parquet", flush=True)


if __name__ == "__main__":
    main()
