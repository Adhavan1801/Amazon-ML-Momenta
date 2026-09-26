"""Stage 4.4 — Qwen2.5-7B-Instruct (Apache-2.0, 7B) as a yes/no judge for the hardest pairs (GPU).

Input : <borderline_dir>/train.parquet, test.parquet  (optionally ce_*.parquet to pick "still uncertain")
Output: <borderline_dir>/llm_train.parquet, llm_test.parquet — columns i, j, score (= P("Yes"))

    python qwen_judge.py --dir <WORK>/borderline [--lo 0.3 --hi 0.85] [--max-train 20000]

Uses vLLM when installed (fast), otherwise Hugging Face transformers. Needs a 24 GB GPU (e.g. ml.g5.xlarge).
Only zero-shot prompting — no external data or lookups; the model only sees the two records.
"""
import argparse
import math
import os

import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SYSTEM = ("You are an expert at entity resolution. Decide whether two business records from different "
          "databases refer to the same real-world business. Records can contain typos, abbreviations, "
          "legal-suffix differences (Pvt Ltd / Private Limited, LLC, SAS...), transliterations "
          "(e.g. Hindi or Tamil script vs English), trade names, missing or partial addresses. "
          "Different branches of a chain at different addresses are DIFFERENT businesses. "
          "Answer with exactly one word: Yes or No.")


def prompt(r):
    return (f"Record A:\n  name: {r['name_raw_1']}\n  address: {r['addr_raw_1'] or '(missing)'}\n  country: {r['country_1']}\n"
            f"Record B:\n  name: {r['name_raw_2']}\n  address: {r['addr_raw_2'] or '(missing)'}\n  country: {r['country_2']}\n"
            "Same business? Answer Yes or No.")


def select(df, d, split, lo, hi, cap):
    ce = os.path.join(d, f"ce_{split}.parquet")
    if os.path.exists(ce):      # "still uncertain" after the cross-encoder
        df = df.join(pl.read_parquet(ce).rename({"score": "ce"}), on=["i", "j"], how="left")
        df = df.filter(((pl.col("p2") + pl.col("ce").fill_null(pl.col("p2"))) / 2).is_between(lo, hi))
    else:
        df = df.filter(pl.col("p2").is_between(lo, hi))
    if cap and df.height > cap:
        df = df.sample(cap, seed=0)
    return df


def run_vllm(texts, tok):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=768, gpu_memory_utilization=0.9)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    outs = llm.generate(texts, sp)
    scores = []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        y = max([v.logprob for v in lp.values() if (v.decoded_token or "").strip().lower() == "yes"], default=-30)
        n = max([v.logprob for v in lp.values() if (v.decoded_token or "").strip().lower() == "no"], default=-30)
        scores.append(math.exp(y) / (math.exp(y) + math.exp(n)))
    return np.array(scores, np.float32)


def run_hf(texts, tok, bs=16):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    tok.padding_side = "left"
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), bs):
            enc = tok(texts[s:s + bs], return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits[:, -1, :].float()
            out.append(torch.softmax(logits[:, [yes, no]], dim=-1)[:, 0].cpu().numpy())
            if s % (bs * 50) == 0:
                print(f"  {s}/{len(texts)}", flush=True)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--max-train", type=int, default=20000, help="train pairs to score (for blend tuning)")
    ap.add_argument("--max-test", type=int, default=0, help="0 = all selected test pairs")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        import vllm  # noqa: F401
        runner = run_vllm
    except ImportError:
        runner = run_hf
    for split, cap in (("train", a.max_train), ("test", a.max_test)):
        df = select(pl.read_parquet(os.path.join(a.dir, f"{split}.parquet")), a.dir, split, a.lo, a.hi, cap)
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(r)}],
                                         tokenize=False, add_generation_prompt=True) for r in df.iter_rows(named=True)]
        print(f"{split}: scoring {len(texts)} pairs with {MODEL} ({runner.__name__})", flush=True)
        s = runner(texts, tok) if texts else np.zeros(0, np.float32)
        df.select("i", "j").with_columns(pl.Series("score", s)).write_parquet(os.path.join(a.dir, f"llm_{split}.parquet"))
        if split == "train" and len(s):
            from sklearn_free_auc import auc
            print(f"train AUC of Qwen on selected pairs: {auc(df['y'].to_numpy(), s):.4f}", flush=True)


if __name__ == "__main__":
    main()
