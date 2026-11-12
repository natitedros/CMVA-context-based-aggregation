"""Step 1: build per-value reward-model datasets and an eval prompt set from MIC.

Output (on Drive, under data/splits/):
  {value}_{train,val,test}.csv   columns: text, label   (one set per moral value)
  eval_prompts.csv               column:  Q             (held-out questions for generation)
  stats.json                     row counts and label balance, for the report

Labelling rule: a MIC row says whether a chatbot answer A agrees with a Rule of
Thumb tied to one or more moral foundations. For each value, we keep rows whose
RoT is tagged with that foundation, and label the (Q, A) pair:
  answer violates the RoT  -> 0
  answer agrees / neutral  -> 1   (mirrors the paper's "negative vs neutral/positive")
Use --neutral drop to discard neutral rows instead.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from config import MIC_PATH, SPLITS_DIR, VALUES, SEED

# Column names in MIC.csv. If your copy differs, change them here.
COL_Q, COL_A, COL_MORAL, COL_AGREE, COL_SPLIT = "Q", "A", "moral", "A_agrees", "split"


def load_mic(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")
    missing = {COL_Q, COL_A, COL_MORAL, COL_AGREE} - set(df.columns)
    if missing:
        raise SystemExit(f"MIC.csv is missing {missing}. Update the COL_* names at the top of this file.")
    df = df.dropna(subset=[COL_Q, COL_A, COL_MORAL, COL_AGREE])
    # A few rows in MIC.csv are malformed (text shifted into the label columns).
    # Keep only rows whose A_agrees is a real label: 0, 1 or 2.
    agree = pd.to_numeric(df[COL_AGREE], errors="coerce")
    bad = ~agree.isin([0, 1, 2])
    if bad.any():
        print(f"Dropping {int(bad.sum())} malformed rows (non-numeric A_agrees).")
    df = df[~bad].copy()
    df[COL_AGREE] = agree[~bad].astype(int)
    print("A_agrees values:", df[COL_AGREE].value_counts().to_dict())
    if COL_SPLIT in df.columns:
        print("split values:", df[COL_SPLIT].value_counts().to_dict())
    print("Sample 'moral' values:", df[COL_MORAL].head(5).tolist())
    return df


def has_value(moral_field, value):
    # MIC tags look like "care-harm" or "care-harm|fairness-cheating"
    tokens = re.split(r"[|,;/]", str(moral_field).lower())
    return any(t.strip().startswith(value) for t in tokens)


def assign_split(df, rng):
    if COL_SPLIT in df.columns:
        mapping = {"train": "train", "dev": "val", "val": "val", "valid": "val", "validation": "val", "test": "test"}
        s = df[COL_SPLIT].astype(str).str.strip().str.lower().map(mapping)
        if s.notna().mean() > 0.99:
            if s.isna().any():
                print(f"Ignoring {int(s.isna().sum())} rows with an unrecognised split value.")
            print("Using MIC's own train/dev/test split.")
            return s
    # Fallback: split by question so the same Q never appears in two splits
    print("No usable split column; splitting by question 80/10/10.")
    qs = df[COL_Q].unique()
    rng.shuffle(qs)
    n = len(qs)
    bucket = {q: "train" for q in qs[: int(0.8 * n)]}
    bucket.update({q: "val" for q in qs[int(0.8 * n): int(0.9 * n)]})
    bucket.update({q: "test" for q in qs[int(0.9 * n):]})
    return df[COL_Q].map(bucket)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral", choices=["pos", "drop"], default="pos")
    ap.add_argument("--max_train", type=int, default=10000)
    ap.add_argument("--max_eval", type=int, default=1000)
    ap.add_argument("--n_prompts", type=int, default=500)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df = load_mic(MIC_PATH)
    df["_split"] = assign_split(df, rng)

    agree = df[COL_AGREE].astype(int)
    if args.neutral == "drop":
        df = df[agree != 1]
        agree = df[COL_AGREE].astype(int)
    df["label"] = (agree != 0).astype(int)

    stats = {}
    caps = {"train": args.max_train, "val": args.max_eval, "test": args.max_eval}
    for v in VALUES:
        sub = df[df[COL_MORAL].apply(has_value, value=v)]
        # One label per (Q, A): majority vote across that pair's RoTs
        sub = (sub.groupby([COL_Q, COL_A, "_split"], as_index=False)["label"].mean())
        sub["label"] = (sub["label"] >= 0.5).astype(int)
        sub["text"] = "Q: " + sub[COL_Q] + "\nA: " + sub[COL_A]
        stats[v] = {}
        for split, cap in caps.items():
            part = sub[sub["_split"] == split]
            part = part.sample(n=min(cap, len(part)), random_state=SEED)
            part[["text", "label"]].to_csv(f"{SPLITS_DIR}/{v}_{split}.csv", index=False)
            stats[v][split] = {"rows": len(part), "pos_rate": round(float(part["label"].mean()), 3) if len(part) else None}
        print(f"{v:10s}", stats[v])

    # Held-out questions the agents will answer during evaluation
    test_qs = pd.Series(df.loc[df["_split"] == "test", COL_Q].unique())
    prompts = test_qs.sample(n=min(args.n_prompts, len(test_qs)), random_state=SEED)
    prompts.to_frame("Q").to_csv(f"{SPLITS_DIR}/eval_prompts.csv", index=False)
    stats["eval_prompts"] = len(prompts)

    with open(f"{SPLITS_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved splits and {len(prompts)} eval prompts to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
