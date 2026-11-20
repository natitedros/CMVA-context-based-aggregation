"""Step 4: build the fine-tuning data for the moral agents.

    python src/prepare_agent_data.py

Output (on Drive, under data/agent_data/):
  {value}.csv    columns: Q, A   answers that annotators judged as respecting that value
  control.csv    columns: Q, A   random answers, not filtered for any value (style control)
  stats.json     row counts

Selection rule for a value: MIC *train* split, Rule of Thumb tagged with the value,
and annotators mostly agree the answer follows it (mean A_agrees >= 1.5 on the 0/1/2
scale, i.e. "agrees", not merely neutral). Only the train split is used, so none of
the 500 evaluation questions (from the test split) are ever trained on.

The control set has the same size and comes from the same pool of train answers,
with no value filter. Its agent shows how much reward comes from MIC's style alone.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, MIC_PATH, SEED, VALUES
from prepare_data import COL_A, COL_AGREE, COL_MORAL, COL_Q, assign_split, has_value, load_mic

OUT_DIR = f"{DATA_DIR}/agent_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000, help="examples per agent")
    ap.add_argument("--min_agree", type=float, default=1.5)
    ap.add_argument("--min_words", type=int, default=3, help="drop near-empty answers")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_mic(MIC_PATH)
    df["_split"] = assign_split(df, np.random.default_rng(SEED))
    train = df[df["_split"] == "train"].copy()
    train = train[train[COL_A].astype(str).str.split().str.len() >= args.min_words]
    print(f"{len(train):,} train-split annotations with answers of >= {args.min_words} words")

    stats = {}
    for v in VALUES:
        sub = train[train[COL_MORAL].apply(has_value, value=v)]
        pairs = sub.groupby([COL_Q, COL_A], as_index=False)[COL_AGREE].mean()
        good = pairs[pairs[COL_AGREE] >= args.min_agree]
        out = good.sample(n=min(args.n_train, len(good)), random_state=SEED)[[COL_Q, COL_A]]
        out.columns = ["Q", "A"]
        out.to_csv(f"{OUT_DIR}/{v}.csv", index=False)
        stats[v] = {"available": len(good), "used": len(out)}
        print(f"{v:10s} {len(good):6,} qualifying pairs -> {len(out):,} used")

    pool = train[[COL_Q, COL_A]].drop_duplicates()
    control = pool.sample(n=min(args.n_train, len(pool)), random_state=SEED)
    control.columns = ["Q", "A"]
    control.to_csv(f"{OUT_DIR}/control.csv", index=False)
    stats["control"] = {"available": len(pool), "used": len(control)}
    print(f"control    {len(pool):6,} unique pairs -> {len(control):,} used")

    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
