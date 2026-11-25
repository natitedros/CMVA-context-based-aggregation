"""Step 7: data for the contextual aggregator.

    python src/prepare_aggregator_data.py

Builds, on Drive under data/aggregator/:
  train.csv     Q, profile (5 columns), reference   MIC train questions the agents never saw
  test.csv      Q, profile (5 columns), reference   the 500 held-out evaluation questions
  answers/{arm}_{value}.csv                         each agent's answers to the training questions

A sample is one MIC annotation whose answer annotators judged as AGREEING with its Rule of
Thumb (A_agrees = 2). The profile is that rule's moral tags, and the reference (the training
target, and the ROUGE reference at test time) is that answer. One annotation per question.

Training questions exclude the 2,000 used to train the agents, so the agents answer them as
they would answer new questions. Generation is resumable: each agent's file is skipped if it
already exists. The test-set agent answers already exist from generate_and_score.py.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from aggregator_common import AGG_DIR_NAME, profile_from_moral
from config import BASE_LM, DATA_DIR, MIC_PATH, MODELS_DIR, SEED, SPLITS_DIR, VALUES
from prepare_data import COL_A, COL_AGREE, COL_MORAL, COL_Q, assign_split, load_mic

AGG_DIR = f"{DATA_DIR}/{AGG_DIR_NAME}"


def one_agreeing_row_per_question(df, questions=None):
    rows = df[df[COL_AGREE] == 2].copy()
    rows["profile"] = rows[COL_MORAL].apply(profile_from_moral)
    rows = rows[rows["profile"].apply(sum) > 0]                      # at least one of our five values
    rows = rows[rows[COL_A].astype(str).str.split().str.len() >= 3]  # no near-empty references
    if questions is not None:
        rows = rows[rows[COL_Q].isin(questions)]
    rows = rows.groupby(COL_Q, group_keys=False).sample(n=1, random_state=SEED)
    out = pd.DataFrame({"Q": rows[COL_Q].values, "reference": rows[COL_A].values})
    out[[f"prof_{v}" for v in VALUES]] = np.array(rows["profile"].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(f"{AGG_DIR}/answers", exist_ok=True)

    df = load_mic(MIC_PATH)
    df["_split"] = assign_split(df, np.random.default_rng(SEED))

    # Test set: the same 500 held-out questions every model has been scored on
    eval_qs = pd.read_csv(f"{SPLITS_DIR}/eval_prompts.csv")["Q"].tolist()
    test = one_agreeing_row_per_question(df[df["_split"] == "test"], eval_qs)
    test.to_csv(f"{AGG_DIR}/test.csv", index=False)

    # Training set: train-split questions not used to train the agents
    distill_qs = set(pd.read_csv(f"{DATA_DIR}/distill/candidates_scored.csv", usecols=["Q"])["Q"])
    pool = one_agreeing_row_per_question(df[(df["_split"] == "train") & ~df[COL_Q].isin(distill_qs)])
    train = pool.sample(n=min(args.n_train, len(pool)), random_state=SEED).reset_index(drop=True)
    train.to_csv(f"{AGG_DIR}/train.csv", index=False)

    stats = {"train": len(train), "test": len(test), "test_questions_without_reference": len(eval_qs) - len(test),
             "train_profile_share": {v: round(float(train[f"prof_{v}"].mean()), 3) for v in VALUES},
             "train_values_per_profile": round(float(train[[f"prof_{v}" for v in VALUES]].sum(axis=1).mean()), 2)}
    print(json.dumps(stats, indent=2))
    with open(f"{AGG_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Each agent answers the training questions (greedy, as at evaluation)
    from generate_and_score import generate
    for arm in ["full", "lora"]:
        for v in VALUES:
            path = f"{AGG_DIR}/answers/{arm}_{v}.csv"
            if os.path.exists(path):
                print(f"{arm}_{v}: already generated")
                continue
            agent_dir = f"{MODELS_DIR}/agents/{arm}_{v}_sd"
            model, adapter = (agent_dir, None) if arm == "full" else (BASE_LM, agent_dir)
            print(f"{arm}_{v}: answering {len(train)} training questions")
            replies, _ = generate(model, adapter, train["Q"].tolist(), args.batch, args.max_new_tokens)
            pd.DataFrame({"Q": train["Q"], "reply": replies}).to_csv(path, index=False)
    print(f"Saved aggregator data to {AGG_DIR}")


if __name__ == "__main__":
    main()
