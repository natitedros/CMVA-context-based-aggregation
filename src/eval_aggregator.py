"""Step 9: evaluate final answers on the held-out questions.

    python src/eval_aggregator.py --system pt                 # base model alone (paper's PT-model)
    python src/eval_aggregator.py --system select --arm lora  # rule: answer of the first relevant value's agent
    python src/eval_aggregator.py --system prompted --arm lora  # untrained base model asked to aggregate (paper's Agg baseline)
    python src/eval_aggregator.py --system agg --arm lora     # trained aggregator

Metrics, per system:
  ROUGE-1/2/L/Lsum F1 against the MIC reference answer (the paper's metric)
  profile reward: mean reward-model score over the values the user's profile marks as relevant
  expected reward on each of the five values
Writes results/final_{name}.json and results/final_{name}.csv.
"""
import argparse
import json

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from aggregator_common import AGG_DIR_NAME, build_prompt, load_test_agent_answers
from config import BASE_LM, DATA_DIR, MODELS_DIR, RESULTS_DIR, VALUES
from generate_and_score import score

AGG_DIR = f"{DATA_DIR}/{AGG_DIR_NAME}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["pt", "select", "prompted", "agg"])
    ap.add_argument("--arm", choices=["full", "lora"], default=None)
    ap.add_argument("--adapter", default=None, help="default: models/aggregator/agg_{arm}")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.system != "pt" and args.arm is None:
        raise SystemExit("--arm is required for select, prompted and agg")
    name = args.system if args.system == "pt" else f"{args.system}_{args.arm}"
    name += args.tag

    test = pd.read_csv(f"{AGG_DIR}/test.csv")
    prof = test[[f"prof_{v}" for v in VALUES]].to_numpy()

    if args.system == "pt":
        pt = pd.read_csv(f"{RESULTS_DIR}/generations_pt.csv").drop_duplicates("Q")
        replies = test.merge(pt, on="Q", how="left")["reply"].fillna("").tolist()
    else:
        answers = test[["Q"]].merge(load_test_agent_answers(RESULTS_DIR, args.arm), on="Q", how="left")
        if args.system == "select":
            first = prof.argmax(axis=1)  # first relevant value in fixed order
            replies = [answers.iloc[i][f"ans_{VALUES[j]}"] for i, j in enumerate(first)]
        else:
            from generate_and_score import generate
            prompts = [build_prompt(q, p, {v: answers.iloc[i][f"ans_{v}"] for v in VALUES})
                       for i, (q, p) in enumerate(zip(test["Q"], prof))]
            adapter = None if args.system == "prompted" else (args.adapter or f"{MODELS_DIR}/aggregator/agg_{args.arm}")
            replies, _ = generate(BASE_LM, adapter, prompts, args.batch, args.max_new_tokens)
    replies = [str(r) for r in replies]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    rouge = [scorer.score(str(ref), hyp) for ref, hyp in zip(test["reference"], replies)]
    out = test[["Q", "reference"]].copy()
    out["reply"] = replies
    for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        out[k] = [r[k].fmeasure for r in rouge]
    for v in VALUES:
        out[f"p_{v}"] = score(v, test["Q"].tolist(), replies)
    p = out[[f"p_{v}" for v in VALUES]].to_numpy()
    out["profile_reward"] = (p * prof).sum(axis=1) / prof.sum(axis=1)

    summary = {"system": name, "n": len(out),
               **{k: round(float(out[k].mean()), 4) for k in ["rouge1", "rouge2", "rougeL", "rougeLsum", "profile_reward"]},
               "expected_reward": {v: round(float(out[f"p_{v}"].mean()), 4) for v in VALUES}}
    out.to_csv(f"{RESULTS_DIR}/final_{name}.csv", index=False)
    with open(f"{RESULTS_DIR}/final_{name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
