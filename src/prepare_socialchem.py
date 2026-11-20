"""Build per-value reward-model datasets from Social Chemistry 101 (the paper's source).

    python src/prepare_socialchem.py

Output (on Drive, under data/splits_socialchem/):
  {value}_train.csv  Social Chemistry actions, official 'train' split
  {value}_val.csv    Social Chemistry actions, official 'dev' split (checkpoint selection)
  {value}_test.csv   the SAME MIC test rows as data/splits/, reduced to the answer text,
                     so results are directly comparable with the MIC-trained classifiers
  stats.json         row counts and label balance

Input text = the action (e.g. "keeping things clean."). The paper scores the LLM's
response alone, so at test time we feed only the chatbot's answer from MIC.
Label: action-moral-judgment < 0 -> 0 (bad), 0..2 -> 1 (neutral/good), as in the paper.
"""
import argparse
import json
import os
import re

import pandas as pd

from config import DATA_DIR, SEED, SPLITS_DIR, VALUES

SC_PATH = f"{DATA_DIR}/social-chem-101.v1.0.tsv"
OUT_DIR = f"{DATA_DIR}/splits_socialchem"


def has_value(field, value):
    tokens = re.split(r"[|,;/]", str(field).lower())
    return any(t.strip().startswith(value) for t in tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_train", type=int, default=30000)
    ap.add_argument("--max_val", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    sc = pd.read_csv(SC_PATH, sep="\t")
    print(f"Loaded {len(sc):,} rows")
    sc = sc.dropna(subset=["action", "action-moral-judgment", "rot-moral-foundations"])
    if "rot-bad" in sc.columns:  # annotators flagged these rules of thumb as unusable
        sc = sc[sc["rot-bad"] != 1]
    sc["action"] = sc["action"].astype(str).str.strip()
    print(f"{len(sc):,} rows with an action, a judgment and a foundation tag")
    print("Judgment values:", sc["action-moral-judgment"].value_counts().sort_index().to_dict())

    stats = {}
    for v in VALUES:
        sub = sc[sc["rot-moral-foundations"].apply(has_value, value=v)]
        stats[v] = {}
        for sc_split, out_split, cap in (("train", "train", args.max_train), ("dev", "val", args.max_val)):
            part = sub[sub["split"] == sc_split]
            # Same action judged several times -> average the judgments, then binarize
            part = part.groupby("action", as_index=False)["action-moral-judgment"].mean()
            part["label"] = (part["action-moral-judgment"] >= 0).astype(int)
            part = part.sample(n=min(cap, len(part)), random_state=SEED)
            part = part.rename(columns={"action": "text"})[["text", "label"]]
            part.to_csv(f"{OUT_DIR}/{v}_{out_split}.csv", index=False)
            stats[v][out_split] = {"rows": len(part), "pos_rate": round(float(part["label"].mean()), 3)}

        # Fixed yardstick: MIC test rows, answer only
        mic = pd.read_csv(f"{SPLITS_DIR}/{v}_test.csv")
        mic["text"] = mic["text"].str.split("\nA: ", n=1).str[1]
        mic.dropna().to_csv(f"{OUT_DIR}/{v}_test.csv", index=False)
        stats[v]["test_mic"] = {"rows": len(mic), "pos_rate": round(float(mic["label"].mean()), 3)}
        print(f"{v:10s}", stats[v])

    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
