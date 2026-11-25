"""Step 6: self-distillation data. The base model is its own teacher; the reward models choose.

    python src/distill_data.py

Three stages, each skipped if its output already exists on Drive (safe to rerun after a
Colab disconnect; generation also resumes chunk by chunk):

  1. Generate: Qwen answers N MIC *train* questions K times each, with sampling.
  2. Score:    every answer is scored by all five reward models (question + answer).
  3. Select:   for each value, keep the highest-scoring answer per question.
               The control set takes one random answer per question instead.

Output (on Drive):
  data/distill/candidates/part_XXX.csv    raw generations, one file per chunk
  data/distill/candidates_scored.csv      all answers with p_care ... p_sanctity
  data/agent_data_distilled/{value}.csv   columns Q, A, score, finished   (training data per value)
  data/agent_data_distilled/control.csv   columns Q, A, finished          (random answer, no selection)
  data/agent_data_distilled/stats.json    the gap check: selected vs. average answer
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from config import BASE_LM, DATA_DIR, MIC_PATH, SEED, VALUES
from generate_and_score import score
from prepare_data import COL_Q, assign_split, load_mic

DISTILL_DIR = f"{DATA_DIR}/distill"
CAND_DIR = f"{DISTILL_DIR}/candidates"
OUT_DIR = f"{DATA_DIR}/agent_data_distilled"


def pick_questions(n):
    df = load_mic(MIC_PATH)
    df["_split"] = assign_split(df, np.random.default_rng(SEED))
    qs = pd.Series(df.loc[df["_split"] == "train", COL_Q].unique())
    return qs.sample(n=min(n, len(qs)), random_state=SEED).tolist()


def generate(questions, args):
    os.makedirs(CAND_DIR, exist_ok=True)
    chunks = [questions[i:i + args.chunk] for i in range(0, len(questions), args.chunk)]
    if all(os.path.exists(f"{CAND_DIR}/part_{c:03d}.csv") for c in range(len(chunks))):
        print(f"All {len(chunks)} chunks already generated")
        return
    tok = AutoTokenizer.from_pretrained(BASE_LM, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(BASE_LM, dtype=torch.float16, device_map="cuda").eval()
    stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

    for c, chunk in enumerate(chunks):
        path = f"{CAND_DIR}/part_{c:03d}.csv"
        if os.path.exists(path):
            print(f"chunk {c + 1}/{len(chunks)} already done")
            continue
        set_seed(SEED + c)  # each chunk reproducible on its own, so resuming gives the same data
        rows = []
        for i in range(0, len(chunk), args.batch):
            batch = chunk[i:i + args.batch]
            prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                               add_generation_prompt=True) for q in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                     num_return_sequences=args.k, max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(new, skip_special_tokens=True)
            for r in range(new.shape[0]):
                finished = any(int(t) in stop_ids for t in new[r])  # stopped on its own, not cut at the limit
                rows.append({"Q": batch[r // args.k], "cand": r % args.k,
                             "reply": texts[r].strip(), "finished": finished})
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"chunk {c + 1}/{len(chunks)}: {len(rows)} answers saved")
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_questions", type=int, default=2000)
    ap.add_argument("--k", type=int, default=5, help="answers sampled per question")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16, help="questions per generate() call")
    ap.add_argument("--chunk", type=int, default=200, help="questions per saved file")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Generate (resumable)
    questions = pick_questions(args.n_questions)
    print(f"{len(questions)} train questions x {args.k} answers each")
    generate(questions, args)

    # 2. Score (once)
    scored_path = f"{DISTILL_DIR}/candidates_scored.csv"
    if os.path.exists(scored_path):
        cands = pd.read_csv(scored_path)
        print("Scores already computed")
    else:
        cands = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{CAND_DIR}/part_*.csv"))], ignore_index=True)
        cands["reply"] = cands["reply"].fillna("").astype(str)
        for v in VALUES:
            print(f"scoring {len(cands):,} answers for {v}")
            cands[f"p_{v}"] = score(v, cands["Q"].tolist(), cands["reply"].tolist())
        cands.to_csv(scored_path, index=False)

    # 3. Select from all non-empty answers. Most answers hit the 128-token limit, exactly as at
    # evaluation, so a cut-off answer is scored and trained on as the same 128-token prefix the
    # reward model saw. The 'finished' column tells train_agent.py whether to teach the stop token.
    cands["finished"] = cands["finished"].astype(bool)
    ok = cands[cands["reply"].fillna("").str.strip() != ""]
    stats = {"answers_total": len(cands), "answers_usable": len(ok),
             "pct_finished": round(float(cands["finished"].mean()), 3),
             "questions_usable": int(ok["Q"].nunique())}
    print(f"\n{len(ok):,} of {len(cands):,} answers usable; "
          f"{stats['pct_finished']:.0%} finished within {args.max_new_tokens} tokens")

    control = ok.groupby("Q", group_keys=False).sample(n=1, random_state=SEED)
    control[["Q", "reply", "finished"]].rename(columns={"reply": "A"}).to_csv(f"{OUT_DIR}/control.csv", index=False)

    print(f"\n{'value':10s} {'average answer':>15s} {'random pick':>12s} {'selected':>9s} {'gap':>6s}")
    for v in VALUES:
        col = f"p_{v}"
        best = ok.loc[ok.groupby("Q")[col].idxmax()]
        best[["Q", "reply", col, "finished"]].rename(columns={"reply": "A", col: "score"}).to_csv(
            f"{OUT_DIR}/{v}.csv", index=False)
        s = {"mean_all_answers": round(float(ok[col].mean()), 4),
             "mean_random_pick": round(float(control[col].mean()), 4),
             "mean_selected": round(float(best[col].mean()), 4),
             "examples": len(best)}
        s["gap_vs_random"] = round(s["mean_selected"] - s["mean_random_pick"], 4)
        stats[v] = s
        print(f"{v:10s} {s['mean_all_answers']:15.3f} {s['mean_random_pick']:12.3f} "
              f"{s['mean_selected']:9.3f} {s['gap_vs_random']:+6.3f}")

    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump({"settings": vars(args), **stats}, f, indent=2)
    print(f"\nSaved training sets to {OUT_DIR}")


if __name__ == "__main__":
    main()
