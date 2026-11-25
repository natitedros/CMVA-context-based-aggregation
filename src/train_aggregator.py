"""Step 8: train the contextual aggregator for one arm (LoRA on the base model, as in the paper).

    python src/train_aggregator.py --arm full
    python src/train_aggregator.py --arm lora

Input: question + moral profile + the five agents' answers from that arm.
Target: the MIC reference answer (agreeing with the rule the profile comes from).
Both arms' aggregators use identical settings; only the agent answers they read differ.
"""
import argparse
import json
import math
import os
import time

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from aggregator_common import AGG_DIR_NAME, build_prompt
from config import BASE_LM, DATA_DIR, MODELS_DIR, RESULTS_DIR, SCRATCH_DIR, SEED, VALUES
from train_agent import LORA_TARGETS, collate, dir_size_mb, tokenize

AGG_DIR = f"{DATA_DIR}/{AGG_DIR_NAME}"


def load_train(arm):
    df = pd.read_csv(f"{AGG_DIR}/train.csv")
    for v in VALUES:
        ans = pd.read_csv(f"{AGG_DIR}/answers/{arm}_{v}.csv")["reply"].fillna("").astype(str)
        df[f"ans_{v}"] = ans.values  # same row order as train.csv
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full", "lora"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)

    name = f"agg_{args.arm}{args.tag}"
    out_dir = f"{MODELS_DIR}/aggregator/{name}"
    if os.path.exists(out_dir) and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists. Use a different --tag, or add --overwrite.")

    tok = AutoTokenizer.from_pretrained(BASE_LM)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    df = load_train(args.arm)
    examples, skipped = [], 0
    for _, r in df.iterrows():
        prompt = build_prompt(r["Q"], [r[f"prof_{v}"] for v in VALUES], {v: r[f"ans_{v}"] for v in VALUES})
        ex = tokenize(tok, prompt, str(r["reference"]), args.max_len, end_id)
        if all(l == -100 for l in ex["labels"]):  # the prompt alone filled max_len: no target left
            skipped += 1
            continue
        examples.append(ex)
    ds = Dataset.from_list(examples)
    lengths = [len(e["input_ids"]) for e in examples]
    print(f"{len(ds)} training examples ({skipped} skipped as too long); "
          f"median length {int(pd.Series(lengths).median())} tokens, max {max(lengths)}")

    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(BASE_LM, dtype=torch.float32)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_steps = math.ceil(len(ds) / (args.bs * args.grad_accum)) * args.epochs
    targs = TrainingArguments(
        output_dir=f"{SCRATCH_DIR}/{name}", num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        warmup_steps=max(1, round(0.03 * total_steps)), fp16=True, logging_steps=10,
        save_strategy="no", report_to="none", remove_unused_columns=False, seed=SEED)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    model.save_pretrained(out_dir)

    metrics = {"aggregator": name, "arm": args.arm, "train_examples": len(ds), "skipped_too_long": skipped,
               "trainable_params": trainable, "lr": args.lr, "rank": args.rank, "epochs": args.epochs,
               "effective_batch": args.bs * args.grad_accum, "max_len": args.max_len,
               "train_seconds": round(time.time() - t0, 1),
               "peak_train_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20),
               "avg_train_loss": round(result.training_loss, 4), "saved_size_mb": dir_size_mb(out_dir)}
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
