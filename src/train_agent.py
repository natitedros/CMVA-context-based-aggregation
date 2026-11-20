"""Step 5: train one moral agent, either as a full fine-tune (arm A) or a LoRA adapter (arm B).

    python src/train_agent.py --mode full --value care
    python src/train_agent.py --mode lora --value care
    python src/train_agent.py --mode lora --value control          # style control
    python src/train_agent.py --mode lora --value care --rank 4 --tag _r4   # rank sweep

Both arms use the same data, prompt format, loss, batch size, epochs and precision.
The only differences are which weights are trainable and the learning rate, which is
each method's standard value (LoRA's small adapters need a ~10x larger step size).

Saves:
  full -> models/agents/full_{value}{tag}   whole model in fp16 (+ tokenizer)
  lora -> models/agents/lora_{value}{tag}   adapter weights only
  results/agent_{mode}_{value}{tag}.json    trainable params, peak memory, time, disk size
"""
import argparse
import json
import os
import time
import math

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from config import BASE_LM, DATA_DIR, MODELS_DIR, RESULTS_DIR, SCRATCH_DIR, SEED, VALUES

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize(tok, q, a, max_len, end_id):
    """Same chat format as generation. Loss is computed on the answer tokens only."""
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                     add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(a, add_special_tokens=False)["input_ids"] + [end_id]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate(batch, pad_id):
    n = max(len(x["input_ids"]) for x in batch)
    pad = lambda seq, v: seq + [v] * (n - len(seq))
    return {
        "input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in batch]),
        "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch]),
        "labels": torch.tensor([pad(x["labels"], -100) for x in batch]),
    }


def dir_size_mb(path):
    return round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 2**20, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["full", "lora"])
    ap.add_argument("--value", required=True, choices=VALUES + ["control"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-5 full, 2e-4 lora")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--n_train", type=int, default=None, help="use only the first N examples")
    ap.add_argument("--grad_ckpt", action="store_true", help="trade speed for memory (use in BOTH arms)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    lr = args.lr or (1e-5 if args.mode == "full" else 2e-4)
    set_seed(SEED)

    name = f"{args.mode}_{args.value}{args.tag}"
    out_dir = f"{MODELS_DIR}/agents/{name}"
    if os.path.exists(out_dir) and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists. Use a different --tag, or add --overwrite.")

    # Data
    tok = AutoTokenizer.from_pretrained(BASE_LM)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    df = pd.read_csv(f"{DATA_DIR}/agent_data/{args.value}.csv").dropna()
    if args.n_train:
        df = df.head(args.n_train)
    ds = Dataset.from_list([tokenize(tok, q, a, args.max_len, end_id) for q, a in zip(df["Q"], df["A"])])
    print(f"{len(ds)} training examples for {name}")

    # Model: fp32 weights + fp16 mixed precision in both arms (the T4 has no bf16)
    model = AutoModelForCausalLM.from_pretrained(BASE_LM, dtype=torch.float32)
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        if args.grad_ckpt:
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    # 3% of all optimizer steps as warmup (computed here; newer transformers dropped warmup_ratio)
    total_steps = math.ceil(len(ds) / (args.bs * args.grad_accum)) * args.epochs
    warmup_steps = max(1, round(0.03 * total_steps))
    targs = TrainingArguments(
        output_dir=f"{SCRATCH_DIR}/{name}",
        num_train_epochs=args.epochs,
        learning_rate=lr,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    train_secs = time.time() - t0
    peak_mb = round(torch.cuda.max_memory_allocated() / 2**20)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]

    if args.mode == "lora":
        model.save_pretrained(out_dir)  # adapter weights only
    else:
        model.to(torch.float16).save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

    metrics = {
        "agent": name, "mode": args.mode, "value": args.value,
        "rank": args.rank if args.mode == "lora" else None,
        "lr": lr, "epochs": args.epochs, "effective_batch": args.bs * args.grad_accum,
        "train_examples": len(ds),
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "train_seconds": round(train_secs, 1),
        "peak_train_gpu_mb": peak_mb,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "avg_train_loss": round(train_result.training_loss, 4),
        "saved_size_mb": dir_size_mb(out_dir),
        "grad_ckpt": args.grad_ckpt,
    }
    with open(f"{RESULTS_DIR}/agent_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved agent to {out_dir}")


if __name__ == "__main__":
    main()
