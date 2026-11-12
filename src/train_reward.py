"""Step 2: train one binary reward classifier for a moral value.

    python src/train_reward.py --value care

Saves the model to Drive (models/reward/{value}) and metrics to results/reward_{value}.json.
These classifiers are both the training signal and the evaluation metric for the
agents, so it's worth checking their quality before moving on (macro-F1 ~0.7+ is usable).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from config import MODELS_DIR, RESULTS_DIR, REWARD_BACKBONE, SCRATCH_DIR, SEED, SPLITS_DIR, VALUES


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights, since MIC labels are imbalanced."""

    def __init__(self, class_weights, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    logits, labels = pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = logits.argmax(-1)
    out = {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}
    if len(set(labels)) > 1:
        out["auc"] = roc_auc_score(labels, probs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", required=True, choices=VALUES)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--backbone", default=REWARD_BACKBONE)
    ap.add_argument("--best_metric", default="auc", choices=["auc", "macro_f1"])
    args = ap.parse_args()
    set_seed(SEED)

    tok = AutoTokenizer.from_pretrained(args.backbone)
    ds = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(f"{SPLITS_DIR}/{args.value}_{split}.csv").dropna()
        ds[split] = Dataset.from_pandas(df, preserve_index=False).map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True
        ).rename_column("label", "labels").remove_columns(["text"])

    labels = np.array(ds["train"]["labels"])
    counts = np.bincount(labels, minlength=2)
    weights = torch.tensor(len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float)
    print(f"Train label counts: {counts.tolist()} -> class weights {weights.tolist()}")

    model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=2)
    targs = TrainingArguments(
        output_dir=f"{SCRATCH_DIR}/reward_{args.value}",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )
    trainer = WeightedTrainer(
        class_weights=weights, model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    test = trainer.evaluate(ds["test"], metric_key_prefix="test")
    majority = max(np.mean(ds["test"]["labels"]), 1 - np.mean(ds["test"]["labels"]))

    out_dir = f"{MODELS_DIR}/reward/{args.value}"
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    metrics = {
        "value": args.value,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "test_accuracy": test.get("test_accuracy"),
        "test_macro_f1": test.get("test_macro_f1"),
        "test_auc": test.get("test_auc"),
        "majority_class_accuracy": float(majority),
        "train_rows": len(ds["train"]),
        "train_seconds": round(train_secs, 1),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
    }
    with open(f"{RESULTS_DIR}/reward_{args.value}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved reward model to {out_dir}")


if __name__ == "__main__":
    main()
