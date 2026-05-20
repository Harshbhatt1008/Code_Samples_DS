"""
train_classifier.py
Trains a transformer-based classifier to predict hallucination risk (Low/Medium/High)
from prompt text alone. Outputs a saved model in safetensors format.
"""

import os
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/deberta-v3-base"
DATA_PATH = "data/prompts.csv"
OUTPUT_DIR = "models/hallucination_classifier"
MAX_LENGTH = 128


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df[["prompt", "hallucination_risk"]].dropna()
    df["hallucination_risk"] = df["hallucination_risk"].str.strip().str.lower()

    label_map = {"low": 0, "medium": 1, "high": 2}
    df["label"] = df["hallucination_risk"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


class PromptDataset(Dataset):
    """Wraps tokenized prompts and integer labels for the Trainer API."""

    def __init__(self, texts, labels, tokenizer, max_length: int = MAX_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted"),
    }


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} samples | label distribution:\n{df['label'].value_counts()}")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    train_dataset = PromptDataset(train_df["prompt"], train_df["label"], tokenizer)
    val_dataset = PromptDataset(val_df["prompt"], val_df["label"], tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        do_eval=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # detailed per-class breakdown on val set
    preds = trainer.predict(val_dataset)
    pred_labels = np.argmax(preds.predictions, axis=1)
    print("\n", classification_report(preds.label_ids, pred_labels,
                                      target_names=["Low", "Medium", "High"], digits=4))

    # save model — safetensors format to avoid the older .bin overhead
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    old_bin = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
    if os.path.exists(old_bin):
        os.remove(old_bin)

    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
