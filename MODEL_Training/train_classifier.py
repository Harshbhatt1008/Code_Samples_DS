import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# -------------------
# 1. Config
# -------------------
MODEL_NAME = "distilroberta-base"   # you can change this
DATA_PATH = "data/prompts.csv"
OUTPUT_DIR = "models/hallucination_binary"
MAX_LENGTH = 128                    # max tokens per prompt


# -------------------
# 2. Load & preprocess data
# -------------------
def load_data(path):
    df = pd.read_csv(path)

    # Clean column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Only keep necessary columns
    df = df[["prompt", "hallucination_risk"]].dropna()

    # Normalize text labels
    df["hallucination_risk"] = df["hallucination_risk"].str.strip().str.lower()

    # Map to integer labels
    label_map = {"no": 0, "yes": 1}
    df["label"] = df["hallucination_risk"].map(label_map)

    # Drop any rows with unknown labels
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df


# -------------------
# 3. Dataset class
# -------------------
class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        prompt = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# -------------------
# 4. Metrics
# -------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}


# -------------------
# 5. Main training pipeline
# -------------------
def main():
    # Load data
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Train/val split
    train_df, val_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["label"],
    )

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # Create Datasets
    train_dataset = PromptDataset(
        texts=train_df["prompt"],
        labels=train_df["label"],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    val_dataset = PromptDataset(
        texts=val_df["prompt"],
        labels=val_df["label"],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    # Training arguments
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # Older models use "eval_steps" or "no evaluation strategy at all"
    do_eval=True,

    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
)


    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Final evaluation
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)

    # Detailed report
    preds = trainer.predict(val_dataset)
    pred_labels = np.argmax(preds.predictions, axis=1)
    true_labels = preds.label_ids
    print("\nClassification report:\n")
    print(classification_report(true_labels, pred_labels, digits=4))

    # -------------------
    # 6. Save model in safetensors format
    # -------------------
    print(f"Saving model to {OUTPUT_DIR} in safetensors format")

    # Make sure directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save tokenizer (unchanged)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save model using safe serialization (creates model.safetensors)
    trainer.model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    # OPTIONAL: remove any old PyTorch .bin file if Trainer created one
    bin_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
    if os.path.exists(bin_path):
        print("Removing old PyTorch .bin file:", bin_path)
        os.remove(bin_path)


if __name__ == "__main__":
    main()
