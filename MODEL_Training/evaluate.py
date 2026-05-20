# evaluate.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/hallucination_binary"
DATA_PATH = "data/prompts.csv"
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {"no": 0, "yes": 1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df = df[["prompt", "hallucination_risk"]].dropna()
    df["hallucination_risk"] = df["hallucination_risk"].str.strip().str.lower()
    df["label"] = df["hallucination_risk"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_batch(prompts, tokenizer, model, max_length=128):
    encodings = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    preds = probs.argmax(dim=-1).cpu().numpy()
    p_yes = probs[:, 1].cpu().numpy()
    return preds, p_yes


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Total samples: {len(df)}")

    # Train/test split (for evaluation only)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=123,
        stratify=df["label"],
    )

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    print("Loading model...")
    tokenizer, model = load_model()

    # Predict on test set in batches
    batch_size = 64
    all_preds = []
    all_probs_yes = []

    prompts = test_df["prompt"].tolist()
    labels = test_df["label"].to_numpy()

    print("Running inference on test set...")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        preds, p_yes = predict_batch(batch_prompts, tokenizer, model, MAX_LENGTH)
        all_preds.extend(preds)
        all_probs_yes.extend(p_yes)

    all_preds = np.array(all_preds)
    all_probs_yes = np.array(all_probs_yes)

    # Metrics
    acc = accuracy_score(labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, all_preds, average="binary", pos_label=1
    )

    print("\n=== Test Set Metrics (Hallucination: label=1) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(labels, all_preds, target_names=["No", "Yes"], digits=4))

    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(labels, all_preds))

    # Optional: quick look at some mistakes
    test_df = test_df.reset_index(drop=True)
    test_df["pred_label"] = all_preds
    test_df["pred_label_str"] = test_df["pred_label"].map(INV_LABEL_MAP)
    test_df["p_yes"] = all_probs_yes

    print("\nExample False Positives (predicted Yes but label No):")
    fp = test_df[(test_df["label"] == 0) & (test_df["pred_label"] == 1)]
    print(fp[["prompt", "hallucination_risk", "pred_label_str", "p_yes"]].head())

    print("\nExample False Negatives (predicted No but label Yes):")
    fn = test_df[(test_df["label"] == 1) & (test_df["pred_label"] == 0)]
    print(fn[["prompt", "hallucination_risk", "pred_label_str", "p_yes"]].head())


if __name__ == "__main__":
    main()
