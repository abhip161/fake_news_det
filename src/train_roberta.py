"""
train_roberta.py — Fine-tune RoBERTa (roberta-base) for fake-news classification.

Replaces BERT with RoBERTa:
  • RobertaTokenizerFast  (byte-pair encoding, no [CLS]/[SEP] vocabulary dependency)
  • RobertaForSequenceClassification
  • Early stopping + best-model checkpoint

Training config (as specified):
  max_length  = 256
  batch_size  = 8
  epochs      = 3
  lr          = 2e-5
  warmup_ratio= 0.1
  weight_decay= 0.01
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional

from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

from utils import get_logger, MODELS_DIR
from data_preprocessing import load_splits, run_preprocessing_pipeline

logger = get_logger("train_roberta")

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

ROBERTA_MODEL   = "roberta-base"
ROBERTA_DIR     = str(MODELS_DIR / "roberta_model")
METRICS_FILE    = MODELS_DIR / "roberta_metrics.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_CFG = {
    "max_length":   256,
    "batch_size":   8,
    "epochs":       3,
    "lr":           2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16":         torch.cuda.is_available(),
}

logger.info(f"Device: {DEVICE} | FP16: {TRAIN_CFG['fp16']}")


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, encodings: Dict, labels: list):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────────────────────────────────────────
# Metrics callback for Trainer
# ─────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
    }


# ─────────────────────────────────────────────────────────────
# Tokenisation helper
# ─────────────────────────────────────────────────────────────

def tokenize(texts: list, tokenizer: RobertaTokenizerFast) -> Dict:
    """Batch tokenise a list of strings with RoBERTa tokenizer."""
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=TRAIN_CFG["max_length"],
        return_tensors="pt",
    )


# ─────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────

def train_roberta() -> Dict:
    """
    Full RoBERTa fine-tuning pipeline.
    Returns test-set metrics dict.
    """
    # 1. Load preprocessed splits
    try:
        train_df, val_df, test_df = load_splits()
    except FileNotFoundError:
        logger.info("Preprocessed splits not found — running pipeline…")
        train_df, val_df, test_df = run_preprocessing_pipeline()

    # Use combined_text (raw, unstemmed) for RoBERTa — it handles its own tokenisation
    train_texts = train_df["combined_text"].fillna("").tolist()
    val_texts   = val_df["combined_text"].fillna("").tolist()
    test_texts  = test_df["combined_text"].fillna("").tolist()

    train_labels = train_df["label"].tolist()
    val_labels   = val_df["label"].tolist()
    test_labels  = test_df["label"].tolist()

    # 2. Tokenizer
    logger.info(f"Loading tokenizer: {ROBERTA_MODEL}")
    tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL)

    logger.info("Tokenising datasets…")
    train_enc = tokenize(train_texts, tokenizer)
    val_enc   = tokenize(val_texts,   tokenizer)
    test_enc  = tokenize(test_texts,  tokenizer)

    # 3. Datasets
    train_ds = NewsDataset(train_enc, train_labels)
    val_ds   = NewsDataset(val_enc,   val_labels)
    test_ds  = NewsDataset(test_enc,  test_labels)

    # 4. Model
    logger.info(f"Loading model: {ROBERTA_MODEL}")
    model = RobertaForSequenceClassification.from_pretrained(
        ROBERTA_MODEL,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1},
    )

    # 5. Training arguments
    args = TrainingArguments(
        output_dir                  = ROBERTA_DIR,
        num_train_epochs            = TRAIN_CFG["epochs"],
        per_device_train_batch_size = TRAIN_CFG["batch_size"],
        per_device_eval_batch_size  = TRAIN_CFG["batch_size"] * 2,
        learning_rate               = TRAIN_CFG["lr"],
        weight_decay                = TRAIN_CFG["weight_decay"],
        warmup_ratio                = TRAIN_CFG["warmup_ratio"],
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        fp16                        = TRAIN_CFG["fp16"],
        logging_steps               = 100,
        report_to                   = "none",
        seed                        = 42,
        dataloader_num_workers      = 0,   # safe on all OS
    )

    # 6. Trainer
    trainer = Trainer(
        model            = model,
        args             = args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        compute_metrics  = compute_metrics,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Fine-tuning RoBERTa…")
    trainer.train()

    # 7. Test evaluation
    logger.info("Evaluating on test set…")
    test_res = trainer.evaluate(test_ds)

    metrics = {
        "model": "RoBERTa (roberta-base)",
        "test": {
            "accuracy":  round(test_res.get("eval_accuracy",  0), 4),
            "precision": round(test_res.get("eval_precision", 0), 4),
            "recall":    round(test_res.get("eval_recall",    0), 4),
            "f1":        round(test_res.get("eval_f1",        0), 4),
        }
    }
    for k, v in metrics["test"].items():
        logger.info(f"  {k:<12}: {v}")

    # 8. Save model + tokenizer
    trainer.save_model(ROBERTA_DIR)
    tokenizer.save_pretrained(ROBERTA_DIR)
    logger.info(f"RoBERTa model saved → {ROBERTA_DIR}")

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────
# Inference helper (used by Streamlit app)
# ─────────────────────────────────────────────────────────────

def predict_with_roberta(
    text: str,
    model:     Optional[RobertaForSequenceClassification] = None,
    tokenizer: Optional[RobertaTokenizerFast] = None,
) -> Dict:
    """
    Classify a single news text using the fine-tuned RoBERTa model.
    Returns dict: {label, confidence, prob_fake, prob_real}
    """
    roberta_dir = Path(ROBERTA_DIR)
    if not roberta_dir.exists():
        raise FileNotFoundError(
            "RoBERTa model not found. Run `python src/train_roberta.py` first."
        )

    if model is None:
        model = RobertaForSequenceClassification.from_pretrained(str(roberta_dir))
        model.eval()
    if tokenizer is None:
        tokenizer = RobertaTokenizerFast.from_pretrained(str(roberta_dir))

    model = model.to(DEVICE)

    inputs = tokenizer(
        text,
        return_tensors   = "pt",
        truncation       = True,
        padding          = "max_length",
        max_length       = TRAIN_CFG["max_length"],
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    label    = "FAKE" if pred_idx == 0 else "REAL"

    return {
        "label":      label,
        "confidence": float(probs[pred_idx]),
        "prob_fake":  float(probs[0]),
        "prob_real":  float(probs[1]),
    }


if __name__ == "__main__":
    result = train_roberta()
    print(f"\n✅  Test F1: {result['test']['f1']}")
