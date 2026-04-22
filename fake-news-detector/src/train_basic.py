"""
train_basic.py — Train TF-IDF + Logistic Regression classifier.

Fixes applied:
  • Vectorizer fitted on train only (no leakage)
  • Reduced TF-IDF vocab (30k) to curb memorisation
  • 5-fold CV on training set to estimate true generalisation
  • Balanced class weights to handle slight label imbalance
  • Metrics saved to JSON for the comparison dashboard
"""

import sys
import json
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from utils import get_logger, save_model, load_model, MODELS_DIR, decode_label
from data_preprocessing import run_preprocessing_pipeline, load_splits, clean_text
from feature_engineering import build_tfidf_features, transform_single, VECTORIZER_FILE

logger = get_logger("train_basic")

MODEL_FILE   = "basic_model.pkl"
METRICS_FILE = MODELS_DIR / "basic_metrics.json"

LR_CONFIG = {
    "C":             2.0,      # softer regularisation than default
    "max_iter":      1000,
    "solver":        "lbfgs",
    "n_jobs":        -1,
    "random_state":  42,
    "class_weight":  "balanced",
}


# ─────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────

def evaluate(model, X, y, split_name: str = "Test") -> dict:
    y_pred = model.predict(X)
    metrics = {
        "split":     split_name,
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
    }
    logger.info(f"\n── {split_name} ──────────────────────────")
    for k, v in metrics.items():
        if k != "split":
            logger.info(f"  {k:<12}: {v}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=['FAKE','REAL'])}")
    return metrics


# ─────────────────────────────────────────────────────────────
# Inference helper (used by Streamlit app)
# ─────────────────────────────────────────────────────────────

def predict_text(text: str, model=None, vectorizer=None) -> dict:
    """
    Classify a raw text string.
    Loads saved model + vectorizer from disk if not passed in.
    Returns dict: {label, confidence, prob_fake, prob_real}
    """
    if model is None:
        model = load_model(MODEL_FILE)
    if vectorizer is None:
        vectorizer = load_model(VECTORIZER_FILE)

    cleaned = clean_text(text)
    X       = vectorizer.transform([cleaned])

    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]   # [P(FAKE), P(REAL)]
    label = decode_label(pred)

    return {
        "label":      label,
        "confidence": float(proba[pred]),
        "prob_fake":  float(proba[0]),
        "prob_real":  float(proba[1]),
    }


# ─────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────

def run_training_pipeline(force_preprocess: bool = False) -> dict:
    """
    End-to-end: preprocess → features → cross-val → train → evaluate → save.
    Returns test-set metrics dict.
    """
    # 1. Load or generate preprocessed splits
    train_csv = MODELS_DIR.parent / "data" / "processed" / "train.csv"
    if force_preprocess or not train_csv.exists():
        logger.info("Running preprocessing pipeline…")
        train_df, val_df, test_df = run_preprocessing_pipeline()
    else:
        train_df, val_df, test_df = load_splits()

    # 2. Build TF-IDF features (fit on train ONLY)
    X_train, X_val, X_test, vec = build_tfidf_features(
        train_df["clean_text"],
        val_df["clean_text"],
        test_df["clean_text"],
    )
    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    # 3. 5-fold stratified CV on training set
    logger.info("5-fold stratified cross-validation…")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(
        LogisticRegression(**LR_CONFIG), X_train, y_train,
        cv=cv, scoring=["accuracy", "f1"], n_jobs=-1,
    )
    logger.info(
        f"CV Accuracy: {cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}"
    )
    logger.info(
        f"CV F1:       {cv_res['test_f1'].mean():.4f} ± {cv_res['test_f1'].std():.4f}"
    )

    # 4. Train final model on full training set
    logger.info("Training final Logistic Regression…")
    model = LogisticRegression(**LR_CONFIG)
    model.fit(X_train, y_train)

    # 5. Evaluate
    val_metrics  = evaluate(model, X_val,  y_val,  "Validation")
    test_metrics = evaluate(model, X_test, y_test, "Test")

    # 6. Save model
    save_model(model, MODEL_FILE)

    # 7. Persist metrics for dashboard
    all_metrics = {
        "model":        "Logistic Regression + TF-IDF",
        "cv_accuracy":  round(float(cv_res["test_accuracy"].mean()), 4),
        "cv_f1":        round(float(cv_res["test_f1"].mean()), 4),
        "validation":   val_metrics,
        "test":         test_metrics,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved → {METRICS_FILE}")

    return test_metrics


if __name__ == "__main__":
    metrics = run_training_pipeline()
    print(f"\n✅  Test Accuracy : {metrics['accuracy']}")
    print(f"    Test F1       : {metrics['f1']}")
