"""
evaluate.py — Evaluation utilities and model comparison tools.

Provides:
  • Load metrics from saved JSON files (basic + roberta)
  • Side-by-side comparison DataFrame
  • Matplotlib plots: confusion matrix, metric bars, ROC curve
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple

from utils import get_logger, MODELS_DIR

logger = get_logger("evaluate")

BASIC_METRICS_FILE   = MODELS_DIR / "basic_metrics.json"
ROBERTA_METRICS_FILE = MODELS_DIR / "roberta_metrics.json"


# ─────────────────────────────────────────────────────────────
# Metrics Loading
# ─────────────────────────────────────────────────────────────

def load_basic_metrics() -> Optional[Dict]:
    if not BASIC_METRICS_FILE.exists():
        return None
    with open(BASIC_METRICS_FILE) as f:
        return json.load(f)


def load_roberta_metrics() -> Optional[Dict]:
    if not ROBERTA_METRICS_FILE.exists():
        return None
    with open(ROBERTA_METRICS_FILE) as f:
        return json.load(f)


def get_comparison_table() -> pd.DataFrame:
    """
    Build a tidy DataFrame comparing both models on test-set metrics.
    Returns empty DataFrame if no metrics are saved.
    """
    rows = []
    basic = load_basic_metrics()
    if basic:
        t = basic.get("test", {})
        rows.append({
            "Model":     "TF-IDF + Logistic Regression",
            "Accuracy":  t.get("accuracy",  "—"),
            "Precision": t.get("precision", "—"),
            "Recall":    t.get("recall",    "—"),
            "F1-Score":  t.get("f1",        "—"),
        })

    rob = load_roberta_metrics()
    if rob:
        t = rob.get("test", {})
        rows.append({
            "Model":     "RoBERTa (roberta-base)",
            "Accuracy":  t.get("accuracy",  "—"),
            "Precision": t.get("precision", "—"),
            "Recall":    t.get("recall",    "—"),
            "F1-Score":  t.get("f1",        "—"),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    from sklearn.metrics import confusion_matrix as sk_cm
    cm  = sk_cm(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_metric_comparison() -> Optional[plt.Figure]:
    df = get_comparison_table()
    if df.empty:
        return None

    cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    nums = df[cols].apply(pd.to_numeric, errors="coerce")
    nums.index = df["Model"]

    fig, ax = plt.subplots(figsize=(8, 4))
    nums.T.plot(kind="bar", ax=ax, width=0.6, colormap="Set2")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticklabels(cols, rotation=0)
    fig.tight_layout()
    return fig
