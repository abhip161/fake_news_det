"""
utils.py — Shared utility functions for TruthLens Fake News Detection System.

Covers: logging, path constants, model I/O, prediction history, label helpers.
"""

import os
import json
import logging
import joblib
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────────────────────

ROOT_DIR           = Path(__file__).resolve().parent.parent
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR         = ROOT_DIR / "models"
HISTORY_FILE       = ROOT_DIR / "data" / "prediction_history.json"

for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ─────────────────────────────────────────────────────────────
# Model Persistence
# ─────────────────────────────────────────────────────────────

def save_model(obj: Any, filename: str) -> str:
    """Serialize obj to models/<filename> using joblib."""
    path = MODELS_DIR / filename
    joblib.dump(obj, path)
    get_logger("utils").info(f"Saved  → {path}")
    return str(path)


def load_model(filename: str) -> Any:
    """Deserialize a model from models/<filename>."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    get_logger("utils").info(f"Loaded ← {path}")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# Prediction History
# ─────────────────────────────────────────────────────────────

def save_prediction(
    text: str,
    model_name: str,
    prediction: str,
    confidence: float,
    top_words: Optional[List[str]] = None,
    human_explanation: str = "",
) -> None:
    """Append one prediction record to the JSON history log."""
    record = {
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "model":             model_name,
        "prediction":        prediction,
        "confidence":        round(float(confidence), 4),
        "text_preview":      text[:150].strip(),
        "top_words":         top_words or [],
        "human_explanation": human_explanation,
    }
    history = load_history()
    history.append(record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_history() -> List[Dict]:
    """Return all saved prediction records, or [] if none."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def clear_history() -> None:
    """Wipe the prediction history file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


# ─────────────────────────────────────────────────────────────
# Label Helpers
# ─────────────────────────────────────────────────────────────

LABEL_MAP   = {0: "FAKE", 1: "REAL"}
LABEL_COLOR = {"FAKE": "#ef4444", "REAL": "#22c55e"}


def decode_label(label: int) -> str:
    return LABEL_MAP.get(int(label), "UNKNOWN")


def label_color(label: str) -> str:
    return LABEL_COLOR.get(label.upper(), "#6b7280")


# ─────────────────────────────────────────────────────────────
# Text Helpers
# ─────────────────────────────────────────────────────────────

def truncate_words(text: str, max_words: int = 400) -> str:
    """Truncate to max_words (safe limit for transformer models)."""
    words = text.split()
    return " ".join(words[:max_words])


def is_valid_input(text: str, min_words: int = 10) -> Tuple[bool, str]:
    """Return (ok, error_msg). Input must have ≥ min_words."""
    stripped = text.strip()
    if not stripped:
        return False, "Input cannot be empty."
    wc = len(stripped.split())
    if wc < min_words:
        return False, f"Please enter at least {min_words} words (got {wc})."
    return True, ""


def extract_domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else url
