"""
data_preprocessing.py — FIXED pipeline with deduplication, bias removal,
and correct split-before-preprocess ordering to eliminate data leakage.

Key fixes vs original:
  1. Duplicates removed on raw text BEFORE any processing
  2. Train/val/test split happens on RAW data — vectorizer fitted only on train
  3. Style-leaking fields (agency bylines, Reuters/AP tags) are stripped
  4. Near-duplicate detection via content hash
  5. Dataset is shuffled with a fixed seed for reproducibility
"""

import re
import string
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils import get_logger, DATA_RAW_DIR, DATA_PROCESSED_DIR

# Download required NLTK resources (idempotent)
for _pkg in ["stopwords", "wordnet", "punkt", "omw-1.4"]:
    nltk.download(_pkg, quiet=True)

logger = get_logger("data_preprocessing")

# ─────────────────────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────────────────────

FAKE_CSV      = DATA_RAW_DIR / "Fake.csv"
TRUE_CSV      = DATA_RAW_DIR / "True.csv"
PROCESSED_CSV = DATA_PROCESSED_DIR / "processed_news.csv"

# ─────────────────────────────────────────────────────────────
# NLTK resources
# ─────────────────────────────────────────────────────────────

STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()

# ─────────────────────────────────────────────────────────────
# Style-leaking noise patterns (agency bylines that act as
# class labels → model learns format, not truth)
# ─────────────────────────────────────────────────────────────

_AGENCY_PATTERNS = [
    r"\breuters\b",
    r"\bap\b",          # Associated Press
    r"\bafp\b",
    r"\(reuters\)",
    r"\(ap\)",
    r"[A-Z]{2,}\s*\([A-Z]+\)\s*[-–]",   # "CITY (AGENCY) -"
    r"^[A-Z\s,]+\([A-Za-z]+\)\s*[-–]\s*",  # leading bylines
]
_AGENCY_RE = re.compile("|".join(_AGENCY_PATTERNS), re.IGNORECASE)

_URL_RE   = re.compile(r"http\S+|www\.\S+")
_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation))
_DIGIT_RE = re.compile(r"\d+")
_MULTI_RE = re.compile(r"\s{2,}")


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load raw data
# ─────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """
    Load Fake.csv / True.csv, assign labels, return merged DataFrame.
    Labels: Fake → 0, Real → 1
    """
    if not FAKE_CSV.exists() or not TRUE_CSV.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {DATA_RAW_DIR}.\n"
            "Download Fake.csv and True.csv from Kaggle and place them in data/raw/."
        )

    logger.info("Loading raw CSVs…")
    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    # Keep only title + text columns; drop anything else (avoids subject/date leakage)
    for df in (fake_df, true_df):
        for col in list(df.columns):
            if col not in ("title", "text"):
                df.drop(columns=[col], inplace=True, errors="ignore")

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    logger.info(f"Raw: {len(df):,} rows  (Fake={len(fake_df):,}, Real={len(true_df):,})")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — Deduplication (BEFORE split)
# ─────────────────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    """MD5 hash of lowercased, whitespace-normalised text (near-dup detection)."""
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact and near-duplicate articles.
    Strategy:
      1. Drop rows with null title or text
      2. Exact duplicate on combined title+text
      3. Near-duplicate via content hash (catches minor whitespace diffs)
      4. Duplicate title (same headline = same story, different copy)
    """
    before = len(df)

    # 1. Drop nulls
    df = df.dropna(subset=["title", "text"]).copy()

    # 2. Combine for dedup key
    df["_combined"] = (df["title"].str.strip() + " " + df["text"].str.strip())

    # 3. Exact duplicate
    df = df.drop_duplicates(subset=["_combined"])

    # 4. Near-duplicate hash
    df["_hash"] = df["_combined"].apply(_content_hash)
    df = df.drop_duplicates(subset=["_hash"])

    # 5. Duplicate title
    df = df.drop_duplicates(subset=["title"])

    df.drop(columns=["_combined", "_hash"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Deduplication: {before:,} → {len(df):,} rows (removed {before-len(df):,})")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — Split BEFORE preprocessing (critical for no leakage)
# ─────────────────────────────────────────────────────────────

def split_raw(
    df: pd.DataFrame,
    test_size: float  = 0.15,
    val_size: float   = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split on RAW (uncleaned) data.
    The vectorizer must be fitted ONLY on train_df to prevent leakage.

    Returns: (train_df, val_df, test_df)
    """
    # Shuffle first so class ordering from concat doesn't bias anything
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label"], random_state=random_state
    )

    logger.info(
        f"Split → Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}"
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STEP 4 — Text cleaning
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean a single string for TF-IDF / basic model:
      1. Strip agency bylines (style leakage)
      2. Remove URLs
      3. Lowercase
      4. Remove punctuation and digits
      5. Tokenise, remove stopwords (len ≥ 2)
      6. Lemmatise
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Strip agency bylines
    text = _AGENCY_RE.sub(" ", text)

    # 2. Remove URLs
    text = _URL_RE.sub(" ", text)

    # 3. Lowercase
    text = text.lower()

    # 4. Remove punctuation & digits
    text = _PUNCT_RE.sub(" ", text)
    text = _DIGIT_RE.sub(" ", text)

    # 5. Tokenise + stopword filter
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) >= 2]

    # 6. Lemmatise
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build combined_text (title + text) and clean it.
    Drops rows where cleaned text is empty after processing.
    """
    df = df.copy()
    df["combined_text"] = (
        df["title"].fillna("") + " " + df["text"].fillna("")
    ).str.strip()

    df["clean_text"] = df["combined_text"].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
    if before - len(df):
        logger.warning(f"Dropped {before-len(df)} rows with empty clean_text.")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 — Persist / load processed splits
# ─────────────────────────────────────────────────────────────

def save_splits(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    val_df  .to_csv(DATA_PROCESSED_DIR / "val.csv",   index=False)
    test_df .to_csv(DATA_PROCESSED_DIR / "test.csv",  index=False)
    logger.info(f"Splits saved to {DATA_PROCESSED_DIR}/")


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for fname in ("train.csv", "val.csv", "test.csv"):
        if not (DATA_PROCESSED_DIR / fname).exists():
            raise FileNotFoundError(
                f"{fname} not found. Run the preprocessing pipeline first."
            )
    train = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val   = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
    test  = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    logger.info(f"Loaded splits — Train:{len(train):,} Val:{len(val):,} Test:{len(test):,}")
    return train, val, test


# ─────────────────────────────────────────────────────────────
# Main pipeline entrypoint
# ─────────────────────────────────────────────────────────────

def run_preprocessing_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline:
      load → deduplicate → split (raw) → clean each split → save
    Returns (train_df, val_df, test_df) with both combined_text and clean_text.
    """
    raw   = load_raw_data()
    dedup = remove_duplicates(raw)

    # Split on RAW data — prevents any cleaning artefact from leaking across splits
    train_raw, val_raw, test_raw = split_raw(dedup)

    logger.info("Cleaning text for each split independently…")
    train_df = apply_cleaning(train_raw)
    val_df   = apply_cleaning(val_raw)
    test_df  = apply_cleaning(test_raw)

    save_splits(train_df, val_df, test_df)
    return train_df, val_df, test_df


if __name__ == "__main__":
    tr, va, te = run_preprocessing_pipeline()
    print(f"\nTrain label dist:\n{tr['label'].value_counts()}")
    print(f"\nTest label dist:\n{te['label'].value_counts()}")
