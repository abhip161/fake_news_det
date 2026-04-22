"""
feature_engineering.py — TF-IDF feature extraction.

Key fix: vectorizer is FIT only on training data, then TRANSFORMS val/test.
This prevents vocabulary leakage from test/val into the model.

Config:
  max_features = 30,000  (reduced from 100k to limit overfitting)
  ngram_range  = (1, 2)  (unigrams + bigrams)
  sublinear_tf = True    (log-scale TF dampening)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from typing import Optional, Tuple

from utils import get_logger, save_model, load_model

logger = get_logger("feature_engineering")

VECTORIZER_FILE = "tfidf_vectorizer.pkl"

TFIDF_CONFIG = {
    "max_features":  30_000,   # smaller vocab → less memorisation
    "ngram_range":   (1, 2),
    "sublinear_tf":  True,
    "min_df":        5,        # ignore very rare terms
    "max_df":        0.90,     # ignore near-universal terms
    "strip_accents": "unicode",
    "analyzer":      "word",
    "dtype":         np.float32,
}


def build_tfidf_features(
    train_texts: pd.Series,
    val_texts:   Optional[pd.Series] = None,
    test_texts:  Optional[pd.Series] = None,
) -> Tuple[csr_matrix, Optional[csr_matrix], Optional[csr_matrix], TfidfVectorizer]:
    """
    Fit TF-IDF on train_texts only, then transform val/test.
    Saves the fitted vectorizer for inference.

    Returns: (X_train, X_val, X_test, vectorizer)
    """
    logger.info("Fitting TF-IDF vectorizer on TRAINING data only…")
    vec = TfidfVectorizer(**TFIDF_CONFIG)

    X_train = vec.fit_transform(train_texts.fillna(""))
    logger.info(f"Vocab size: {len(vec.vocabulary_):,}  |  X_train: {X_train.shape}")

    X_val  = vec.transform(val_texts.fillna(""))  if val_texts  is not None else None
    X_test = vec.transform(test_texts.fillna("")) if test_texts is not None else None

    save_model(vec, VECTORIZER_FILE)
    return X_train, X_val, X_test, vec


def transform_single(text: str) -> csr_matrix:
    """Transform one cleaned text string at inference time."""
    vec = load_model(VECTORIZER_FILE)
    return vec.transform([text])


if __name__ == "__main__":
    from data_preprocessing import load_splits
    train, val, test = load_splits()
    X_tr, X_va, X_te, v = build_tfidf_features(
        train["clean_text"], val["clean_text"], test["clean_text"]
    )
    print(f"X_train={X_tr.shape}  X_val={X_va.shape}  X_test={X_te.shape}")
