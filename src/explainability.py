"""
explainability.py — LIME-based word importance + human-readable explanations.

Two layers:
  1. LIME / coefficient-based word highlighting (visual)
  2. explain_prediction_text() → natural-language "WHY" paragraph
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple

from utils import get_logger

logger = get_logger("explainability")

# ─────────────────────────────────────────────────────────────
# Word lists for human-readable explanation
# ─────────────────────────────────────────────────────────────

SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "secret", "exposed", "scandal",
    "bombshell", "breaking", "urgent", "alert", "conspiracy",
    "hoax", "fraud", "coverup", "cover-up", "crisis", "outrage",
    "betrayal", "lies", "fake", "propaganda", "bribed", "rigged",
    "leaked", "banned", "censored", "silenced", "truth",
    "globalist", "deep state", "mainstream media", "establishment",
    "they don't want you to know", "you won't believe",
}

CREDIBILITY_WORDS = {
    "according", "stated", "reported", "confirmed", "official",
    "government", "university", "research", "study", "analysis",
    "spokesperson", "department", "authority", "committee",
    "published", "data", "evidence", "percent", "statistics",
    "representative", "source", "investigation", "review",
}

FORMAL_WORDS = {
    "furthermore", "however", "nevertheless", "subsequently",
    "consequently", "therefore", "whereas", "regarding",
    "pursuant", "legislative", "infrastructure", "policy",
    "administration", "amendment", "jurisdiction", "regulation",
}


def _find_overlap(text_lower: str, word_set: set) -> List[str]:
    """Return words from word_set that appear in text_lower."""
    found = []
    for w in word_set:
        if w in text_lower:
            found.append(w)
    return found


def explain_prediction_text(
    text: str,
    prediction: str,
    lime_top_fake: List[Tuple[str, float]],
    lime_top_real: List[Tuple[str, float]],
    confidence: float,
) -> str:
    """
    Generate a human-readable paragraph explaining WHY the model
    predicted FAKE or REAL.

    Args:
        text:           Original input text.
        prediction:     "FAKE" or "REAL".
        lime_top_fake:  Top LIME words contributing toward FAKE.
        lime_top_real:  Top LIME words contributing toward REAL.
        confidence:     Model confidence (0–1).

    Returns:
        A natural-language explanation string.
    """
    text_lower = text.lower()
    conf_pct   = f"{confidence * 100:.1f}%"

    sensational_found = _find_overlap(text_lower, SENSATIONAL_WORDS)
    credibility_found = _find_overlap(text_lower, CREDIBILITY_WORDS)
    formal_found      = _find_overlap(text_lower, FORMAL_WORDS)

    # Extract top contributing word labels from LIME
    lime_fake_words = [w for w, _ in lime_top_fake[:5]]
    lime_real_words = [w for w, _ in lime_top_real[:5]]

    if prediction == "FAKE":
        parts = [
            f"This article is predicted as **FAKE** with {conf_pct} confidence."
        ]

        if sensational_found:
            sample = "', '".join(sensational_found[:4])
            parts.append(
                f"It contains sensational or emotionally charged language such as '{sample}', "
                "which is commonly associated with misinformation."
            )

        if lime_fake_words:
            sample = "', '".join(lime_fake_words)
            parts.append(
                f"The most influential words driving the FAKE prediction are: '{sample}'."
            )

        if not credibility_found:
            parts.append(
                "The article lacks references to credible institutions, official sources, "
                "or verifiable data, which are hallmarks of reliable reporting."
            )

        if not formal_found:
            parts.append(
                "The writing style appears informal or emotionally driven rather than "
                "structured and neutral."
            )

        if confidence < 0.70:
            parts.append(
                "⚠️ Note: The model's confidence is relatively low — treat this prediction "
                "with caution and verify with additional sources."
            )

        return " ".join(parts)

    else:  # REAL
        parts = [
            f"This article is classified as **REAL** with {conf_pct} confidence."
        ]

        if credibility_found:
            sample = "', '".join(credibility_found[:4])
            parts.append(
                f"It references credible signals such as '{sample}', "
                "which are consistent with factual reporting."
            )

        if formal_found:
            sample = "', '".join(formal_found[:3])
            parts.append(
                f"The language is formal and structured (e.g., '{sample}'), "
                "suggesting professional journalism."
            )

        if lime_real_words:
            sample = "', '".join(lime_real_words)
            parts.append(
                f"Key terms reinforcing the REAL classification: '{sample}'."
            )

        if not sensational_found:
            parts.append(
                "The article avoids sensational or emotionally manipulative language, "
                "which is a positive credibility indicator."
            )

        if confidence < 0.70:
            parts.append(
                "⚠️ Note: Confidence is below 70% — the article may contain mixed signals. "
                "Cross-reference with trusted sources."
            )

        return " ".join(parts)


# ─────────────────────────────────────────────────────────────
# LIME explanation (basic model)
# ─────────────────────────────────────────────────────────────

def explain_basic_model(
    text: str,
    model,
    vectorizer,
    num_features: int = 15,
    num_samples:  int = 400,
) -> Dict:
    """
    Use LIME to explain a basic-model prediction.
    Falls back to coefficient-based explanation if LIME is unavailable.
    """
    try:
        from lime.lime_text import LimeTextExplainer
        from data_preprocessing import clean_text
    except ImportError:
        logger.warning("LIME not installed — using coefficient fallback.")
        return explain_with_coefficients(text, model, vectorizer)

    explainer = LimeTextExplainer(class_names=["FAKE", "REAL"], random_state=42)

    def _predict_fn(texts: List[str]) -> np.ndarray:
        from data_preprocessing import clean_text
        cleaned = [clean_text(t) for t in texts]
        X = vectorizer.transform(cleaned)
        return model.predict_proba(X)

    try:
        exp = explainer.explain_instance(
            text, _predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )
        return _parse_lime(text, exp)
    except Exception as e:
        logger.warning(f"LIME failed ({e}) — using coefficient fallback.")
        return explain_with_coefficients(text, model, vectorizer)


# ─────────────────────────────────────────────────────────────
# LIME explanation (RoBERTa)
# ─────────────────────────────────────────────────────────────

def explain_roberta(
    text: str,
    model,
    tokenizer,
    num_features: int = 12,
    num_samples:  int = 150,
) -> Dict:
    """
    LIME explanation for RoBERTa. Fewer samples due to inference cost.
    """
    try:
        import torch
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return _empty_explanation()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()

    explainer = LimeTextExplainer(class_names=["FAKE", "REAL"], random_state=42)

    def _predict_fn(texts: List[str]) -> np.ndarray:
        results = []
        for t in texts:
            inputs = tokenizer(
                t, return_tensors="pt", truncation=True,
                padding="max_length", max_length=128,
            ).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=-1)
                results.append(probs.squeeze().cpu().numpy())
        return np.array(results)

    try:
        exp = explainer.explain_instance(
            text, _predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )
        return _parse_lime(text, exp)
    except Exception as e:
        logger.warning(f"LIME/RoBERTa failed ({e}).")
        return _empty_explanation()


# ─────────────────────────────────────────────────────────────
# Coefficient-based fallback (no LIME)
# ─────────────────────────────────────────────────────────────

def explain_with_coefficients(text: str, model, vectorizer) -> Dict:
    """
    Fast explanation using LR coefficients × TF-IDF weights.
    No LIME needed; works as an instant fallback.
    """
    from data_preprocessing import clean_text

    cleaned = clean_text(text)
    X       = vectorizer.transform([cleaned])

    coefs  = model.coef_[0]                        # coefficient per feature
    fnames = vectorizer.get_feature_names_out()

    _, cols = X.nonzero()
    word_scores: Dict[str, float] = {}
    for idx in cols:
        word  = fnames[idx]
        score = float(coefs[idx] * X[0, idx])
        word_scores[word] = score

    return _build_output(text, word_scores)


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _parse_lime(text: str, explanation) -> Dict:
    word_weights = dict(explanation.as_list())
    return _build_output(text, word_weights)


def _build_output(text: str, word_scores: Dict[str, float]) -> Dict:
    tokens = re.findall(r"\S+|\s+", text)
    html_parts = []
    for token in tokens:
        key   = token.strip().lower()
        score = word_scores.get(key, 0.0)
        if score > 0.005:
            opacity = min(abs(score) * 5, 0.75)
            html_parts.append(
                f'<mark style="background:rgba(34,197,94,{opacity:.2f});'
                f'border-radius:4px;padding:1px 4px;font-weight:600;" '
                f'title="REAL +{score:.3f}">{token}</mark>'
            )
        elif score < -0.005:
            opacity = min(abs(score) * 5, 0.75)
            html_parts.append(
                f'<mark style="background:rgba(239,68,68,{opacity:.2f});'
                f'border-radius:4px;padding:1px 4px;font-weight:600;" '
                f'title="FAKE {score:.3f}">{token}</mark>'
            )
        else:
            html_parts.append(token)

    sorted_asc  = sorted(word_scores.items(), key=lambda x: x[1])
    top_fake = [(w, round(s, 4)) for w, s in sorted_asc[:8]    if s < 0]
    top_real = [(w, round(s, 4)) for w, s in sorted_asc[-8:]   if s > 0][::-1]

    return {
        "word_scores":    word_scores,
        "html_highlight": "".join(html_parts),
        "top_fake_words": top_fake,
        "top_real_words": top_real,
    }


def _empty_explanation() -> Dict:
    return {
        "word_scores":    {},
        "html_highlight": "<em>Explanation unavailable.</em>",
        "top_fake_words": [],
        "top_real_words": [],
    }
