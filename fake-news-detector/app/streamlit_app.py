"""
streamlit_app.py — TruthLens Premium UI (upgraded).

Features:
  • Premium dark gradient design with animated hero
  • Text input + URL extraction + Live NewsAPI feed
  • Basic (TF-IDF + LR) and RoBERTa model selection
  • LIME word highlighting + human-readable WHY explanation
  • Low-confidence warning (<70%)
  • Confidence gauge + probability stacked bar
  • Model comparison dashboard (TF-IDF vs RoBERTa)
  • Prediction history with CSV export
  • Hindi → English translation
"""

import sys
import os
import json
import time
import requests
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils import (
    load_model, save_prediction, load_history, clear_history,
    decode_label, label_color, is_valid_input, truncate_words,
    MODELS_DIR,
)
from data_preprocessing import clean_text
from evaluate import load_basic_metrics, load_roberta_metrics, get_comparison_table
from explainability import (
    explain_basic_model, explain_with_coefficients,
    explain_roberta, explain_prediction_text,
)

# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "TruthLens — Fake News Detector",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────
# Global CSS — premium dark theme
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ──────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background: #08080f;
  color: #d4d4e8;
}
h1,h2,h3 { font-family:'Syne',sans-serif; }

/* ── Hide Streamlit chrome ──────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0d0d1a 0%,#111127 100%);
  border-right: 1px solid #1e1e35;
}
[data-testid="stSidebar"] * { color: #b0b0cc !important; }

/* ── HERO ───────────────────────────────────── */
.hero {
  background: linear-gradient(135deg,#0d1b3e 0%,#111827 40%,#1a0d3e 100%);
  border: 1px solid rgba(99,102,241,.3);
  border-radius: 20px;
  padding: 2.5rem 3rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
  animation: fadeSlide .6s ease both;
}
.hero::before {
  content:'';position:absolute;top:-30%;right:-5%;
  width:500px;height:500px;
  background:radial-gradient(circle,rgba(139,92,246,.12) 0%,transparent 70%);
  pointer-events:none;
}
.hero::after {
  content:'';position:absolute;bottom:-40%;left:-5%;
  width:400px;height:400px;
  background:radial-gradient(circle,rgba(59,130,246,.08) 0%,transparent 70%);
  pointer-events:none;
}
@keyframes fadeSlide {
  from {opacity:0;transform:translateY(-12px);}
  to   {opacity:1;transform:translateY(0);}
}
.hero-badge {
  display:inline-block;
  background:rgba(99,102,241,.15);
  border:1px solid rgba(99,102,241,.4);
  color:#a5b4fc;
  border-radius:999px;
  padding:3px 14px;
  font-size:.78rem;
  font-weight:600;
  letter-spacing:.08em;
  text-transform:uppercase;
  margin-bottom:1rem;
}
.hero-title {
  font-family:'Syne',sans-serif;
  font-size:3rem;
  font-weight:800;
  color:#fff;
  margin:0 0 .5rem;
  line-height:1.1;
  background:linear-gradient(90deg,#e0e7ff,#c7d2fe,#a5b4fc);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
}
.hero-sub { color:#8080a0; font-size:1rem; max-width:600px; line-height:1.6; }

/* ── Cards ──────────────────────────────────── */
.card {
  background:#0f0f1e;
  border:1px solid #1e1e35;
  border-radius:14px;
  padding:1.5rem;
  margin-bottom:1rem;
  transition:border-color .2s;
}
.card:hover { border-color:#3333aa; }

/* ── Result cards ───────────────────────────── */
.result-card {
  border-radius:16px;
  padding:2rem;
  text-align:center;
  margin:1rem 0;
  animation:pop .4s cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes pop {
  from {opacity:0;transform:scale(.92);}
  to   {opacity:1;transform:scale(1);}
}
.fake-card {
  background:linear-gradient(135deg,#2d0a0a,#4a1010);
  border:1.5px solid #ef4444;
  box-shadow:0 0 40px rgba(239,68,68,.15);
}
.real-card {
  background:linear-gradient(135deg,#0a2d12,#104a1c);
  border:1.5px solid #22c55e;
  box-shadow:0 0 40px rgba(34,197,94,.15);
}
.result-emoji { font-size:3.5rem; display:block; margin-bottom:.3rem; }
.result-label {
  font-family:'Syne',sans-serif;
  font-size:2.8rem;
  font-weight:800;
  margin:0;
  letter-spacing:.05em;
}
.result-conf { font-size:1rem; color:#aaa; margin-top:.4rem; }
.result-model { font-size:.82rem; color:#666; margin-top:.2rem; }

/* ── Warning banner ─────────────────────────── */
.warn-banner {
  background:rgba(234,179,8,.1);
  border:1px solid rgba(234,179,8,.4);
  border-radius:10px;
  padding:.8rem 1.2rem;
  color:#fde047;
  font-size:.9rem;
  margin:.8rem 0;
}

/* ── Explanation box ────────────────────────── */
.explain-box {
  background:#0f0f1e;
  border:1px solid #252540;
  border-radius:12px;
  padding:1.2rem 1.5rem;
  line-height:1.9;
  font-size:.93rem;
  color:#c0c0d8;
}
.human-explain {
  background:linear-gradient(135deg,#0f1929,#111827);
  border-left:3px solid #6366f1;
  border-radius:8px;
  padding:1rem 1.2rem;
  font-size:.92rem;
  color:#b0bcd8;
  line-height:1.7;
  margin:1rem 0;
}

/* ── Word chips ─────────────────────────────── */
.chip {
  display:inline-block;
  padding:3px 12px;
  border-radius:999px;
  font-size:.83rem;
  font-weight:600;
  margin:2px 3px;
}
.chip-fake { background:rgba(239,68,68,.15); color:#fca5a5; border:1px solid rgba(239,68,68,.4); }
.chip-real { background:rgba(34,197,94,.15);  color:#86efac; border:1px solid rgba(34,197,94,.4); }

/* ── Metric box ─────────────────────────────── */
.metric-box {
  background:#0f0f1e;
  border:1px solid #1e1e35;
  border-radius:12px;
  padding:1.2rem 1rem;
  text-align:center;
  transition:border-color .2s, transform .2s;
}
.metric-box:hover { border-color:#6366f1; transform:translateY(-2px); }
.metric-val { font-size:2rem; font-weight:700; color:#818cf8; font-family:'Syne',sans-serif; }
.metric-lbl { font-size:.78rem; color:#555; margin-top:.3rem; text-transform:uppercase; letter-spacing:.06em; }

/* ── News card ──────────────────────────────── */
.news-card {
  background:#0f0f1e;
  border:1px solid #1e1e35;
  border-radius:10px;
  padding:1rem 1.2rem;
  margin-bottom:.7rem;
  transition:border-color .2s;
}
.news-card:hover { border-color:#6366f1; }

/* ── History row ────────────────────────────── */
.hist-row {
  background:#0f0f1e;
  border:1px solid #1a1a2e;
  border-radius:8px;
  padding:.65rem 1rem;
  margin-bottom:.4rem;
  display:flex;
  align-items:center;
  gap:.8rem;
  font-size:.84rem;
}

/* ── Buttons ────────────────────────────────── */
.stButton > button {
  background:linear-gradient(135deg,#4f46e5,#7c3aed);
  color:#fff;
  border:none;
  border-radius:10px;
  padding:.65rem 2rem;
  font-weight:600;
  font-size:1rem;
  transition:opacity .2s, transform .15s;
  font-family:'Inter',sans-serif;
}
.stButton > button:hover { opacity:.88; transform:translateY(-1px); }

/* ── Tab styling ────────────────────────────── */
.stTabs [data-baseweb="tab"] {
  font-family:'Inter',sans-serif;
  font-size:.93rem;
  font-weight:500;
  color:#888;
}
.stTabs [aria-selected="true"] { color:#a5b4fc !important; }

/* ── Inputs ─────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
  background:#0f0f1e !important;
  border:1px solid #252540 !important;
  border-radius:10px !important;
  color:#d4d4e8 !important;
  font-family:'Inter',sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color:#6366f1 !important;
  box-shadow:0 0 0 2px rgba(99,102,241,.2) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────

_DEFAULTS = {
    "last_result":      None,
    "last_explanation": None,
    "last_human_exp":   "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────
# Cached model loaders
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_basic():
    try:
        m = load_model("basic_model.pkl")
        v = load_model("tfidf_vectorizer.pkl")
        return m, v
    except FileNotFoundError:
        return None, None


@st.cache_resource(show_spinner=False)
def _load_roberta():
    rob_dir = MODELS_DIR / "roberta_model"
    if not rob_dir.exists():
        return None, None
    try:
        from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
        m = RobertaForSequenceClassification.from_pretrained(str(rob_dir))
        t = RobertaTokenizerFast.from_pretrained(str(rob_dir))
        m.eval()
        return m, t
    except Exception as e:
        return None, None


# ─────────────────────────────────────────────────────────────
# Prediction logic
# ─────────────────────────────────────────────────────────────

def run_prediction(text: str, model_choice: str) -> Optional[Dict]:
    if "Basic" in model_choice:
        model, vec = _load_basic()
        if model is None:
            st.error("Basic model not found. Run `python src/train_basic.py`.")
            return None
        cleaned = clean_text(text)
        X       = vec.transform([cleaned])
        pred    = model.predict(X)[0]
        proba   = model.predict_proba(X)[0]
        label   = decode_label(pred)
        return {
            "label":      label,
            "confidence": float(proba[pred]),
            "prob_fake":  float(proba[0]),
            "prob_real":  float(proba[1]),
            "model_name": "Basic Model (TF-IDF + LR)",
        }
    else:
        import torch, numpy as np
        model, tok = _load_roberta()
        if model is None:
            st.error("RoBERTa model not found. Run `python src/train_roberta.py`.")
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device)
        inputs = tok(
            truncate_words(text, 400), return_tensors="pt",
            truncation=True, padding="max_length", max_length=256,
        ).to(device)
        with torch.no_grad():
            proba = torch.softmax(model(**inputs).logits, dim=-1).squeeze().cpu().numpy()
        idx = int(np.argmax(proba))
        return {
            "label":      "FAKE" if idx == 0 else "REAL",
            "confidence": float(proba[idx]),
            "prob_fake":  float(proba[0]),
            "prob_real":  float(proba[1]),
            "model_name": "RoBERTa (roberta-base)",
        }


def run_explanation(text: str, model_choice: str) -> Optional[Dict]:
    model, vec = _load_basic()
    if model is None:
        return None
    if "Basic" in model_choice:
        try:
            return explain_basic_model(text, model, vec, num_features=12)
        except Exception:
            return explain_with_coefficients(text, model, vec)
    else:
        rob_model, rob_tok = _load_roberta()
        if rob_model:
            return explain_roberta(text, rob_model, rob_tok, num_samples=100)
        return explain_with_coefficients(text, model, vec)


# ─────────────────────────────────────────────────────────────
# URL extraction
# ─────────────────────────────────────────────────────────────

def extract_from_url(url: str) -> Optional[str]:
    try:
        from newspaper import Article
        art = Article(url)
        art.download(); art.parse()
        return (art.title or "") + "\n\n" + (art.text or "")
    except Exception as e:
        st.error(f"Could not extract article: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# NewsAPI
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_news(api_key: str, query: str, n: int = 6) -> list:
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "pageSize": n, "sortBy": "publishedAt",
                    "language": "en", "apiKey": api_key},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# Hindi translation
# ─────────────────────────────────────────────────────────────

def translate_hi_en(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="hi", target="en").translate(text)
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text


# ─────────────────────────────────────────────────────────────
# Plotly charts
# ─────────────────────────────────────────────────────────────

def gauge_chart(conf: float, label: str) -> go.Figure:
    color = "#22c55e" if label == "REAL" else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(conf * 100, 1),
        title = {"text": f"Confidence", "font": {"size": 14, "color": "#888"}},
        number= {"suffix": "%", "font": {"size": 40, "color": color}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "#333", "tickfont": {"color": "#555"}},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "#0a0a18",
            "bordercolor": "#1e1e35",
            "steps": [
                {"range": [0,  50], "color": "#1a0a0a"},
                {"range": [50, 70], "color": "#1a1a0a"},
                {"range": [70,100], "color": "#0a1a10"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": .8, "value": round(conf*100,1)},
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
    )
    return fig


def prob_bar(pf: float, pr: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[round(pf*100,1)], name="FAKE", orientation="h",
                         marker_color="#ef4444", text=[f"FAKE  {pf*100:.1f}%"],
                         textposition="inside", insidetextanchor="middle"))
    fig.add_trace(go.Bar(x=[round(pr*100,1)], name="REAL", orientation="h",
                         marker_color="#22c55e", text=[f"REAL  {pr*100:.1f}%"],
                         textposition="inside", insidetextanchor="middle"))
    fig.update_layout(
        barmode="stack", height=70,
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={"range":[0,100],"showticklabels":False,"showgrid":False},
        yaxis={"showticklabels":False,"showgrid":False},
        font={"color":"#ccc","size":13},
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem;">
      <div style="font-size:2.8rem;">🔍</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.5rem;
                  color:#e0e0f8;font-weight:800;margin-top:.3rem;">TruthLens</div>
      <div style="font-size:.72rem;color:#444;letter-spacing:.1em;
                  text-transform:uppercase;margin-top:.2rem;">Fake News Detector v2</div>
    </div>
    <hr style="border:none;border-top:1px solid #1e1e35;margin:1rem 0;">
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "🤖 Model",
        ["Basic Model (TF-IDF + LR)", "RoBERTa (roberta-base)"],
        help="Basic is fast; RoBERTa is more accurate but requires GPU for fast inference.",
    )

    lang_choice = st.selectbox("🌐 Input Language", ["English", "Hindi (हिंदी)"])

    st.markdown("<hr style='border:none;border-top:1px solid #1e1e35;'>", unsafe_allow_html=True)

    news_api_key = st.text_input(
        "📡 NewsAPI Key", type="password", placeholder="newsapi.org key…"
    )
    news_query = st.text_input("News query", value="technology AI")

    st.markdown("<hr style='border:none;border-top:1px solid #1e1e35;'>", unsafe_allow_html=True)

    # Model status
    st.markdown("**Model Status**")
    bm, bv = _load_basic()
    st.markdown(
        f"{'🟢' if bm else '🔴'} Basic Model — {'ready' if bm else 'not trained'}"
    )
    rm, rt = _load_roberta()
    st.markdown(
        f"{'🟢' if rm else '🟡'} RoBERTa — {'ready' if rm else 'not trained'}"
    )

    st.markdown("<hr style='border:none;border-top:1px solid #1e1e35;'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear History", use_container_width=True):
        clear_history()
        st.success("Cleared.")


# ─────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-badge">AI · NLP · Explainability</div>
  <div class="hero-title">TruthLens</div>
  <div class="hero-sub">
    Detect fake news with explainable AI. Powered by TF-IDF + Logistic Regression
    and fine-tuned RoBERTa — with LIME word-level explanations and human-readable reasoning.
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────

tab_text, tab_url, tab_live, tab_compare, tab_history = st.tabs([
    "📝 Classify Text", "🔗 URL", "📡 Live News", "📊 Comparison", "📜 History"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Classify Text
# ══════════════════════════════════════════════════════════════

with tab_text:
    st.markdown("### Paste your article or headline")

    user_text = st.text_area(
        label="",
        placeholder=(
            "Enter a full news article or headline here…\n\n"
            "Tip: longer text gives more accurate and explainable results."
        ),
        height=220,
        key="main_input",
    )

    col_btn, col_exp = st.columns([1, 2])
    with col_btn:
        predict_btn = st.button("🔍 Analyze Article", use_container_width=True)
    with col_exp:
        show_explain = st.checkbox("🧠 Show LIME Explanation", value=True)

    if predict_btn:
        valid, err = is_valid_input(user_text)
        if not valid:
            st.warning(f"⚠️ {err}")
        else:
            classify_text = user_text

            # Translate Hindi if needed
            if lang_choice == "Hindi (हिंदी)":
                with st.spinner("Translating Hindi → English…"):
                    classify_text = translate_hi_en(user_text)
                st.info(f"**Translated text (preview):** {classify_text[:300]}…")

            with st.spinner("🔍 Analyzing article…"):
                result = run_prediction(classify_text, model_choice)

            if result:
                st.session_state["last_result"] = result
                label = result["label"]
                conf  = result["confidence"]

                # ── Low-confidence warning ────────────────────
                if conf < 0.70:
                    st.markdown("""
                    <div class="warn-banner">
                      ⚠️ <strong>Low confidence ({:.0f}%)</strong> — The model is uncertain.
                      Cross-reference this result with multiple trusted sources before drawing conclusions.
                    </div>
                    """.format(conf * 100), unsafe_allow_html=True)

                # ── Result card ───────────────────────────────
                card_cls = "fake-card" if label == "FAKE" else "real-card"
                emoji    = "🚫" if label == "FAKE" else "✅"

                st.markdown(f"""
                <div class="result-card {card_cls}">
                  <span class="result-emoji">{emoji}</span>
                  <div class="result-label">{label}</div>
                  <div class="result-conf">Confidence: <strong>{conf*100:.1f}%</strong></div>
                  <div class="result-model">Model: {result['model_name']}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Charts ────────────────────────────────────
                col_g, col_p = st.columns([1, 1])
                with col_g:
                    st.plotly_chart(gauge_chart(conf, label), use_container_width=True)
                with col_p:
                    st.markdown("#### Probability Split")
                    st.plotly_chart(
                        prob_bar(result["prob_fake"], result["prob_real"]),
                        use_container_width=True,
                    )
                    st.markdown(f"""
| Class | Probability |
|-------|------------|
| 🚫 FAKE | {result['prob_fake']*100:.2f}% |
| ✅ REAL | {result['prob_real']*100:.2f}% |
""")

                # ── LIME Explanation ──────────────────────────
                exp = None
                if show_explain:
                    with st.spinner("🧠 Computing LIME explanation…"):
                        exp = run_explanation(classify_text, model_choice)

                    if exp:
                        st.session_state["last_explanation"] = exp

                        st.markdown("---")
                        st.markdown("### 🧠 Explainability")

                        # Human-readable WHY
                        human_exp = explain_prediction_text(
                            text        = classify_text,
                            prediction  = label,
                            lime_top_fake = exp["top_fake_words"],
                            lime_top_real = exp["top_real_words"],
                            confidence  = conf,
                        )
                        st.session_state["last_human_exp"] = human_exp

                        st.markdown(
                            f'<div class="human-explain">💬 {human_exp}</div>',
                            unsafe_allow_html=True,
                        )

                        # Highlighted text
                        st.markdown("**Word-level influence** *(hover for score)*")
                        st.markdown(
                            f'<div class="explain-box">{exp["html_highlight"]}</div>',
                            unsafe_allow_html=True,
                        )

                        # Word chips
                        col_f, col_r = st.columns(2)
                        with col_f:
                            st.markdown("**🚫 FAKE-signal words**")
                            chips = " ".join(
                                f'<span class="chip chip-fake">{w}</span>'
                                for w, _ in exp["top_fake_words"]
                            )
                            st.markdown(chips or "_None found_", unsafe_allow_html=True)
                        with col_r:
                            st.markdown("**✅ REAL-signal words**")
                            chips = " ".join(
                                f'<span class="chip chip-real">{w}</span>'
                                for w, _ in exp["top_real_words"]
                            )
                            st.markdown(chips or "_None found_", unsafe_allow_html=True)

                # ── Save to history ───────────────────────────
                top_words = []
                if exp:
                    top_words = (
                        [w for w, _ in exp["top_fake_words"][:3]]
                        + [w for w, _ in exp["top_real_words"][:3]]
                    )
                save_prediction(
                    text              = classify_text,
                    model_name        = result["model_name"],
                    prediction        = label,
                    confidence        = conf,
                    top_words         = top_words,
                    human_explanation = st.session_state.get("last_human_exp", ""),
                )


# ══════════════════════════════════════════════════════════════
# TAB 2 — URL Input
# ══════════════════════════════════════════════════════════════

with tab_url:
    st.markdown("### Classify a news article from a URL")

    url_in = st.text_input(
        "🔗 Article URL",
        placeholder="https://www.bbc.com/news/…",
        key="url_input",
    )

    if st.button("📥 Fetch & Classify", key="url_btn"):
        if not url_in.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching article…"):
                article = extract_from_url(url_in)

            if article:
                st.success("✅ Article extracted.")
                with st.expander("📄 Article preview"):
                    st.write(article[:1200] + "…")

                with st.spinner("🔍 Classifying…"):
                    res = run_prediction(article, model_choice)

                if res:
                    label    = res["label"]
                    card_cls = "fake-card" if label == "FAKE" else "real-card"
                    emoji    = "🚫" if label == "FAKE" else "✅"

                    st.markdown(f"""
                    <div class="result-card {card_cls}">
                      <span class="result-emoji">{emoji}</span>
                      <div class="result-label">{label}</div>
                      <div class="result-conf">Confidence: {res['confidence']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if res["confidence"] < 0.70:
                        st.markdown("""
                        <div class="warn-banner">⚠️ <strong>Low confidence</strong> — verify with additional sources.</div>
                        """, unsafe_allow_html=True)

                    st.plotly_chart(
                        prob_bar(res["prob_fake"], res["prob_real"]),
                        use_container_width=True,
                    )
                    save_prediction(article, res["model_name"], label, res["confidence"])


# ══════════════════════════════════════════════════════════════
# TAB 3 — Live News
# ══════════════════════════════════════════════════════════════

with tab_live:
    st.markdown("### 📡 Scan Live Headlines")
    st.info("Enter your [NewsAPI](https://newsapi.org/) key in the sidebar, then click Fetch.")

    if st.button("🔄 Fetch & Classify Headlines"):
        if not news_api_key:
            st.warning("NewsAPI key required — add it in the sidebar.")
        else:
            with st.spinner("Fetching live news…"):
                articles = fetch_news(news_api_key, news_query)

            if not articles:
                st.error("No articles returned. Check your API key or query.")
            else:
                basic_model, basic_vec = _load_basic()
                for art in articles:
                    title   = art.get("title", "") or ""
                    body    = art.get("content", "") or art.get("description", "") or ""
                    full    = (title + " " + body).strip()

                    if basic_model and full:
                        res = run_prediction(full, "Basic Model (TF-IDF + LR)")
                    else:
                        res = None

                    l   = res["label"]    if res else "UNKNOWN"
                    c   = res["confidence"] if res else 0.0
                    col = label_color(l)
                    src = art.get("source", {}).get("name", "?")
                    dt  = (art.get("publishedAt", "") or "")[:10]
                    url = art.get("url", "#")

                    st.markdown(f"""
                    <div class="news-card">
                      <div style="font-weight:600;color:#d0d0e8;">
                        <span style="color:{col};font-weight:700;">[{l} {c*100:.0f}%]</span>
                        &nbsp;{title}
                      </div>
                      <div style="font-size:.78rem;color:#555;margin-top:.4rem;">
                        📰 {src} &nbsp;|&nbsp; 📅 {dt} &nbsp;|&nbsp;
                        <a href="{url}" target="_blank" style="color:#6366f1;">Read full article →</a>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — Model Comparison
# ══════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown("### 📊 Model Comparison Dashboard")

    df = get_comparison_table()
    if df.empty:
        st.info(
            "No trained models found yet.\n\n"
            "**Train models:**\n"
            "```\npython src/train_basic.py\npython src/train_roberta.py\n```"
        )
    else:
        METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1-Score"]

        # Per-model metric cards
        for _, row in df.iterrows():
            st.markdown(f"**{row['Model']}**")
            cols = st.columns(4)
            for i, m in enumerate(METRIC_COLS):
                val = row[m]
                try:
                    display = f"{float(val)*100:.1f}%"
                except (ValueError, TypeError):
                    display = str(val)
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-box">
                      <div class="metric-val">{display}</div>
                      <div class="metric-lbl">{m}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("")

        # Bar chart
        st.markdown("#### Side-by-Side Bar Chart")
        plot_rows = []
        for _, row in df.iterrows():
            for m in METRIC_COLS:
                try:
                    plot_rows.append({"Model": row["Model"], "Metric": m, "Score": float(row[m])})
                except (ValueError, TypeError):
                    pass

        if plot_rows:
            fig = px.bar(
                pd.DataFrame(plot_rows), x="Metric", y="Score", color="Model",
                barmode="group", text_auto=".3f",
                color_discrete_sequence=["#6366f1", "#22c55e"],
                template="plotly_dark",
            )
            fig.update_layout(
                yaxis_range=[0, 1.05],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Raw Metrics Table")
        st.dataframe(df.set_index("Model"), use_container_width=True)

    # CV info for basic model
    basic_m = load_basic_metrics()
    if basic_m and "cv_f1" in basic_m:
        st.markdown("---")
        st.markdown(
            f"**Basic model — 5-fold CV F1:** `{basic_m['cv_f1']}` "
            f"| **CV Accuracy:** `{basic_m.get('cv_accuracy','—')}`"
        )


# ══════════════════════════════════════════════════════════════
# TAB 5 — History
# ══════════════════════════════════════════════════════════════

with tab_history:
    st.markdown("### 📜 Prediction History")
    history = load_history()

    if not history:
        st.info("No predictions yet. Classify some articles to see history here.")
    else:
        hdf = pd.DataFrame(history[::-1])

        total    = len(hdf)
        n_fake   = int((hdf["prediction"] == "FAKE").sum())
        n_real   = int((hdf["prediction"] == "REAL").sum())
        avg_conf = float(hdf["confidence"].mean())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{total}</div><div class="metric-lbl">Total</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#ef4444">{n_fake}</div><div class="metric-lbl">FAKE</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#22c55e">{n_real}</div><div class="metric-lbl">REAL</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_conf*100:.0f}%</div><div class="metric-lbl">Avg Conf</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # Pie chart
        fig_pie = px.pie(
            values=[n_fake, n_real], names=["FAKE", "REAL"],
            color_discrete_sequence=["#ef4444", "#22c55e"],
            template="plotly_dark",
            title="Distribution",
            hole=0.45,
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40,b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

        # History rows
        st.markdown("#### Recent Predictions")
        for rec in history[::-1][:25]:
            lbl    = rec.get("prediction", "?")
            conf   = rec.get("confidence", 0)
            mdl    = rec.get("model", "?")
            ts     = rec.get("timestamp", "")[:16].replace("T", " ")
            prev   = rec.get("text_preview", "")[:80]
            col    = label_color(lbl)
            hexp   = rec.get("human_explanation", "")

            st.markdown(f"""
            <div class="hist-row">
              <span style="color:{col};font-weight:700;min-width:50px;">{lbl}</span>
              <span style="flex:1;color:#9090b0;">{prev}…</span>
              <span style="color:#444;font-size:.78rem;white-space:nowrap;">{conf*100:.0f}% | {mdl.split('(')[0].strip()} | {ts}</span>
            </div>
            """, unsafe_allow_html=True)

            if hexp:
                with st.expander("💬 Why?"):
                    st.markdown(hexp)

        # CSV export
        csv = hdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export CSV", csv, "truthlens_history.csv", "text/csv",
            use_container_width=True,
        )
