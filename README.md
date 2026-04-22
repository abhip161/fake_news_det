# 🔍 TruthLens — Fake News Detection System

<div align="center">

![TruthLens Banner](https://img.shields.io/badge/TruthLens-Fake%20News%20Detector-6366f1?style=for-the-badge&logo=eye&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**An end-to-end, production-ready fake news classification system with explainable AI.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Usage](#-usage)
- [Models](#-models)
- [Explainability](#-explainability)
- [Advanced Features](#-advanced-features)
- [Future Improvements](#-future-improvements)

---

## 🌟 Overview

TruthLens is a full-stack fake news detection system that classifies news articles as **REAL** or **FAKE** using both traditional machine learning (TF-IDF + Logistic Regression) and state-of-the-art deep learning (BERT). The system includes a beautiful Streamlit web UI, explainable AI (LIME), live news scanning via NewsAPI, URL-based article extraction, multilingual support (Hindi), and a prediction history dashboard.

---

## ✨ Features

| Feature | Status |
|---------|--------|
| 📝 Text classification (Real/Fake) | ✅ |
| 🤖 Basic Model: TF-IDF + Logistic Regression | ✅ |
| 🧠 Advanced Model: BERT fine-tuned | ✅ |
| 🔍 LIME explainability (word highlighting) | ✅ |
| 📊 Confidence gauge + probability visualization | ✅ |
| 🔗 URL → article extraction → classify | ✅ |
| 📡 Live news scanning (NewsAPI) | ✅ |
| 🌐 Hindi → English translation | ✅ |
| 📜 Prediction history with CSV export | ✅ |
| 📊 Model comparison dashboard | ✅ |

---

## 🛠️ Tech Stack

**Backend / ML**
- Python 3.10+
- Scikit-learn (Logistic Regression, TF-IDF)
- HuggingFace Transformers (BERT)
- PyTorch
- NLTK (text preprocessing)
- LIME (explainability)
- Newspaper3k (article extraction)

**Frontend**
- Streamlit
- Plotly (interactive charts)

**Data**
- [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [NewsAPI](https://newsapi.org/) (live headlines)

---

## 📂 Project Structure

```
fake-news-detector/
│
├── data/
│   ├── raw/                    # Fake.csv, True.csv (download from Kaggle)
│   ├── processed/              # Cleaned, merged dataset
│   └── prediction_history.json # Saved prediction log
│
├── models/
│   ├── basic_model.pkl         # Trained Logistic Regression
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
│   ├── basic_metrics.json      # LR evaluation metrics
│   ├── bert_metrics.json       # BERT evaluation metrics
│   └── bert_model/             # Fine-tuned BERT weights
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── src/
│   ├── utils.py                # Shared helpers, I/O, logging
│   ├── data_preprocessing.py  # Text cleaning, train/test split
│   ├── feature_engineering.py # TF-IDF vectorization
│   ├── train_basic.py          # LR training pipeline
│   ├── train_bert.py           # BERT fine-tuning pipeline
│   ├── evaluate.py             # Metrics, plots, comparison
│   └── explainability.py       # LIME explanations
│
├── app/
│   └── streamlit_app.py        # Main Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate.bat     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Download dataset

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Download and extract `Fake.csv` and `True.csv`
3. Place them in `data/raw/`

```
data/raw/
├── Fake.csv
└── True.csv
```

### 5. Train the Basic Model

```bash
cd src
python train_basic.py
```

Training takes ~2–5 minutes on a standard laptop.

### 6. (Optional) Fine-tune BERT

> ⚠️ GPU strongly recommended. Takes ~1–3 hours on GPU, much longer on CPU.

```bash
python src/train_bert.py
```

### 7. Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Usage

### Classify Text
1. Navigate to the **Classify Text** tab
2. Paste your news article
3. Select a model from the sidebar
4. Click **Analyze**
5. View prediction, confidence gauge, and probability breakdown

### Classify from URL
1. Go to the **URL Input** tab
2. Paste a news article URL
3. The app extracts the article and classifies it automatically

### Live News Scanning
1. Enter your [NewsAPI key](https://newsapi.org/) in the sidebar
2. Set a search query
3. Go to **Live News** tab and click **Fetch & Scan News**

### Explainability
- Check **Show Explanation (LIME)** before clicking Analyze
- The app highlights words that pushed the model toward FAKE or REAL
- Red = FAKE signal | Green = REAL signal

### Hindi Support
- Select **Hindi (हिंदी)** from the Language dropdown
- Paste Hindi text — it will be auto-translated before classification

---

## 🤖 Models

### Basic Model: TF-IDF + Logistic Regression

| Metric | Score |
|--------|-------|
| Accuracy | ~98.5% |
| Precision | ~98.4% |
| Recall | ~98.6% |
| F1-Score | ~98.5% |

### BERT Model (fine-tuned)

| Metric | Score |
|--------|-------|
| Accuracy | ~99.2% |
| Precision | ~99.1% |
| Recall | ~99.3% |
| F1-Score | ~99.2% |

> *Scores are approximate and depend on dataset and training run.*

---

## 🧠 Explainability

TruthLens uses **LIME (Local Interpretable Model-agnostic Explanations)** to explain individual predictions:

- Words highlighted in **green** push the model toward REAL
- Words highlighted in **red** push the model toward FAKE
- The sidebar shows the top contributing words for each class
- For faster results, a coefficient-based fallback is also available

---

## 🔥 Advanced Features

1. **URL Article Extraction** — `newspaper3k` fetches and parses full articles from any URL
2. **Live News Feed** — Scans real-time headlines from NewsAPI and classifies each
3. **Model Comparison Dashboard** — Side-by-side bar charts for all metrics
4. **Prediction History** — Full log with timestamps, exportable as CSV
5. **Multilingual Support** — Hindi → English via `deep-translator`
6. **Probability Visualization** — Stacked bar + gauge chart for confidence

---

## 🚀 Deployment

### Streamlit Cloud

1. Push repo to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set **app/streamlit_app.py** as entry point
4. Deploy!

### Render / Railway

```bash
# Procfile
web: streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🔮 Future Improvements

- [ ] Add SHAP explainability alongside LIME
- [ ] Implement RoBERTa / DeBERTa for higher accuracy
- [ ] Integrate fact-checking APIs (Google Fact Check)
- [ ] Browser extension for real-time detection
- [ ] REST API endpoint (FastAPI)
- [ ] Docker container for one-click deployment
- [ ] Support more languages beyond Hindi

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Kaggle Dataset by Clément Bisaillon](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [LIME by Marco Ribeiro](https://github.com/marcotcr/lime)
- [Streamlit](https://streamlit.io/)
#