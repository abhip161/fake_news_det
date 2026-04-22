#!/usr/bin/env bash
# setup.sh — TruthLens v2 environment bootstrap
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍  TruthLens v2 — Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ ! -d venv ] && python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt -q

python3 - <<'EOF'
import nltk
for pkg in ['stopwords','wordnet','punkt','omw-1.4']:
    nltk.download(pkg, quiet=True)
print("NLTK assets ready.")
EOF

mkdir -p data/raw data/processed models/roberta_model

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  Setup complete!"
echo ""
echo "  1. Place Fake.csv + True.csv in  data/raw/"
echo "  2. python src/train_basic.py"
echo "  3. streamlit run app/streamlit_app.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
