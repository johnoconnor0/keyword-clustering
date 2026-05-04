# SEO Keyword Clustering

Interactive SEO keyword clustering, page mapping, and content-gap analysis with 2D/3D visualisations, CSV exports, and a Streamlit dashboard.

## What it does

- Clusters keywords by semantic similarity (KMeans, Agglomerative, HDBSCAN)
- Maps keyword groups to your website pages via cosine similarity
- Detects content gaps — keywords with no suitable target page
- Flags cannibalization — clusters where multiple pages compete for the same intent
- Flags page mismatches from `current_url` vs the recommended page
- Scores opportunities — prioritises by volume, difficulty, and ranking gap
- Auto-labels clusters from top TF-IDF terms
- Supports SERP feature columns (featured snippet, local pack, PAA, image pack, video)
- Generates interactive Plotly charts and exports full CSV + HTML reports

## Quickstart

```bash
git clone https://github.com/johnoconnor0/keyword-clustering.git
cd keyword-clustering
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

python -m keyword_clustering.cli run \
  --keywords examples/sample_keywords.csv \
  --pages    examples/sample_pages.csv \
  --topics   examples/sample_topics.csv \
  --clusters 8 \
  --output   outputs/
```

## Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

## Docker

```bash
docker build -t keyword-clustering .
docker run -p 8501:8501 keyword-clustering
```
