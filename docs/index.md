# SEO Keyword Clustering

Interactive SEO keyword clustering, page mapping, and content-gap analysis with 2D/3D visualisations, CSV exports, and a Streamlit dashboard.

## What it does

- Clusters keywords by semantic similarity — **KMeans, Agglomerative, HDBSCAN, and graph-community (Louvain / Leiden)**
- Hybrid similarity scoring blending semantic embeddings, TF-IDF, and SERP-URL Jaccard overlap
- Maps keyword groups to your website pages via cosine similarity
- Detects content gaps — keywords with no suitable target page (fixed / percentile / adaptive thresholds)
- Flags cannibalisation across five distinct types (ranking conflict, mapping conflict, intent split, page mismatch, consolidation candidates)
- Scores opportunities with four profiles (`balanced`, `quick-wins`, `growth`, `commercial`)
- Auto-labels clusters from TF-IDF, BERTopic-style class TF-IDF, centroid keyword, or Carbonell-Goldstein MMR
- Run history + run comparison — every clustering run can be archived and diffed against another
- Connector imports for Google Search Console, GA4, Ahrefs, Semrush, Screaming Frog, Sitebulb
- Page enrichment crawler that pulls title/meta/h1/headings/body excerpts for richer page-mapping
- Generates interactive Plotly charts and exports full CSV + HTML reports
- Streamlit dashboard with manual cluster reassignment and merge

## Install

The project is split into optional extras so you can install only what you need.

```bash
git clone https://github.com/johnoconnor0/keyword-clustering.git
cd keyword-clustering

# Slim install (TF-IDF clustering, no UI):
pip install .

# Recommended install (UI + semantic embeddings + HDBSCAN/UMAP/hnswlib):
pip install .[app,semantic,advanced]

# Add tests + linting:
pip install .[app,semantic,advanced,dev]

python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Quickstart (CLI)

```bash
keyword-cluster run \
  --keywords examples/sample_keywords.csv \
  --pages    examples/sample_pages.csv \
  --topics   examples/sample_topics.csv \
  --clusters 8 \
  --output   outputs/
```

Run `keyword-cluster --help` and `keyword-cluster run --help` to see every available flag. Other subcommands:

| Subcommand                           | What it does                                                  |
|--------------------------------------|----------------------------------------------------------------|
| `keyword-cluster run`                | Cluster keywords and write the full report bundle              |
| `keyword-cluster compare`            | Diff two run-history folders                                   |
| `keyword-cluster tune`               | Bounded grid search over methods, k, embeddings, weights       |
| `keyword-cluster crawl`              | Crawl a Pages CSV and add title/meta/h1/headings/body_excerpt  |
| `keyword-cluster import-{gsc,ga4,ahrefs,semrush,screamingfrog,sitebulb}` | Normalise a connector export into the canonical schema |

## Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

## Docker

```bash
docker build -t keyword-clustering .
docker run -p 8501:8501 keyword-clustering
```
