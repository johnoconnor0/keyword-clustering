# Development Guide

## Setup

```bash
git clone https://github.com/johnoconnor0/keyword-clustering.git
cd keyword-clustering
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

## Running tests

```bash
pytest tests/ -v
```

Tests run across Python 3.10, 3.11, and 3.12 in GitHub Actions CI.

## Module overview

| Module | Responsibility |
|---|---|
| `preprocessing.py` | Text normalisation, CSV loading, intent classification, brand flagging, SERP tag parsing |
| `vectorization.py` | TF-IDF vectorization (with preprocessing), Sentence Transformer encoding |
| `clustering.py` | KMeans, Agglomerative, HDBSCAN; PCA, UMAP, t-SNE reduction |
| `scoring.py` | Cosine similarity scoring, page mapping, opportunity scoring, notes generation, gap/cannibalization detection |
| `labeling.py` | TF-IDF cluster label extraction from original keywords |
| `visualization.py` | All Plotly chart functions (`plot_3d_clusters`, `plot_treemap`, etc.) |
| `export.py` | CSV and HTML report writers (`save_all_outputs`, `build_interactive_report`) |
| `cli.py` | `argparse` entry point wiring all modules together |

## Adding a new clustering method

1. Add a branch in `cluster_keywords()` in `clustering.py`.
2. Update the `choices` list in `cli.py` `--method` argument.
3. Add a test in `tests/test_clustering.py`.

## Adding a new chart

1. Add a `plot_*()` function in `visualization.py`.
2. Call it inside `save_all_charts()` in `visualization.py`.
3. Add its filename to `build_interactive_report()` in `export.py`.

## Docs site

```bash
pip install mkdocs mkdocs-material
mkdocs serve        # local preview at http://127.0.0.1:8000
mkdocs build        # build static site into site/
mkdocs gh-deploy    # deploy to GitHub Pages
```
