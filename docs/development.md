# Development Guide

## Setup

```bash
git clone https://github.com/johnoconnor0/keyword-clustering.git
cd keyword-clustering
pip install .[app,semantic,advanced,dev]
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Running tests

```bash
pytest tests/ -v --cov=keyword_clustering --cov-report=term-missing
```

CI runs three jobs against Python 3.10 / 3.11 / 3.12:
1. **Full extras** with `--cov-fail-under=75`.
2. **Slim install** (`pip install .`) running the TF-IDF-only test subset.
3. **End-to-end CLI smoke** that runs `keyword-cluster run` against the bundled `examples/sample_*.csv`.

## Linting & typing

```bash
ruff check .
ruff format --check .
mypy keyword_clustering
```

`pre-commit` is wired up via `.pre-commit-config.yaml` — install it once locally with `pre-commit install` and the
ruff + mypy + trailing-whitespace hooks run on every commit.

## Module overview

| Module | Responsibility |
|---|---|
| `preprocessing.py` | Text normalisation, CSV loading, rule/SERP/embedding intent classification, brand flagging |
| `vectorization.py` | TF-IDF + Sentence Transformer encoding, similarity matrices, hybrid feature composition, SERP-overlap matrix |
| `clustering.py`   | KMeans, MiniBatchKMeans, Agglomerative, HDBSCAN, graph (Louvain / Leiden) — plus PCA / UMAP / t-SNE reduction and auto-k (silhouette / Calinski-Harabasz) |
| `labeling.py`     | TF-IDF / class TF-IDF / centroid / MMR cluster label generation |
| `scoring.py`      | Page mapping, gap thresholds (fixed / percentile / adaptive), confidence bands, cannibalisation, opportunity scoring with profiles, cluster quality report |
| `pipeline.py`     | `run_keyword_clustering` — the shared orchestration service that both the CLI and the Streamlit app call |
| `integrations.py` | Connector CSV normalisers (GSC, GA4, Ahrefs, Semrush, Screaming Frog, Sitebulb) + simple HTML page-enrichment crawler |
| `visualization.py`| Plotly chart functions (`plot_3d_clusters`, `plot_treemap`, etc.) |
| `export.py`       | CSV / HTML / Markdown report writers (`save_all_outputs`, `build_interactive_report`) |
| `cli.py`          | `argparse` entry point — `run`, `compare`, `tune`, `crawl`, `enrich-pages`, `import-{gsc,ga4,ahrefs,semrush,screamingfrog,sitebulb}` |

## Adding a new clustering method

1. Add a branch to `cluster_keywords()` in `clustering.py` and (optionally) any new fields to `ClusteringConfig`.
2. Wire it through `pipeline.py:run_keyword_clustering` if it needs special pre-processing (see how `graph` consumes
   `cluster_similarity`).
3. Update the `choices` list on `--method` in `cli.py` and the matching dropdown in `app/streamlit_app.py`.
4. Add a test in `tests/test_clustering.py` covering both label-count and a sensible-clustering invariant.

## Adding a new chart

1. Add a `plot_*()` function in `visualization.py` and use the `CLUSTER_COLOURS` palette for cluster colour.
2. Call it from `save_all_charts()` if you want the static HTML/PNG export.
3. Add its filename to `build_interactive_report()` in `export.py` if it should appear in the bundled HTML report.
4. Surface it in `app/streamlit_app.py` under the relevant tab.

## Docs site

```bash
pip install mkdocs mkdocs-material
mkdocs serve        # local preview at http://127.0.0.1:8000
mkdocs build        # build static site into site/
mkdocs gh-deploy    # deploy to GitHub Pages
```
