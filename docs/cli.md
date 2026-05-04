# CLI Reference

## Usage

```bash
python -m keyword_clustering.cli run [OPTIONS]
```

Or, if installed as a package:

```bash
keyword-cluster run [OPTIONS]
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--keywords` | required | CSV file with a `keyword` column (plus optional metrics) |
| `--pages` | optional | CSV with `url` and `page_name` columns |
| `--topics` | optional | CSV with a `topic` column |
| `--clusters` | `8` | Number of clusters (KMeans / Agglomerative) |
| `--method` | `kmeans` | Clustering algorithm: `kmeans`, `agglomerative`, `hdbscan` |
| `--embedding` | `tfidf` | Embedding: `tfidf` or any Sentence Transformers model name |
| `--reduction` | `pca` | Dimensionality reduction for plots: `pca`, `umap`, `tsne` |
| `--output` | `outputs/` | Directory where all output files are written |
| `--brand` | `[]` | Space-separated brand terms for branded/non-branded flagging |

## Examples

**Basic run with TF-IDF:**

```bash
python -m keyword_clustering.cli run \
  --keywords keywords.csv \
  --pages pages.csv \
  --clusters 10 \
  --output results/
```

**Semantic clustering with HDBSCAN:**

```bash
python -m keyword_clustering.cli run \
  --keywords keywords.csv \
  --embedding all-MiniLM-L6-v2 \
  --method hdbscan \
  --reduction umap \
  --output results/
```

**With brand flagging:**

```bash
python -m keyword_clustering.cli run \
  --keywords keywords.csv \
  --brand acme "acme seo" \
  --output results/
```

## Clustering methods

| Method | Notes |
|---|---|
| `kmeans` | Fast, deterministic with `random_state=42`, equal-ish cluster sizes |
| `agglomerative` | Hierarchical, Ward linkage — good for dendrogram-style topic maps |
| `hdbscan` | Density-based, handles noise (`label == -1`), requires `pip install hdbscan` |

## Dimensionality reduction

| Method | Notes |
|---|---|
| `pca` | Fastest, fully deterministic |
| `umap` | Better cluster separation — requires `pip install umap-learn` |
| `tsne` | Good visualisation quality, slower, perplexity auto-scaled |
