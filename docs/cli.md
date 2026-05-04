# CLI Reference

## Usage

```bash
keyword-cluster <command> [options]
```

Top-level options apply to every subcommand:

| Flag | Default | Description |
|---|---|---|
| `--verbose`, `-v` | off | Print full traceback on error instead of the concise message |

Main subcommands:

- `run` — Cluster keywords and generate reports.
- `tune` — Bounded grid search over methods, k, embeddings, and weights.
- `compare` — Diff two historical runs.
- `crawl` / `enrich-pages` — Enrich a Pages CSV with on-page content signals.
- `import-{gsc,ga4,ahrefs,semrush,screamingfrog,sitebulb}` — Normalise connector exports.

## `run` — full flag reference

### Inputs

| Flag | Default | Description |
|---|---|---|
| `--keywords` | required | CSV with `keyword` column |
| `--pages` | optional | Pages CSV (`url`, `page_name`, optional `title`/`meta_description`/`h1`/`headings`/`body_excerpt`/`target_keyword`/`page_type`) |
| `--topics` | optional | CSV with `topic` column |
| `--serp-file` | optional | SERP CSV (`keyword`, `position`, `url`, `title`) — joined onto keywords as `serp_urls` |
| `--output` | `outputs` | Output directory |
| `--run-history` | off | Mirror this run's bundle into `outputs/runs/<timestamp>_<id>/` |
| `--run-output-root` | `outputs/runs` | Override the run-history root directory |
| `--brand` | (none) | One or more brand terms; matches in keywords get `branded=True` |

### Clustering method

| Flag | Default | Description |
|---|---|---|
| `--method` | `kmeans` | `kmeans` \| `agglomerative` \| `hdbscan` \| `graph` |
| `--clusters` | `8` | Target k for fixed-k methods (kmeans, agglomerative) |
| `--auto-k` | `none` | `none` \| `silhouette` \| `calinski_harabasz` — pick k automatically over `[k-min, k-max]` |
| `--k-min` | `4` | Lower bound for auto-k |
| `--k-max` | `40` | Upper bound for auto-k |
| `--use-minibatch` | off | Use `MiniBatchKMeans` instead of `KMeans` (faster on large data) |

### Agglomerative tuning

| Flag | Default | Description |
|---|---|---|
| `--agglomerative-linkage` | `ward` | `ward` \| `single` \| `average` \| `complete` |
| `--distance-metric` | `euclidean` | Distance metric for agglomerative (`ward` requires `euclidean`) |

> **Note.** When using a transformer embedding with `similarity=semantic` or `hybrid`, the pipeline overrides `ward + euclidean` to `average + cosine` automatically — that combination matches the L2-normalised geometry of semantic embeddings.

### HDBSCAN tuning

| Flag | Default | Description |
|---|---|---|
| `--min-cluster-size` | `5` | Minimum size for a region to qualify as a cluster |
| `--min-samples` | `2` | Higher values = more conservative clustering, more `-1` noise |
| `--cluster-selection-method` | `eom` | `eom` \| `leaf` |
| `--hdbscan-metric` | `euclidean` | Distance metric for HDBSCAN |

### Graph clustering

| Flag | Default | Description |
|---|---|---|
| `--graph-k` | `10` | Number of nearest neighbours per node in the kNN graph |
| `--graph-min-similarity` | `0.55` | Drop edges below this cosine similarity before community detection |
| `--community-algorithm` | `louvain` | `louvain` \| `leiden` (Leiden requires `leidenalg` + `igraph`) |
| `--graph-use-ann` | off | Use `hnswlib` for approximate-nearest-neighbour kNN — much faster on large data |
| `--graph-ann-ef` | `100` | hnswlib ef search parameter (higher = more accurate, slower) |

### Embedding model

| Flag | Default | Description |
|---|---|---|
| `--embedding-model` | `tfidf` | `tfidf` \| any SentenceTransformer model name (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `intfloat/e5-small-v2`, …) |
| `--embedding` | alias | Alias for `--embedding-model` |
| `--embedding-text-mode` | `keyword` | `keyword` \| `expanded` (encode keyword + intent + SERP context for richer separation) |
| `--embedding-batch-size` | `64` | Batch size passed to `model.encode()` |
| `--embedding-device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `--normalize-embeddings` | off | L2-normalise transformer outputs (recommended for cosine workloads) |
| `--embedding-cache` | `.cache/embeddings` | Disk cache directory keyed by `sha256(texts)` |
| `--embedding-chunk-size` | `0` | Encode embeddings in chunks of this size (0 = whole batch) |
| `--tfidf-svd-components` | `0` | Reduce TF-IDF dimensionality with TruncatedSVD before clustering (0 = leave sparse) |
| `--preprocess` | `stem` | `none` \| `light` \| `stem` \| `lemmatize` — applied before vectorisation, including the embedding pass |

### Similarity blending

| Flag | Default | Description |
|---|---|---|
| `--similarity` | `tfidf` | `semantic` \| `tfidf` \| `hybrid` |
| `--semantic-weight` | `0.65` | Hybrid semantic-channel weight |
| `--tfidf-weight` | `0.35` | Hybrid TF-IDF-channel weight |
| `--serp-weight` | `0.0` | Hybrid SERP-overlap weight (used by graph mode when `serp_urls` is present) |

### Page mapping & gap detection

| Flag | Default | Description |
|---|---|---|
| `--gap-threshold` | `0.25` | Content-gap threshold (used by `fixed` mode) |
| `--gap-threshold-mode` | `fixed` | `fixed` \| `percentile` \| `adaptive` |

### Intent classification

| Flag | Default | Description |
|---|---|---|
| `--intent-mode` | `rules` | `rules` \| `serp` \| `embedding` \| `manual` |
| `--local-intent-tokens` | (built-in AU set) | Override the local-intent gazetteer used by the rules-based classifier. Pass region-specific tokens (e.g. `--local-intent-tokens near nearby london manchester edinburgh`). Empty = revert to defaults (Brisbane / Sydney / Melbourne / Perth / Adelaide / Gold Coast). |

### Opportunity scoring

| Flag | Default | Description |
|---|---|---|
| `--opportunity-profile` | `balanced` | `balanced` \| `quick-wins` \| `growth` \| `commercial` |

### Cluster labelling

| Flag | Default | Description |
|---|---|---|
| `--labeling` | `tfidf` | `tfidf` \| `c-tfidf` \| `centroid` \| `mmr` (`keybert` raises until KeyBERT is wired) |

### Visualisation

| Flag | Default | Description |
|---|---|---|
| `--reduction` | `pca` | `pca` \| `umap` \| `tsne` |
| `--umap-min-dist` | `0.1` | UMAP `min_dist` parameter — controls how tightly UMAP packs points. Set `0.0` for BERTopic-style tight clusters; the default `0.1` matches umap-learn's own default. |

## `tune` — bounded grid search

Inherits every `run` flag and adds a small grid:

| Flag | Default | Description |
|---|---|---|
| `--methods` | `kmeans agglomerative hdbscan graph` | Methods to grid-search |
| `--clusters-grid` | `6 8 10` | k values for fixed-k methods |
| `--embedding-models` | `tfidf all-MiniLM-L6-v2` | Embedding models to grid-search |
| `--reductions` | `pca umap` | Dimensionality reduction methods |
| `--semantic-weights` | `0.45 0.55 0.65` | Hybrid semantic weights |
| `--tfidf-weights` | `0.55 0.45 0.35` | Hybrid TF-IDF weights |
| `--hdbscan-min-cluster-sizes` | `5 10` | HDBSCAN `min_cluster_size` values |
| `--hdbscan-min-samples-grid` | `1 2 4` | HDBSCAN `min_samples` values |

## `compare`

| Flag | Default | Description |
|---|---|---|
| `--run-a` | required | First run-history folder |
| `--run-b` | required | Second run-history folder |
| `--output` | `outputs/compare` | Output directory |

## `crawl` / `enrich-pages`

| Flag | Default | Description |
|---|---|---|
| `--pages` | required | Pages CSV with at least a `url` column |
| `--output` | `outputs/enriched_pages.csv` | Output CSV path |
| `--timeout` | `20` | Per-URL HTTP timeout in seconds (`crawl` only) |

## `import-*`

Each connector subcommand takes the same flags:

| Flag | Default | Description |
|---|---|---|
| `--file` | required | Connector CSV |
| `--output` | `outputs/<source>_normalized.csv` | Normalised CSV path |

Sources: `gsc`, `ga4`, `ahrefs`, `semrush`, `screamingfrog`, `sitebulb`.

---

## Examples

### Hybrid semantic + lexical run

```bash
keyword-cluster run \
  --keywords keywords.csv \
  --pages pages.csv \
  --embedding-model all-MiniLM-L6-v2 \
  --similarity hybrid \
  --semantic-weight 0.65 --tfidf-weight 0.35 \
  --method kmeans --clusters 12 \
  --output outputs/
```

### SEO-expanded embeddings + auto-k

```bash
keyword-cluster run \
  --keywords keywords.csv \
  --embedding-model all-MiniLM-L6-v2 \
  --embedding-text-mode expanded \
  --similarity semantic \
  --auto-k silhouette --k-min 4 --k-max 30
```

### Crawl pages and enrich mapping inputs

```bash
keyword-cluster crawl --pages pages.csv --output outputs/enriched_pages.csv
```

### Tune clustering configuration

```bash
keyword-cluster tune \
  --keywords keywords.csv --pages pages.csv \
  --similarity hybrid \
  --methods kmeans agglomerative hdbscan graph \
  --embedding-models tfidf all-MiniLM-L6-v2 \
  --reductions pca umap \
  --output outputs/tuning_report
```

### Compare two runs

```bash
keyword-cluster compare \
  --run-a outputs/runs/20260501T010000Z_aaaa1111 \
  --run-b outputs/runs/20260504T023000Z_bbbb2222 \
  --output outputs/compare
```

### Region-specific intent gazetteer

```bash
# UK keyword set — override the AU-centric local-intent tokens.
keyword-cluster run \
  --keywords keywords.csv --pages pages.csv \
  --intent-mode rules \
  --local-intent-tokens near nearby london manchester edinburgh birmingham glasgow liverpool

# BERTopic-style tight UMAP visualisation.
keyword-cluster run \
  --keywords keywords.csv \
  --reduction umap \
  --umap-min-dist 0.0
```
