# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.1] — Unreleased

### Fixed
- **Streamlit dashboard survives package-version skew** — when the deployed
  `app/streamlit_app.py` runs against an older installed `keyword_clustering`
  package (a common Docker / Streamlit Cloud pattern after a partial rebuild),
  fields added in newer versions (`text_mode`, `umap_min_dist`, …) no longer
  raise `TypeError: __init__() got an unexpected keyword argument`. The new
  `app/helpers.build_pipeline_config` filters kwargs against the dataclass's
  actual fields and lets unknown ones fall back to their dataclass defaults.

## [Unreleased]

### Fixed (correctness)
- **Index-mismatch trio.** `enrich_keywords` (`preprocessing.py:175`), `generate_cluster_labels` (`labeling.py:105`),
  and `build_cluster_quality_report` (`scoring.py:303`) all conflated DataFrame label index with positional offsets.
  On non-RangeIndex inputs this caused `IndexError` or — worse — silent row mis-pairing. Each function now resets
  the index defensively at its boundary.
- **Streamlit Cluster Explorer mis-charted filtered rows.** `coords[plot_df.index]` after `apply_filters` could
  produce mis-paired coordinates; explicit `.to_numpy()` indexing plus an empty-state guard fixes both the silent
  bug and the divide-by-zero crash on filter-to-zero.
- **Redundant `gap_threshold` arg in `map_keywords_to_pages`.** The positional float and `gap_config` could
  diverge; the float arg has been removed and `GapThresholdConfig` is now the sole source of truth.
- **`compute_opportunity_score` silent constant fallback.** When no signal columns are present the score
  collapsed to a constant with no warning. Now logs `WARNING` and writes `opportunity_reason="insufficient
  signal"` so users can spot the degenerate case in CSVs.

### Fixed (label-method honesty)
- **`labeling.py` no longer lies about which algorithm ran.**
  - `keybert` now raises `NotImplementedError` instead of silently aliasing to `mmr`.
  - `c-tfidf` is now a real BERTopic-style class TF-IDF (one fit over per-cluster pseudo-docs), not the per-cluster
    TF-IDF it used to alias to.
  - `mmr` is now real Carbonell-Goldstein MMR — relevance to centroid balanced against redundancy — when keyword
    embeddings are passed; falls back to the previous token-coverage heuristic if not.
  - The `label_method` column now records the strategy that actually executed.

### Added (algorithms)
- **Calinski-Harabasz auto-k.** `ClusteringConfig.auto_k` accepts `"calinski_harabasz"` for O(n·k) scoring as an
  alternative to silhouette, which is O(n²).
- **Silhouette sub-sampling.** Auto-k silhouette uses `sample_size=2000` once n exceeds 5 000 to keep it usable
  on larger datasets.
- **Sparse graph clustering edges.** `_graph_cluster` now iterates only the kNN-yielded edges instead of building
  a dense N×N similarity matrix.
- **Vectorised SERP-overlap.** `compute_serp_overlap_matrix` is now a `MultiLabelBinarizer + sparse @ sparse.T`
  pipeline rather than a Python double-loop — O(n · avg_urls) instead of O(n²).
- **Agglomerative defaults switch on transformers.** When the embedding model is a transformer and similarity
  is semantic/hybrid, the default `linkage="ward", metric="euclidean"` is upgraded to `average + cosine` to
  match L2-normalised semantic geometry.
- **True semantic intent classification.** `classify_intents_semantic(keywords, model_name)` encodes keywords
  and prototype phrases together, assigns each keyword the nearest-prototype label by cosine similarity, and
  is opt-in via `IntentConfig.embedding_model`. The previous overlap-only function is now correctly named
  `classify_intent_overlap` (with `classify_intent_embedding` kept as a backwards-compatible alias).

### Added (Streamlit UX)
- **`st.status` with stage updates** replaces the single coarse `st.spinner` over the multi-minute pipeline.
- **Tooltips on every sidebar control** (`help=` strings) — clustering method, similarity mode, gap-threshold
  mode, intent mode, weight sliders, and every Advanced Settings input.
- **Sub-tabbed Advanced Settings** (Hybrid weights / Embeddings / Graph / HDBSCAN / Run history) replaces the
  flat 17-control list.
- **Live weight-normalisation caption** under the three hybrid sliders.
- **`run_history` toggle** in the new Run-history sub-tab — the Run Comparison tab is now reachable from the UI.
- **Friendly error wrapping** with a curated set of common-cause hints plus a collapsible technical-detail
  expander.
- **Empty-state guard** when filters reduce the cluster explorer to zero rows.
- **Colour-blind-safe cluster palette** (`px.colors.qualitative.Safe`) on 3D / 2D scatter and opportunity matrix.
- **Dark-mode-safe treemap** (`Viridis` instead of `Blues`).
- **Sankey vectorised** — replaces `df.iterrows()` aggregation with a single groupby.
- **Explicit chart heights** — 600 px scatter/treemap/opportunity, 700 px Sankey.

### Added (CLI)
- **`--verbose` flag** at the parser root re-raises any caught exception so users can see a full traceback when
  the concise error message isn't enough.

### Added (tests)
- `tests/test_index_safety.py` — regression guard for the index-mismatch trio.
- `tests/test_labeling.py` — pins the `label_method` honesty contract and the c-TF-IDF / MMR / KeyBERT
  behaviour.
- `tests/test_dead_extras.py` — fails the build if any first-party module imports `gspread`, `googleapiclient`,
  or if `requirements.txt` reappears.
- `tests/test_export.py` — file-existence + schema for every artefact in `save_all_outputs`.
- `tests/test_cli.py` — argparse smoke per subcommand.
- `tests/test_integrations_errors.py` — error-path coverage for `crawl_page`, `enrich_pages_dataframe`, and
  every connector branch in `normalize_connector_csv`.
- `tests/test_scoring_branches.py` — `score_confidence_band` boundaries, opportunity-profile invalid raise,
  insufficient-signal warning, gap-threshold modes.
- `tests/test_vectorization.py` — Jaccard correctness, NaN handling, hybrid feature concatenation.
- `tests/test_intent_semantic.py` — semantic intent classifier separation, fallback behaviour.
- `tests/conftest.py` — autouse NLTK data fixture so `pytest tests/` works on a fresh clone.

### Changed (CI)
- `.github/workflows/ci.yml` now runs three jobs:
  1. Full extras matrix (Python 3.10 / 3.11 / 3.12) with `--cov-fail-under=75`.
  2. Slim install (no `[app,semantic,advanced]`) with the TF-IDF-only test subset.
  3. End-to-end `keyword-cluster run` smoke against `examples/sample_*.csv`.
- The `pip install -r requirements.txt` step is gone.

### Removed
- **Dead `[integrations]` extra** (`gspread`, `google-api-python-client`) — never imported by first-party
  code. Re-introduce together with an actual GSC/GA4/Sheets connector.
- **Dead `vectorize_keywords_st` wrapper** in `vectorization.py:103` — never called; pipeline uses
  `vectorize_keywords_st_configured`.
- **`requirements.txt`** — `pyproject.toml [project.dependencies]` is now the single source of truth.
- **Bare `except Exception:`** in `clustering.py` (`louvain_communities` fallback) and `labeling.py` —
  replaced with explicit, narrower exception types.

### Repo hygiene
- `.coverage` is now `.gitignore`d.

## [0.1.0] — 2024

### Added
- `keyword_clustering` Python package with modular architecture
- CSV input for keywords, pages, and topics with optional metrics columns
- True clustering: KMeans, Agglomerative, HDBSCAN
- TF-IDF and Sentence Transformer embedding options
- PCA and UMAP dimensionality reduction
- Vectorizer/preprocessor consistency fix (corpus preprocessed before fitting)
- Argparse CLI (`keyword-cluster run`)
- Interactive Plotly charts: 3D scatter, 2D topic map, treemap, heatmap, Sankey, opportunity matrix, network graph
- SEO analysis: page mapping, content gap detection, cannibalization detection, opportunity scoring
- Auto cluster labeling from top TF-IDF terms
- CSV and HTML report exports
- Streamlit dashboard with upload, cluster explorer, gap/cannibalization tabs, and download buttons
- Example CSVs in `examples/`
- Pytest test suite
- GitHub Actions CI
