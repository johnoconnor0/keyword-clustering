# Changelog

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
