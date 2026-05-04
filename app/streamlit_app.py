"""Streamlit dashboard for SEO keyword clustering."""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Pure helpers extracted to keep this file maintainable. Single import block —
# ruff reorders multi-name `from X import` correctly without splitting per name.
from app.helpers import STAGE_LABELS as _STAGE_LABELS  # isort: skip
from app.helpers import build_pipeline_config as _build_pipeline_config  # isort: skip
from app.helpers import chart_png as _chart_png  # isort: skip
from app.helpers import cluster_summary_bytes as _cluster_summary_bytes  # isort: skip
from app.helpers import column_config as _column_config  # isort: skip
from app.helpers import friendly_pipeline_error as _friendly_pipeline_error  # isort: skip
from app.helpers import interactive_report_html as _interactive_report_html  # isort: skip
from app.helpers import recommendations_text as _recommendations_text  # isort: skip
from keyword_clustering.labeling import apply_cluster_labels, generate_cluster_labels
from keyword_clustering.pipeline import run_keyword_clustering
from keyword_clustering.preprocessing import set_local_intent_tokens
from keyword_clustering.scoring import (
    build_cluster_quality_report,
    detect_cannibalization,
    detect_content_gaps,
)
from keyword_clustering.visualization import (
    plot_2d_clusters,
    plot_3d_clusters,
    plot_opportunity_matrix,
    plot_sankey,
    plot_treemap,
)

st.set_page_config(
    page_title="SEO Keyword Clustering",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_embedding_model_metadata(model_name: str, device: str = "cpu") -> dict[str, object]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return {
        "model_name": model_name,
        "embedding_dimension": int(model.get_sentence_embedding_dimension()),
        "device": device,
    }


st.title("SEO Keyword Clustering & Intelligence Tool")
st.caption("Upload keywords, pages, and topics — get clusters, gap analysis, and visual reports.")

_summary_placeholder = st.empty()


def _render_run_summary() -> None:
    """Persistent run-summary chip — shown above the tabs once a clustering result exists."""
    rdf = st.session_state.get("result_df")
    if rdf is None:
        return
    settings = st.session_state.get("last_run_settings", {})
    method_ = settings.get("method", "?")
    embedding_ = settings.get("embedding", "?")
    n_clusters_ = int(rdf["cluster_id"].nunique()) if "cluster_id" in rdf.columns else 0
    _summary_placeholder.info(
        f"**Active run** · method=`{method_}` · embedding=`{embedding_}` · "
        f"clusters=`{n_clusters_}` · keywords=`{len(rdf)}` ",
        icon="🧭",
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────


def _read_csv_text_block(text: str, name: str) -> pd.DataFrame | None:
    if not text.strip():
        return None
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception as exc:
        st.error(f"{name} pasted CSV could not be parsed: {exc}")
        return None


st.subheader("Data Input")
input_mode = st.radio("Input mode", ["Upload CSV", "Paste CSV text"], horizontal=True)

if input_mode == "Upload CSV":
    in_col1, in_col2, in_col3 = st.columns(3)
    with in_col1:
        kw_file = st.file_uploader("Keywords CSV (required)", type="csv", key="kw_main")
    with in_col2:
        pages_file = st.file_uploader("Pages CSV (optional)", type="csv", key="pages_main")
    with in_col3:
        topics_file = st.file_uploader("Topics CSV (optional)", type="csv", key="topics_main")
    kw_text = ""
    pages_text = ""
    topics_text = ""
else:
    in_col1, in_col2, in_col3 = st.columns(3)
    with in_col1:
        kw_text = st.text_area(
            "Keywords CSV text (required)",
            height=180,
            placeholder="keyword,search_volume,keyword_difficulty\nseo audit,1200,42\ntechnical seo audit,800,51",
            key="kw_text_main",
        )
    with in_col2:
        pages_text = st.text_area(
            "Pages CSV text (optional)",
            height=180,
            placeholder="url,page_name,title,h1\n/seo-audit,SEO Audit,SEO Audit Services,Technical SEO Audit",
            key="pages_text_main",
        )
    with in_col3:
        topics_text = st.text_area(
            "Topics CSV text (optional)",
            height=180,
            placeholder="topic\ntechnical seo\nwebsite design",
            key="topics_text_main",
        )
    kw_file = None
    pages_file = None
    topics_file = None

_format_help_col1, _format_help_col2, _format_help_col3 = st.columns(3)
with _format_help_col1:
    with st.popover("Keywords CSV format", use_container_width=True):
        st.markdown(
            "Required column: `keyword`. "
            "Optional: `search_volume`, `keyword_difficulty`, `cpc`, `current_url`, `rank`, "
            "`clicks`, `impressions`, `ctr`, `intent`, `featured_snippet`, `local_pack`, "
            "`people_also_ask`, `image_pack`, `video_result`, `serp_urls`."
        )
with _format_help_col2:
    with st.popover("Pages CSV format", use_container_width=True):
        st.markdown(
            "Required columns: `url`, `page_name`. "
            "Optional: `title`, `meta_description`, `h1`, `headings`, `body_excerpt`, "
            "`target_keyword`, `page_type` — all used to enrich keyword→page similarity."
        )
with _format_help_col3:
    with st.popover("Topics CSV format", use_container_width=True):
        st.markdown("Required column: `topic`. One row per topic; all other columns are ignored.")


with st.sidebar:
    st.header("Clustering Settings")
    method = st.selectbox(
        "Clustering method",
        ["kmeans", "agglomerative", "hdbscan", "graph"],
        index=0,
        help=(
            "kmeans = fast, fixed-k centroids. agglomerative = hierarchical bottom-up. "
            "hdbscan = density-based, picks its own count and finds noise. "
            "graph = k-NN graph + Louvain/Leiden community detection."
        ),
    )
    n_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=30,
        value=8,
        help="Used by kmeans and agglomerative. Ignored by hdbscan/graph (those pick automatically).",
    )
    embedding = st.selectbox(
        "Embedding model",
        ["tfidf", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "intfloat/e5-small-v2"],
        index=0,
        help="tfidf = no model download required. The transformer models give better semantic clustering on first run after a model download (~80-400 MB).",
    )
    similarity_mode = st.selectbox(
        "Similarity mode",
        ["tfidf", "semantic", "hybrid"],
        index=0,
        help="tfidf = lexical only. semantic = embedding cosine. hybrid = weighted blend of both, optionally adding SERP overlap.",
    )
    preprocess_mode = st.selectbox(
        "Preprocess mode",
        ["none", "light", "stem", "lemmatize"],
        index=2,
        help="Text normalisation applied to the keyword string before vectorisation.",
    )
    gap_threshold = st.slider(
        "Gap threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Keywords whose best page-similarity is below this threshold are flagged as content gaps.",
    )
    gap_threshold_mode = st.selectbox(
        "Gap threshold mode",
        ["fixed", "percentile", "adaptive"],
        index=0,
        help="fixed = absolute floor. percentile = bottom-N% of distribution. adaptive = max(percentile, adaptive_floor).",
    )
    intent_mode = st.selectbox(
        "Intent mode",
        ["rules", "serp", "embedding", "manual"],
        index=0,
        help="rules = keyword-pattern dictionary. serp = SERP-feature flags. embedding = prototype-overlap heuristic. manual = honour pre-existing 'intent' column.",
    )
    local_intent_tokens_text = st.text_input(
        "Local-intent tokens (override)",
        value="",
        help=(
            "Space- or comma-separated tokens that should trigger the 'local' intent class. "
            "Empty = use the built-in AU set (Brisbane / Sydney / Melbourne / Perth / Adelaide / Gold Coast). "
            "Example UK set: near nearby london manchester edinburgh"
        ),
    )
    reduction = st.radio(
        "Dimensionality reduction",
        ["pca", "umap", "tsne"],
        index=0,
        help="Used to project embeddings into 2D/3D for the Cluster Explorer plot.",
    )
    with st.expander("Advanced Settings"):
        adv_tabs = st.tabs(["Hybrid weights", "Embeddings", "Graph", "HDBSCAN", "Run history"])

        with adv_tabs[0]:
            semantic_weight = st.slider(
                "Semantic weight",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                help="Weight of the semantic-embedding similarity channel in hybrid mode.",
            )
            tfidf_weight = st.slider(
                "TF-IDF weight",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01,
                help="Weight of the TF-IDF channel in hybrid mode.",
            )
            serp_weight = st.slider(
                "SERP weight",
                min_value=0.0,
                max_value=1.0,
                value=0.20,
                step=0.01,
                help="Weight of SERP-URL Jaccard overlap (0 disables; only used when serp_urls column is present).",
            )
            _w_total = max(1e-9, semantic_weight + tfidf_weight + serp_weight)
            st.caption(
                f"Weights normalised at runtime: semantic={semantic_weight / _w_total:.2f} / "
                f"tfidf={tfidf_weight / _w_total:.2f} / serp={serp_weight / _w_total:.2f}"
            )

        with adv_tabs[1]:
            embedding_cache_dir = st.text_input(
                "Embedding cache dir",
                value=".cache/embeddings",
                help="Embeddings are cached on disk by content-hash. Reuse this directory to avoid recomputing on rerun.",
            )
            embedding_text_mode = st.selectbox(
                "Embedding text mode",
                ["keyword", "expanded"],
                index=0,
                help="keyword = encode the bare keyword. expanded = encode keyword + intent + SERP context for richer separation.",
            )
            embedding_chunk_size = st.number_input(
                "Embedding chunk size (0=off)",
                min_value=0,
                max_value=50000,
                value=0,
                step=100,
                help="Encode embeddings in chunks of this size to limit RAM. 0 = encode the whole batch at once.",
            )
            tfidf_svd_components = st.number_input(
                "TF-IDF SVD components (0=off)",
                min_value=0,
                max_value=1000,
                value=0,
                step=10,
                help="Reduce TF-IDF dimensionality with TruncatedSVD before clustering. 0 = leave sparse.",
            )
            umap_min_dist = st.slider(
                "UMAP min_dist",
                min_value=0.0,
                max_value=0.99,
                value=0.1,
                step=0.05,
                help=(
                    "Controls how tightly UMAP packs points. 0.0 produces BERTopic-style tight clusters; "
                    "0.1 matches umap-learn's default. Only used when 'Dimensionality reduction' is umap."
                ),
            )
            cache_dir = Path(embedding_cache_dir)
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.npy"))
                st.caption(f"Embedding cache status: {len(cache_files)} vector file(s)")
            else:
                st.caption("Embedding cache status: directory not found yet")
            load_model_meta = st.checkbox(
                "Probe selected model status",
                value=False,
                help="Tries to load the selected sentence-transformer model and report its embedding dimension.",
            )
            if embedding != "tfidf" and load_model_meta:
                try:
                    meta = load_embedding_model_metadata(embedding, device="cpu")
                    st.caption(f"Model download status: ready ({meta['embedding_dimension']} dims)")
                except (OSError, ValueError, RuntimeError) as exc:
                    st.caption(f"Model download status: unavailable ({exc})")

        with adv_tabs[2]:
            graph_k = st.number_input(
                "Graph k-NN",
                min_value=2,
                max_value=100,
                value=10,
                step=1,
                help="Number of nearest neighbours per keyword in the graph-clustering kNN graph.",
            )
            graph_min_similarity = st.slider(
                "Graph min similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                help="Edges below this cosine similarity are dropped before community detection.",
            )
            community_algorithm = st.selectbox(
                "Community algorithm",
                ["louvain", "leiden"],
                index=0,
                help="leiden requires the optional 'leidenalg' + 'igraph' packages.",
            )
            graph_use_ann = st.checkbox(
                "Use ANN for graph k-NN (hnswlib)",
                value=False,
                help="Approximate-nearest-neighbour kNN — much faster on large datasets, requires hnswlib.",
            )
            graph_ann_ef = st.number_input(
                "Graph ANN ef",
                min_value=20,
                max_value=500,
                value=100,
                step=10,
                help="hnswlib search parameter — higher = more accurate, slower.",
            )

        with adv_tabs[3]:
            hdbscan_min_cluster_size = st.number_input(
                "HDBSCAN min_cluster_size",
                min_value=2,
                max_value=200,
                value=5,
                step=1,
                help="Minimum size for a region to qualify as a cluster. Smaller = more clusters, more noise.",
            )
            hdbscan_min_samples = st.number_input(
                "HDBSCAN min_samples",
                min_value=1,
                max_value=200,
                value=2,
                step=1,
                help="Higher values produce more conservative clustering with more noise points (cluster_id = -1).",
            )

        with adv_tabs[4]:
            save_run_history = st.checkbox(
                "Save this run to outputs/runs/",
                value=False,
                help="Writes a timestamped folder with config.json, metrics.json, and all CSVs — required for the Run Comparison tab.",
            )

    st.header("Filters")
    keyword_search = st.text_input(
        "Global keyword search", value="", help="Substring filter applied to the keyword column."
    )
    min_volume = st.number_input("Min search volume", min_value=0, value=0, step=100)
    max_difficulty = st.number_input("Max keyword difficulty", min_value=0, max_value=100, value=100, step=5)
    show_branded = st.radio("Branded keywords", ["All", "Branded only", "Non-branded only"], index=0)
    intent_filter = st.multiselect(
        "Intent filter",
        ["informational", "commercial", "transactional", "local", "navigational"],
        default=[],
    )

    run_btn = st.button("Run Clustering", type="primary", use_container_width=True)
    load_sample_btn = st.button(
        "Load example dataset",
        use_container_width=True,
        help="Loads the bundled examples/sample_keywords.csv + sample_pages.csv + sample_topics.csv.",
    )

    st.divider()
    st.markdown(
        "**Docs** · "
        "[CLI reference](https://github.com/johnoconnor0/keyword-clustering/blob/main/docs/cli.md) · "
        "[Output schema](https://github.com/johnoconnor0/keyword-clustering/blob/main/docs/outputs.md)"
    )


# ── Session state ─────────────────────────────────────────────────────────────

if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "coords" not in st.session_state:
    st.session_state.coords = None
if "keyword_vectors" not in st.session_state:
    st.session_state.keyword_vectors = None
if "use_example_dataset" not in st.session_state:
    st.session_state.use_example_dataset = False

if load_sample_btn:
    st.session_state.use_example_dataset = True
    st.toast("Example dataset loaded — click Run Clustering to process.", icon="✅")


# ── Pipeline ──────────────────────────────────────────────────────────────────


def _parse_pages(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    if "url" not in df.columns or "page_name" not in df.columns:
        st.error("Pages CSV must have 'url' and 'page_name' columns.")
        return None
    return df


def _parse_topics(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    if "topic" not in df.columns:
        st.error("Topics CSV must have a 'topic' column.")
        return None
    return df


def _resolve_input_frames() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    if st.session_state.get("use_example_dataset"):
        examples_root = Path(__file__).resolve().parent.parent / "examples"
        kw_path = examples_root / "sample_keywords.csv"
        pages_path = examples_root / "sample_pages.csv"
        topics_path = examples_root / "sample_topics.csv"
        if kw_path.exists():
            kw_df = pd.read_csv(kw_path)
            pages_df = pd.read_csv(pages_path) if pages_path.exists() else None
            topics_df = pd.read_csv(topics_path) if topics_path.exists() else None
            return kw_df, pages_df, topics_df
    if input_mode == "Upload CSV":
        kw_df = pd.read_csv(kw_file) if kw_file is not None else None
        pages_df = pd.read_csv(pages_file) if pages_file is not None else None
        topics_df = pd.read_csv(topics_file) if topics_file is not None else None
        return kw_df, pages_df, topics_df
    kw_df = _read_csv_text_block(kw_text, "Keywords")
    pages_df = _read_csv_text_block(pages_text, "Pages")
    topics_df = _read_csv_text_block(topics_text, "Topics")
    return kw_df, pages_df, topics_df


# Pipeline-config builder + UI-side helpers all live in app.helpers.


if run_btn:
    kw_df, raw_pages_df, raw_topics_df = _resolve_input_frames()
    if kw_df is None:
        st.error("Provide Keywords data via CSV upload or pasted CSV text.")
    elif "keyword" not in kw_df.columns:
        st.error("Keywords input must have a 'keyword' column.")
    else:
        pages_df = _parse_pages(raw_pages_df)
        topics_df = _parse_topics(raw_topics_df)

        # Apply (or reset) the local-intent gazetteer override before the pipeline runs.
        tokens = [t.strip() for t in local_intent_tokens_text.replace(",", " ").split() if t.strip()]
        set_local_intent_tokens(tokens or None)

        cfg = _build_pipeline_config(
            method=method,
            n_clusters=n_clusters,
            embedding=embedding,
            reduction=reduction,
            similarity=similarity_mode,
            semantic_weight=semantic_weight,
            tfidf_weight=tfidf_weight,
            serp_weight=serp_weight,
            preprocess=preprocess_mode,
            gap_threshold=gap_threshold,
            gap_threshold_mode=gap_threshold_mode,
            intent_mode=intent_mode,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
            graph_k=graph_k,
            graph_min_similarity=graph_min_similarity,
            community_algorithm=community_algorithm,
            graph_use_ann=graph_use_ann,
            graph_ann_ef=graph_ann_ef,
            embedding_cache_dir=embedding_cache_dir,
            embedding_text_mode=embedding_text_mode,
            embedding_chunk_size=embedding_chunk_size,
            tfidf_svd_components=tfidf_svd_components,
            umap_min_dist=umap_min_dist,
            save_run_history=save_run_history,
        )

        with st.status("Running clustering pipeline…", expanded=True) as status:
            progress_bar = st.progress(0.0)

            def _on_progress(stage: str, fraction: float) -> None:
                label = _STAGE_LABELS.get(stage, stage)
                status.update(label=label)
                try:
                    progress_bar.progress(min(1.0, max(0.0, fraction)))
                except Exception:  # noqa: BLE001 — progress widget may be torn down on error
                    pass

            try:
                result = run_keyword_clustering(kw_df, pages_df, topics_df, cfg, progress_callback=_on_progress)
                df_result = result.df
                coords = result.coords
                keyword_vectors = result.keyword_vectors
                status.update(label="Clustering, labelling, and scoring complete.", state="complete")
                progress_bar.progress(1.0)
                st.session_state.result_df = df_result
                st.session_state.coords = coords
                st.session_state.keyword_vectors = keyword_vectors
                st.session_state.last_run_settings = {
                    "method": method,
                    "embedding": embedding,
                    "similarity": similarity_mode,
                    "preprocess": preprocess_mode,
                }
                st.success(f"Clustered {len(df_result)} keywords into {df_result['cluster_id'].nunique()} clusters.")
            except Exception as exc:  # noqa: BLE001 — top-level UX boundary; we re-display the error
                status.update(label="Pipeline failed.", state="error")
                hint = _friendly_pipeline_error(exc)
                if hint:
                    st.error(f"Pipeline error: {hint}")
                else:
                    st.error(
                        "Pipeline error. Common causes: missing required column in Keywords/Pages CSV, "
                        "fewer keywords than the requested cluster count, embedding model download blocked."
                    )
                with st.expander("Technical detail"):
                    st.code(f"{type(exc).__name__}: {exc}", language=None)


# ── Helpers ───────────────────────────────────────────────────────────────────


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if keyword_search:
        df = df[df["keyword"].astype(str).str.contains(keyword_search, case=False, na=False)]
    if "search_volume" in df.columns:
        df = df[df["search_volume"].fillna(0) >= min_volume]
    if "keyword_difficulty" in df.columns:
        df = df[df["keyword_difficulty"].fillna(100) <= max_difficulty]
    if show_branded == "Branded only" and "branded" in df.columns:
        df = df[df["branded"]]
    elif show_branded == "Non-branded only" and "branded" in df.columns:
        df = df[~df["branded"]]
    if intent_filter and "primary_intent" in df.columns:
        df = df[df["primary_intent"].isin(intent_filter)]
    return df


def _drop_hdbscan_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Hide HDBSCAN noise rows (cluster_id == -1) from the visualisation tabs.
    The Cluster Explorer keeps a dedicated 'Noise/Outliers' panel — every other
    chart should exclude the -1 mega-cluster by default."""
    if method != "hdbscan" or "cluster_id" not in df.columns:
        return df
    return df[df["cluster_id"] != -1]


# ── Tabs ──────────────────────────────────────────────────────────────────────

_render_run_summary()

tabs = st.tabs(
    [
        "Cluster Explorer",
        "Page Mapping",
        "Content Gaps",
        "Cannibalization",
        "Opportunity Matrix",
        "Cluster Quality",
        "Run Comparison",
        "Exports",
    ]
)

if st.session_state.result_df is not None:
    df_raw = st.session_state.result_df.reset_index(drop=True)
    coords = st.session_state.coords
    df = apply_filters(df_raw)

    # Tab 1: Cluster Explorer
    with tabs[0]:
        st.subheader("Cluster Explorer")
        col1, col2 = st.columns(2)
        clusters_available = (
            ["All"] + sorted(df["cluster_label"].unique().tolist()) if "cluster_label" in df.columns else ["All"]
        )
        selected_cluster = col1.selectbox("Filter by cluster", clusters_available)
        view_dim = col2.radio("Chart dimension", ["3D", "2D"], horizontal=True)

        plot_df = df if selected_cluster == "All" else df[df["cluster_label"] == selected_cluster]
        if plot_df.empty:
            st.info("No keywords match the current filters.")
            st.stop()
        # Index by row position into `coords`, not by label — apply_filters preserves df_raw labels.
        plot_coords = coords[plot_df.index.to_numpy()]

        fig = (
            plot_3d_clusters(plot_df.reset_index(drop=True), plot_coords)
            if view_dim == "3D"
            else plot_2d_clusters(plot_df.reset_index(drop=True), plot_coords)
        )
        st.plotly_chart(fig, use_container_width=True)
        if view_dim == "3D":
            st.caption("Drag to rotate · scroll to zoom · double-click to reset. Hover for keyword details.")
        else:
            st.caption("Scroll to zoom · click-drag to pan · double-click to reset.")

        display_cols = [
            c
            for c in [
                "keyword",
                "cluster_label",
                "primary_intent",
                "recommended_page",
                "search_volume",
                "keyword_difficulty",
                "opportunity_score",
                "notes",
            ]
            if c in plot_df.columns
        ]
        st.dataframe(
            plot_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            column_config=_column_config(),
        )
        if method == "hdbscan":
            noise_df = plot_df[plot_df["cluster_id"] == -1] if "cluster_id" in plot_df.columns else pd.DataFrame()
            st.markdown("**HDBSCAN Noise/Outliers**")
            if noise_df.empty:
                st.caption("No outliers/noise points (`cluster_id = -1`).")
            else:
                st.dataframe(
                    noise_df[display_cols].reset_index(drop=True),
                    use_container_width=True,
                    column_config=_column_config(),
                )

        @st.fragment
        def _manual_cluster_editor() -> None:
            """Scoped fragment so typing into the keyword filter doesn't re-render every other tab."""
            base_df = st.session_state.result_df
            cluster_ids = (
                sorted(base_df["cluster_id"].dropna().astype(int).unique().tolist())
                if "cluster_id" in base_df.columns
                else []
            )
            if not cluster_ids:
                st.info("Run clustering first to enable editing.")
                return

            st.markdown("**Reassign keywords**")
            st.caption(
                "Filter the table to a manageable slice, edit the cluster_id column inline, then click *Apply edits*."
            )
            edit_filter = st.text_input(
                "Filter keywords (substring, case-insensitive)",
                value="",
                key="edit_filter",
            )
            edit_view = (
                (
                    base_df[base_df["keyword"].astype(str).str.contains(edit_filter, case=False, na=False)]
                    if edit_filter
                    else base_df
                )
                .head(500)[["keyword", "cluster_id", "cluster_label"]]
                .reset_index(drop=True)
            )
            edited = st.data_editor(
                edit_view,
                use_container_width=True,
                num_rows="fixed",
                disabled=["keyword", "cluster_label"],
                key="edit_table",
                column_config={
                    "cluster_id": st.column_config.SelectboxColumn(
                        "cluster_id",
                        options=cluster_ids,
                        required=True,
                    ),
                },
            )
            if st.button("Apply edits", key="edit_apply"):
                updated = base_df.copy()
                # Apply per-keyword reassignments captured by the data_editor.
                for _, row in edited.iterrows():
                    updated.loc[updated["keyword"] == row["keyword"], "cluster_id"] = int(row["cluster_id"])
                labels_df = generate_cluster_labels(updated)
                updated = apply_cluster_labels(updated, labels_df)
                st.session_state.result_df = updated
                st.success("Edits applied — labels regenerated.")
                st.rerun()

            st.markdown("**Merge clusters**")
            merge_col_a, merge_col_b = st.columns(2)
            merge_from = merge_col_a.selectbox("Merge from cluster", cluster_ids, key="edit_merge_from")
            merge_to = merge_col_b.selectbox("Merge into cluster", cluster_ids, key="edit_merge_to")
            if st.button("Apply cluster merge", key="edit_merge_apply"):
                if int(merge_from) == int(merge_to):
                    st.warning("Choose two different clusters.")
                else:
                    updated = base_df.copy()
                    updated.loc[updated["cluster_id"] == int(merge_from), "cluster_id"] = int(merge_to)
                    labels_df = generate_cluster_labels(updated)
                    updated = apply_cluster_labels(updated, labels_df)
                    st.session_state.result_df = updated
                    st.success("Clusters merged.")
                    st.rerun()

        with st.expander("Manual cluster editing"):
            _manual_cluster_editor()

    # Tab 2: Page Mapping
    with tabs[1]:
        st.subheader("Keyword → Page Mapping")
        if "recommended_page" in df.columns:
            cols = [
                c
                for c in [
                    "keyword",
                    "current_page",
                    "recommended_page",
                    "recommended_url",
                    "page_similarity_score",
                    "cluster_label",
                    "primary_intent",
                    "search_volume",
                ]
                if c in df.columns
            ]
            df_no_noise = _drop_hdbscan_noise(df)
            st.dataframe(
                df_no_noise[cols].reset_index(drop=True),
                use_container_width=True,
                column_config=_column_config(),
            )
            st.plotly_chart(plot_sankey(df_no_noise), use_container_width=True)
        else:
            st.info("Upload a pages CSV to see page mapping.")

    # Tab 3: Content Gaps
    with tabs[2]:
        st.subheader("Content Gap Analysis")
        if "content_gap" in df.columns:
            gaps = detect_content_gaps(_drop_hdbscan_noise(df))
            if gaps.empty:
                st.success("No content gaps detected with current settings.")
            else:
                st.warning(f"{len(gaps)} keywords have no suitable page match.")
                st.dataframe(gaps, use_container_width=True, column_config=_column_config())
        else:
            st.info("Upload a pages CSV to enable content gap detection.")

    # Tab 4: Cannibalization
    with tabs[3]:
        st.subheader("Cannibalization Detection")
        cannibal = detect_cannibalization(_drop_hdbscan_noise(df))
        if cannibal.empty:
            st.success("No cannibalization detected.")
        else:
            st.warning(f"{len(cannibal)} clusters have competing pages.")
            st.dataframe(cannibal, use_container_width=True, column_config=_column_config())

    # Tab 5: Opportunity Matrix
    with tabs[4]:
        st.subheader("SERP Opportunity Matrix")
        st.plotly_chart(plot_opportunity_matrix(_drop_hdbscan_noise(df)), use_container_width=True)

    # Tab 6: Cluster Quality
    with tabs[5]:
        st.subheader("Cluster Quality")
        quality = build_cluster_quality_report(df_raw, st.session_state.keyword_vectors)
        st.dataframe(quality, use_container_width=True, column_config=_column_config())
        st.download_button(
            "Cluster Quality Report (CSV)",
            data=quality.to_csv(index=False).encode(),
            file_name="cluster_quality_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Tab 7: Run Comparison
    with tabs[6]:
        st.subheader("Run Comparison")
        run_root = Path("outputs/runs")
        run_options = [p.name for p in run_root.iterdir() if p.is_dir()] if run_root.exists() else []
        if len(run_options) < 2:
            st.info("Need at least two run-history folders in outputs/runs.")
        else:
            col_ra, col_rb = st.columns(2)
            run_a = col_ra.selectbox("Run A", run_options, index=max(0, len(run_options) - 2))
            run_b = col_rb.selectbox("Run B", run_options, index=len(run_options) - 1)
            if st.button("Load Comparison"):
                a = pd.read_csv(run_root / run_a / "clustered_keywords.csv")
                b = pd.read_csv(run_root / run_b / "clustered_keywords.csv")
                merged = a[["keyword", "cluster_id", "opportunity_score", "recommended_url"]].merge(
                    b[["keyword", "cluster_id", "opportunity_score", "recommended_url"]],
                    on="keyword",
                    suffixes=("_a", "_b"),
                    how="outer",
                )
                summary = {
                    "clusters_a": int(a["cluster_id"].nunique()),
                    "clusters_b": int(b["cluster_id"].nunique()),
                    "keywords_moved": int((merged["cluster_id_a"] != merged["cluster_id_b"]).fillna(True).sum()),
                    "mapping_changes": int(
                        (merged["recommended_url_a"] != merged["recommended_url_b"]).fillna(True).sum()
                    ),
                    "avg_opportunity_delta": float(
                        (merged["opportunity_score_b"].fillna(0) - merged["opportunity_score_a"].fillna(0)).mean()
                    ),
                }
                st.json(summary)
                st.dataframe(merged, use_container_width=True, column_config=_column_config())

    # Tab 8: Exports
    with tabs[7]:
        st.subheader("Download CSV Reports")
        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                "Clustered Keywords (CSV)",
                data=df_raw.to_csv(index=False).encode(),
                file_name="clustered_keywords.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if "recommended_page" in df_raw.columns:
                page_map_cols = [
                    c
                    for c in [
                        "keyword",
                        "current_page",
                        "recommended_page",
                        "recommended_url",
                        "page_similarity_score",
                        "cluster_label",
                    ]
                    if c in df_raw.columns
                ]
                st.download_button(
                    "Keyword → Page Map (CSV)",
                    data=df_raw[page_map_cols].to_csv(index=False).encode(),
                    file_name="keyword_page_map.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if "content_gap" in df_raw.columns:
                gaps = detect_content_gaps(df_raw)
                if not gaps.empty:
                    st.download_button(
                        "Content Gap Report (CSV)",
                        data=gaps.to_csv(index=False).encode(),
                        file_name="content_gap_report.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        with col_b:
            cannibal = detect_cannibalization(df_raw)
            if not cannibal.empty:
                st.download_button(
                    "Cannibalization Report (CSV)",
                    data=cannibal.to_csv(index=False).encode(),
                    file_name="cannibalization_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.download_button(
                "Cluster Summary (CSV)",
                data=_cluster_summary_bytes(df_raw),
                file_name="cluster_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Recommendations (Markdown)",
                data=_recommendations_text(df_raw).encode(),
                file_name="recommendations.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Download Charts")
        png_note = st.empty()

        chart_configs = [
            ("3D Cluster Map", plot_3d_clusters(df_raw, coords), "cluster_map_3d.png"),
            ("2D Topic Map", plot_2d_clusters(df_raw, coords), "cluster_map_2d.png"),
            ("Opportunity Matrix", plot_opportunity_matrix(df_raw), "opportunity_matrix.png"),
            ("Cluster Treemap", plot_treemap(df_raw), "treemap.png"),
        ]
        chart_cols = st.columns(len(chart_configs))
        png_failed = False
        for col, (label, fig, fname) in zip(chart_cols, chart_configs):
            png_data = _chart_png(fig)
            if png_data:
                col.download_button(
                    label + " (PNG)", data=png_data, file_name=fname, mime="image/png", use_container_width=True
                )
            else:
                col.write(f"_{label}_")
                png_failed = True
        if png_failed:
            png_note.warning(
                "PNG export requires the `kaleido` package. Install with `pip install kaleido` "
                "(or `pip install .[app]`), then rerun the pipeline.",
                icon="⚠️",
            )

        st.divider()
        st.subheader("Download Full Report")
        st.download_button(
            "Interactive HTML Report (all charts)",
            data=_interactive_report_html(df_raw, coords),
            file_name="interactive_report.html",
            mime="text/html",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Project State")
        project_state = {
            "settings": {
                "method": method,
                "n_clusters": n_clusters,
                "embedding": embedding,
                "similarity_mode": similarity_mode,
                "preprocess_mode": preprocess_mode,
                "gap_threshold": gap_threshold,
                "gap_threshold_mode": gap_threshold_mode,
                "intent_mode": intent_mode,
                "reduction": reduction,
            },
            "row_count": int(len(df_raw)),
            "cluster_count": int(df_raw["cluster_id"].nunique()) if "cluster_id" in df_raw.columns else 0,
        }
        st.download_button(
            "Save Project (JSON)",
            data=json.dumps(project_state, indent=2).encode(),
            file_name="project_state.json",
            mime="application/json",
            use_container_width=True,
        )
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("clustered_keywords.csv", df_raw.to_csv(index=False))
            zf.writestr("cluster_summary.csv", _cluster_summary_bytes(df_raw).decode())
            zf.writestr("project_state.json", json.dumps(project_state, indent=2))
        st.download_button(
            "Download Full ZIP",
            data=bundle.getvalue(),
            file_name="keyword_cluster_project.zip",
            mime="application/zip",
            use_container_width=True,
        )

else:
    for tab in tabs:
        with tab:
            st.info("Run clustering to view this section.")
