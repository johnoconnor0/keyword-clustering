"""Streamlit dashboard for SEO keyword clustering."""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from keyword_clustering.clustering import cluster_keywords, reduce_dimensions
from keyword_clustering.labeling import apply_cluster_labels, generate_cluster_labels
from keyword_clustering.preprocessing import enrich_keywords
from keyword_clustering.scoring import (
    add_notes,
    compute_opportunity_score,
    detect_cannibalization,
    detect_content_gaps,
    map_keywords_to_pages,
)
from keyword_clustering.vectorization import (
    build_tfidf_vectorizer,
    compute_similarity_matrix,
    vectorize_keywords_st,
    vectorize_keywords_tfidf,
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

st.title("SEO Keyword Clustering & Intelligence Tool")
st.caption("Upload keywords, pages, and topics — get clusters, gap analysis, and visual reports.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data Upload")
    kw_file = st.file_uploader("Keywords CSV (required)", type="csv", key="kw")
    pages_file = st.file_uploader("Pages CSV (optional)", type="csv", key="pages")
    topics_file = st.file_uploader("Topics CSV (optional)", type="csv", key="topics")

    st.header("Clustering Settings")
    method = st.selectbox("Clustering method", ["kmeans", "agglomerative", "hdbscan"], index=0)
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=30, value=8)
    embedding_label = st.radio(
        "Embedding model",
        ["TF-IDF (fast)", "all-MiniLM-L6-v2 (semantic)"],
        index=0,
        help="Semantic embeddings produce better clusters but require ~200 MB download on first run.",
    )
    embedding = "tfidf" if embedding_label == "TF-IDF (fast)" else "all-MiniLM-L6-v2"
    reduction = st.radio("Dimensionality reduction", ["pca", "umap", "tsne"], index=0)

    st.header("Filters")
    min_volume = st.number_input("Min search volume", min_value=0, value=0, step=100)
    max_difficulty = st.number_input("Max keyword difficulty", min_value=0, max_value=100, value=100, step=5)
    show_branded = st.radio("Branded keywords", ["All", "Branded only", "Non-branded only"], index=0)
    intent_filter = st.multiselect(
        "Intent filter",
        ["informational", "commercial", "transactional", "navigational"],
        default=[],
    )

    run_btn = st.button("Run Clustering", type="primary", use_container_width=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "coords" not in st.session_state:
    st.session_state.coords = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _parse_pages(upload) -> tuple[list[str], list[str]]:
    df = pd.read_csv(upload)
    if "url" not in df.columns or "page_name" not in df.columns:
        st.error("Pages CSV must have 'url' and 'page_name' columns.")
        return [], []
    rows = df[["url", "page_name"]].dropna().to_dict(orient="records")
    return [r["page_name"] for r in rows], [r["url"] for r in rows]


def _parse_topics(upload) -> list[str]:
    df = pd.read_csv(upload)
    if "topic" not in df.columns:
        st.error("Topics CSV must have a 'topic' column.")
        return []
    return df["topic"].dropna().tolist()


@st.cache_data(show_spinner=False)
def run_pipeline(
    kw_df: pd.DataFrame,
    page_names: list[str],
    page_urls: list[str],
    topics: list[str],
    method_: str,
    n_clusters_: int,
    embedding_: str,
    reduction_: str,
) -> tuple[pd.DataFrame, object]:
    kw_df = enrich_keywords(kw_df)
    kw_df = kw_df.rename(columns={"intent": "primary_intent"})
    keywords = kw_df["keyword"].tolist()

    # Vectorize
    if embedding_ == "tfidf":
        corpus = keywords + page_names + topics
        vectorizer, _ = build_tfidf_vectorizer(corpus)
        kw_vectors = vectorize_keywords_tfidf(keywords, vectorizer)
        page_vectors = vectorize_keywords_tfidf(page_names, vectorizer) if page_names else None
        topic_vectors = vectorize_keywords_tfidf(topics, vectorizer) if topics else None
    else:
        all_texts = keywords + page_names + topics
        all_vecs = vectorize_keywords_st(all_texts, embedding_)
        n_kw, n_pg = len(keywords), len(page_names)
        kw_vectors = all_vecs[:n_kw]
        page_vectors = all_vecs[n_kw:n_kw + n_pg] if page_names else None
        topic_vectors = all_vecs[n_kw + n_pg:] if topics else None

    n_clust = min(n_clusters_, len(kw_df))
    labels = cluster_keywords(kw_vectors, method=method_, n_clusters=n_clust)
    kw_df["cluster_id"] = labels

    label_map = generate_cluster_labels(kw_df)
    kw_df = apply_cluster_labels(kw_df, label_map)

    if page_names and page_vectors is not None:
        kw_df = map_keywords_to_pages(kw_df, kw_vectors, page_vectors, page_names, page_urls)

    if topics and topic_vectors is not None:
        sim = compute_similarity_matrix(kw_vectors, topic_vectors)
        kw_df["primary_topic"] = [topics[i] for i in sim.argmax(axis=1)]
        kw_df["topic_similarity_score"] = sim.max(axis=1).round(4).tolist()

    kw_df = compute_opportunity_score(kw_df)
    kw_df = add_notes(kw_df)
    coords = reduce_dimensions(kw_vectors, n_components=3, method=reduction_)

    return kw_df, coords


if run_btn and kw_file:
    kw_df = pd.read_csv(kw_file)
    if "keyword" not in kw_df.columns:
        st.error("Keywords CSV must have a 'keyword' column.")
    else:
        page_names, page_urls = _parse_pages(pages_file) if pages_file else ([], [])
        topics = _parse_topics(topics_file) if topics_file else []

        with st.spinner("Running clustering pipeline…"):
            try:
                df_result, coords = run_pipeline(
                    kw_df, page_names, page_urls, topics,
                    method, n_clusters, embedding, reduction,
                )
                st.session_state.result_df = df_result
                st.session_state.coords = coords
                st.success(
                    f"Clustered {len(df_result)} keywords into "
                    f"{df_result['cluster_id'].nunique()} clusters."
                )
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
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


def _cluster_summary_bytes(df: pd.DataFrame) -> bytes:
    def _top5(x: pd.Series) -> str:
        return ", ".join(x.head(5).tolist())

    agg: dict = {"keyword": ["count", _top5]}
    if "search_volume" in df.columns:
        agg["search_volume"] = ["sum", "mean"]
    if "keyword_difficulty" in df.columns:
        agg["keyword_difficulty"] = "mean"
    if "opportunity_score" in df.columns:
        agg["opportunity_score"] = "mean"

    group_cols = ["cluster_id", "cluster_label"] if "cluster_label" in df.columns else ["cluster_id"]
    summary = df.groupby(group_cols).agg(agg)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.rename(columns={"keyword__top5": "top_keywords"})
    return summary.reset_index().to_csv(index=False).encode()


def _recommendations_text(df: pd.DataFrame) -> str:
    lines = ["# SEO Keyword Cluster Recommendations\n"]
    for cluster_id, group in df.groupby("cluster_id"):
        label = group["cluster_label"].iloc[0] if "cluster_label" in group.columns else f"Cluster {cluster_id}"
        intent = group["primary_intent"].mode().iloc[0] if "primary_intent" in group.columns else "mixed"
        page = group["recommended_page"].mode().iloc[0] if "recommended_page" in group.columns else "unknown"
        top_kws = ", ".join(group["keyword"].head(5).tolist())
        vol = int(group["search_volume"].sum()) if "search_volume" in group.columns else "N/A"
        opp = round(group["opportunity_score"].mean(), 3) if "opportunity_score" in group.columns else "N/A"
        lines += [
            f"\n## {label}",
            f"- **Recommended page:** {page}",
            f"- **Primary intent:** {intent}",
            f"- **Total search volume:** {vol}",
            f"- **Avg opportunity score:** {opp}",
            f"- **Top keywords:** {top_kws}",
            "",
        ]
    return "\n".join(lines)


def _interactive_report_html(df: pd.DataFrame, coords) -> bytes:
    charts = [
        (plot_3d_clusters(df, coords), "3D Cluster Map"),
        (plot_2d_clusters(df, coords), "2D Topic Map"),
        (plot_treemap(df), "Cluster Treemap"),
        (plot_opportunity_matrix(df), "Opportunity Matrix"),
        (plot_sankey(df), "Page → Cluster Flow"),
    ]
    sections = []
    for i, (fig, title) in enumerate(charts):
        plotlyjs = "cdn" if i == 0 else False
        html = fig.to_html(full_html=False, include_plotlyjs=plotlyjs)
        sections.append(f"<h2>{title}</h2><div>{html}</div>")

    report = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>SEO Keyword Cluster Report</title>
<style>body{{font-family:sans-serif;padding:20px}} h1{{border-bottom:2px solid #333}} h2{{margin-top:40px}}</style>
</head>
<body>
<h1>SEO Keyword Cluster Report</h1>
{"".join(sections)}
</body></html>"""
    return report.encode()


def _chart_png(fig) -> bytes | None:
    try:
        return fig.to_image(format="png", width=1400, height=900, scale=2)
    except Exception:
        return None


# ── Tabs ──────────────────────────────────────────────────────────────────────

if st.session_state.result_df is not None:
    df_raw = st.session_state.result_df
    coords = st.session_state.coords
    df = apply_filters(df_raw)

    tabs = st.tabs([
        "Cluster Explorer",
        "Page Mapping",
        "Content Gaps",
        "Cannibalization",
        "Opportunity Matrix",
        "Exports",
    ])

    # Tab 1: Cluster Explorer
    with tabs[0]:
        st.subheader("Cluster Explorer")
        col1, col2 = st.columns(2)
        clusters_available = (
            ["All"] + sorted(df["cluster_label"].unique().tolist())
            if "cluster_label" in df.columns else ["All"]
        )
        selected_cluster = col1.selectbox("Filter by cluster", clusters_available)
        view_dim = col2.radio("Chart dimension", ["3D", "2D"], horizontal=True)

        plot_df = df if selected_cluster == "All" else df[df["cluster_label"] == selected_cluster]
        plot_coords = coords[plot_df.index]

        fig = (
            plot_3d_clusters(plot_df.reset_index(drop=True), plot_coords)
            if view_dim == "3D"
            else plot_2d_clusters(plot_df.reset_index(drop=True), plot_coords)
        )
        st.plotly_chart(fig, use_container_width=True)

        display_cols = [c for c in [
            "keyword", "cluster_label", "primary_intent", "recommended_page",
            "search_volume", "keyword_difficulty", "opportunity_score", "notes",
        ] if c in plot_df.columns]
        st.dataframe(plot_df[display_cols].reset_index(drop=True), use_container_width=True)

    # Tab 2: Page Mapping
    with tabs[1]:
        st.subheader("Keyword → Page Mapping")
        if "recommended_page" in df.columns:
            cols = [c for c in [
                "keyword", "current_page", "recommended_page", "recommended_url",
                "page_similarity_score", "cluster_label", "primary_intent", "search_volume",
            ] if c in df.columns]
            st.dataframe(df[cols].reset_index(drop=True), use_container_width=True)
            st.plotly_chart(plot_sankey(df), use_container_width=True)
        else:
            st.info("Upload a pages CSV to see page mapping.")

    # Tab 3: Content Gaps
    with tabs[2]:
        st.subheader("Content Gap Analysis")
        if "content_gap" in df.columns:
            gaps = detect_content_gaps(df)
            if gaps.empty:
                st.success("No content gaps detected with current settings.")
            else:
                st.warning(f"{len(gaps)} keywords have no suitable page match.")
                st.dataframe(gaps, use_container_width=True)
        else:
            st.info("Upload a pages CSV to enable content gap detection.")

    # Tab 4: Cannibalization
    with tabs[3]:
        st.subheader("Cannibalization Detection")
        cannibal = detect_cannibalization(df)
        if cannibal.empty:
            st.success("No cannibalization detected.")
        else:
            st.warning(f"{len(cannibal)} clusters have competing pages.")
            st.dataframe(cannibal, use_container_width=True)

    # Tab 5: Opportunity Matrix
    with tabs[4]:
        st.subheader("SERP Opportunity Matrix")
        st.plotly_chart(plot_opportunity_matrix(df), use_container_width=True)

    # Tab 6: Exports
    with tabs[5]:
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
                page_map_cols = [c for c in [
                    "keyword", "current_page", "recommended_page", "recommended_url",
                    "page_similarity_score", "cluster_label",
                ] if c in df_raw.columns]
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
                col.download_button(label + " (PNG)", data=png_data,
                                    file_name=fname, mime="image/png",
                                    use_container_width=True)
            else:
                col.write(f"_{label}_")
                png_failed = True
        if png_failed:
            png_note.caption("PNG export requires `pip install kaleido`.")

        st.divider()
        st.subheader("Download Full Report")
        st.download_button(
            "Interactive HTML Report (all charts)",
            data=_interactive_report_html(df_raw, coords),
            file_name="interactive_report.html",
            mime="text/html",
            use_container_width=True,
        )

else:
    st.info("Upload a keywords CSV in the sidebar and click **Run Clustering** to begin.")

    with st.expander("Expected CSV columns"):
        st.markdown("""
**Keywords CSV** (required column: `keyword`):
```
keyword, search_volume, keyword_difficulty, cpc, current_url, rank,
clicks, impressions, ctr, intent,
featured_snippet, local_pack, people_also_ask, image_pack, video_result
```

**Pages CSV** (required columns: `url`, `page_name`):
```
url, page_name
```

**Topics CSV** (required column: `topic`):
```
topic
```
        """)
