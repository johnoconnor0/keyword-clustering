"""Pure helpers extracted from streamlit_app.py to keep that module under control.

Everything here is callable from outside the Streamlit script body — no closure
on widget variables, no global session-state mutation. The Streamlit-specific
column-config builder is wrapped in a graceful fallback so unit tests can import
this module without `streamlit` available.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pandas as pd

from keyword_clustering.clustering import ClusteringConfig
from keyword_clustering.pipeline import PageMappingConfig, PipelineConfig
from keyword_clustering.preprocessing import IntentConfig
from keyword_clustering.scoring import SimilarityConfig
from keyword_clustering.vectorization import EmbeddingConfig
from keyword_clustering.visualization import (
    plot_2d_clusters,
    plot_3d_clusters,
    plot_opportunity_matrix,
    plot_sankey,
    plot_treemap,
)

try:
    import streamlit as st
except ImportError:  # pragma: no cover — tests can import helpers without Streamlit installed
    st = None  # type: ignore[assignment]


def _filter_supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs that aren't valid fields on `cls`.

    The Streamlit dashboard ships ahead of the installed `keyword_clustering`
    package on some deploys (Docker layer caching, pinned wheels, etc.). When
    we add a new dataclass field on the package side, the dashboard can crash
    with `TypeError: __init__() got an unexpected keyword argument`. Filter
    here so the dashboard degrades gracefully — the missing field falls back
    to its dataclass default and the rest of the run continues.
    """
    if not is_dataclass(cls):
        return kwargs
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in kwargs.items() if k in valid}


def build_pipeline_config(
    *,
    method: str,
    n_clusters: int,
    embedding: str,
    reduction: str,
    similarity: str,
    semantic_weight: float,
    tfidf_weight: float,
    serp_weight: float,
    preprocess: str,
    gap_threshold: float,
    gap_threshold_mode: str,
    intent_mode: str,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    graph_k: int,
    graph_min_similarity: float,
    community_algorithm: str,
    graph_use_ann: bool,
    graph_ann_ef: int,
    embedding_cache_dir: str,
    embedding_text_mode: str,
    embedding_chunk_size: int,
    tfidf_svd_components: int,
    umap_min_dist: float,
    save_run_history: bool,
) -> PipelineConfig:
    """Pure builder — assemble a PipelineConfig from the Streamlit widget values.

    Each sub-config is constructed via `_filter_supported_kwargs` so the dashboard
    survives running against an older installed `keyword_clustering` package that
    doesn't yet have every field (e.g. `text_mode`, `umap_min_dist`).
    """
    embedding_cfg = EmbeddingConfig(
        **_filter_supported_kwargs(
            EmbeddingConfig,
            {
                "model_name": embedding,
                "preprocess_mode": preprocess,
                "cache_dir": embedding_cache_dir,
                "text_mode": embedding_text_mode,
                "chunk_size": int(embedding_chunk_size),
                "tfidf_svd_components": int(tfidf_svd_components),
            },
        )
    )
    similarity_cfg = SimilarityConfig(
        **_filter_supported_kwargs(
            SimilarityConfig,
            {
                "mode": similarity,
                "semantic_weight": semantic_weight,
                "tfidf_weight": tfidf_weight,
                "serp_weight": serp_weight,
            },
        )
    )
    clustering_cfg = ClusteringConfig(
        **_filter_supported_kwargs(
            ClusteringConfig,
            {
                "method": method,
                "n_clusters": n_clusters,
                "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
                "hdbscan_min_samples": hdbscan_min_samples,
                "graph_k": graph_k,
                "graph_min_similarity": graph_min_similarity,
                "community_algorithm": community_algorithm,
                "graph_use_ann": graph_use_ann,
                "graph_ann_ef": int(graph_ann_ef),
            },
        )
    )
    page_mapping_cfg = PageMappingConfig(
        **_filter_supported_kwargs(
            PageMappingConfig,
            {"gap_threshold": gap_threshold, "gap_threshold_mode": gap_threshold_mode},
        )
    )
    pipeline_kwargs = {
        "embedding": embedding_cfg,
        "similarity": similarity_cfg,
        "clustering": clustering_cfg,
        "intent": IntentConfig(mode=intent_mode),
        "page_mapping": page_mapping_cfg,
        "preprocess_mode": preprocess,
        "reduction": reduction,
        "umap_min_dist": float(umap_min_dist),
        "run_history": save_run_history,
    }
    return PipelineConfig(**_filter_supported_kwargs(PipelineConfig, pipeline_kwargs))


# Friendly-error vocabulary surfaced under the failing-pipeline expander.
FRIENDLY_HINTS: tuple[tuple[str, str], ...] = (
    ("'keyword'", "The Keywords CSV must have a column named exactly 'keyword'."),
    ("Pages data must include", "The Pages CSV must include both 'url' and 'page_name' columns."),
    (
        "similarity=semantic requires semantic",
        "Switch to similarity=tfidf, or pick a transformer (e.g. all-MiniLM-L6-v2) from the Embedding model dropdown.",
    ),
    ("similarity=tfidf requires", "Pick the 'tfidf' embedding model or switch similarity to semantic/hybrid."),
    ("hdbscan", "HDBSCAN requires the 'hdbscan' optional dependency. Install with: pip install .[advanced]"),
    ("umap", "UMAP requires 'umap-learn'. Install with: pip install .[advanced]"),
    ("Leiden clustering", "Leiden requires 'leidenalg' and 'igraph'. Use Louvain or install both packages."),
    (
        "Number of unique features",
        "Too few unique tokens after preprocessing — try 'light' preprocess mode or add more keywords.",
    ),
)


# Pipeline progress-stage display labels.
STAGE_LABELS: dict[str, str] = {
    "load": "Loading and enriching keywords…",
    "embed": "Embedding keywords (this is the slow step on first run)…",
    "similarity": "Computing similarity matrices…",
    "cluster": "Clustering…",
    "label": "Labelling clusters…",
    "page-mapping": "Mapping keywords to pages…",
    "score": "Scoring opportunities and gaps…",
    "reduce": "Reducing dimensions for visualisation…",
    "done": "Pipeline complete.",
}


def friendly_pipeline_error(exc: Exception) -> str | None:
    msg = str(exc)
    for needle, hint in FRIENDLY_HINTS:
        if needle.lower() in msg.lower():
            return hint
    return None


def safe_mode(series: pd.Series | None, default: str) -> str:
    """`series.mode().iloc[0]` raises on all-NaN input — fall back to a default instead."""
    if series is None:
        return default
    modes = series.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else default


_DATAFRAME_COLUMN_CONFIG: dict[str, Any] = {}


def column_config() -> dict[str, Any]:
    """Lazily build a column_config dict; gracefully degrade on older Streamlit."""
    if _DATAFRAME_COLUMN_CONFIG or st is None:
        return _DATAFRAME_COLUMN_CONFIG
    try:
        _DATAFRAME_COLUMN_CONFIG.update(
            {
                "search_volume": st.column_config.NumberColumn("Search Volume", format="%d"),
                "keyword_difficulty": st.column_config.NumberColumn("Difficulty", format="%d"),
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "clicks": st.column_config.NumberColumn("Clicks", format="%d"),
                "impressions": st.column_config.NumberColumn("Impressions", format="%d"),
                "ctr": st.column_config.NumberColumn("CTR", format="%.2f%%"),
                "opportunity_score": st.column_config.NumberColumn("Opportunity", format="%.2f"),
                "page_similarity_score": st.column_config.NumberColumn("Page Sim.", format="%.2f"),
                "topic_similarity_score": st.column_config.NumberColumn("Topic Sim.", format="%.2f"),
                "label_confidence": st.column_config.NumberColumn("Label Conf.", format="%.2f"),
                "intent_confidence": st.column_config.NumberColumn("Intent Conf.", format="%.2f"),
            }
        )
    except (AttributeError, ImportError):  # pragma: no cover — older Streamlit
        pass
    return _DATAFRAME_COLUMN_CONFIG


def cluster_summary_bytes(df: pd.DataFrame) -> bytes:
    """Per-cluster aggregate CSV used by the Streamlit Exports tab."""

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


def recommendations_text(df: pd.DataFrame) -> str:
    """Markdown recommendations document — one section per cluster."""
    lines = ["# SEO Keyword Cluster Recommendations\n"]
    for cluster_id, group in df.groupby("cluster_id"):
        label = group["cluster_label"].iloc[0] if "cluster_label" in group.columns else f"Cluster {cluster_id}"
        intent = safe_mode(group.get("primary_intent"), "mixed")
        page = safe_mode(group.get("recommended_page"), "unknown")
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


def interactive_report_html(df: pd.DataFrame, coords) -> bytes:
    """Self-contained HTML report bundling all five charts."""
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


def chart_png(fig) -> bytes | None:
    """Convert a Plotly figure to PNG bytes; returns None if PNG export is unavailable.

    Catches:
      * ImportError — Kaleido isn't installed.
      * ValueError — figure can't be serialised (rare).
      * RuntimeError — Kaleido 1.x raises this when system Chrome is missing.

    The dashboard treats `None` as a non-fatal degraded state and surfaces an
    `st.warning` to the user instead of crashing the page.
    """
    try:
        return fig.to_image(format="png", width=1400, height=900, scale=2)
    except (ImportError, ValueError, RuntimeError):
        return None
