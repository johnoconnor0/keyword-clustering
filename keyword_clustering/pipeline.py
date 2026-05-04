"""Shared keyword clustering pipeline service."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .clustering import ClusteringConfig, cluster_keywords, reduce_dimensions
from .labeling import LabelingConfig, apply_cluster_labels, generate_cluster_labels
from .preprocessing import IntentConfig, enrich_keywords
from .scoring import (
    GapThresholdConfig,
    OpportunityConfig,
    SimilarityConfig,
    add_notes,
    build_cluster_quality_report,
    compose_hybrid_similarity_matrix,
    compute_opportunity_score,
    map_keywords_to_pages,
    require_similarity_inputs,
)
from .vectorization import (
    EmbeddingConfig,
    build_tfidf_vectorizer,
    compose_hybrid_feature_vectors,
    compute_serp_overlap_matrix,
    compute_similarity_matrix,
    normalise_rows,
    vectorize_keywords_st_configured,
    vectorize_keywords_tfidf,
)

ProgressCallback = Callable[[str, float], None]
# Progress contract:
#   stage label  : human-readable name of the current pipeline stage
#   fraction     : 0.0 .. 1.0 cumulative progress
# Callbacks are advisory — they must not raise; pipeline catches and logs.

_PIPELINE_STAGES = [
    "load",
    "embed",
    "similarity",
    "cluster",
    "label",
    "page-mapping",
    "score",
    "reduce",
]


@dataclass(frozen=True)
class PageMappingConfig:
    gap_threshold: float = 0.25
    gap_threshold_mode: str = "fixed"


@dataclass(frozen=True)
class PipelineConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    page_mapping: PageMappingConfig = field(default_factory=PageMappingConfig)
    opportunity: OpportunityConfig = field(default_factory=OpportunityConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    preprocess_mode: str = "stem"
    reduction: str = "pca"
    umap_min_dist: float = 0.1
    brand_terms: list[str] = field(default_factory=list)
    run_history: bool = False
    run_output_root: str = "outputs/runs"


@dataclass
class PipelineResult:
    df: pd.DataFrame
    keyword_vectors: object
    page_vectors: object | None
    topic_vectors: object | None
    labels: np.ndarray
    coords: np.ndarray
    metrics: dict
    run_dir: str | None = None


def _keyword_text(row: pd.Series, mode: str = "keyword") -> str:
    kw = str(row.get("keyword", "")).strip()
    if mode != "expanded":
        return kw
    intent = str(row.get("primary_intent", "")).strip() or str(row.get("intent", "")).strip()
    rank = row.get("rank")
    serp_flags = []
    for col, label in (
        ("featured_snippet", "featured snippet"),
        ("people_also_ask", "PAA"),
        ("local_pack", "local pack"),
        ("image_pack", "image pack"),
        ("video_result", "video"),
    ):
        val = row.get(col, 0)
        if str(val).strip().lower() in {"1", "true", "yes"} or val == 1:
            serp_flags.append(label)
    serp_urls = str(row.get("serp_urls", "")).strip()
    parts = [f"Keyword: {kw}"]
    if intent:
        parts.append(f"Intent: {intent}")
    if serp_flags:
        parts.append(f"SERP features: {', '.join(serp_flags)}")
    if serp_urls:
        sample_urls = [u.strip() for u in serp_urls.split("|") if u.strip()][:3]
        if sample_urls:
            parts.append(f"Top SERP URLs: {', '.join(sample_urls)}")
    if pd.notna(rank):
        parts.append(f"Current rank: {rank}")
    return ". ".join(parts) + "."


def _compose_page_text(row: pd.Series) -> str:
    parts = []
    for col in (
        "page_name",
        "title",
        "h1",
        "meta_description",
        "headings",
        "body_excerpt",
        "target_keyword",
        "page_type",
    ):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return ". ".join(parts)


def _prepare_pages(pages_df: pd.DataFrame | None) -> tuple[list[str], list[str], list[str]]:
    if pages_df is None or pages_df.empty:
        return [], [], []
    if "url" not in pages_df.columns or "page_name" not in pages_df.columns:
        raise ValueError("Pages data must include 'url' and 'page_name' columns.")
    page_rows = pages_df.copy()
    page_rows["page_text"] = page_rows.apply(_compose_page_text, axis=1)
    page_rows["page_text"] = page_rows["page_text"].fillna("").astype(str)
    empty_mask = page_rows["page_text"].str.strip() == ""
    page_rows.loc[empty_mask, "page_text"] = page_rows.loc[empty_mask, "page_name"].astype(str)
    return (
        page_rows["page_text"].tolist(),
        page_rows["page_name"].astype(str).tolist(),
        page_rows["url"].astype(str).tolist(),
    )


def run_keyword_clustering(
    keywords_df: pd.DataFrame,
    pages_df: pd.DataFrame | None,
    topics_df: pd.DataFrame | None,
    config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    def _report(stage: str, fraction: float) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage, max(0.0, min(1.0, float(fraction))))
        except Exception:  # noqa: BLE001 — advisory callback, must not break the pipeline
            pass

    if "keyword" not in keywords_df.columns:
        raise ValueError("Keywords data must include a 'keyword' column.")
    _report("load", 0.0)
    df = keywords_df.copy()
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = enrich_keywords(df, brand_terms=config.brand_terms, intent_config=config.intent)
    df = df.rename(columns={"intent": "primary_intent"})

    keywords = df["keyword"].tolist()
    keyword_texts = df.apply(lambda r: _keyword_text(r, config.embedding.text_mode), axis=1).tolist()
    page_texts, page_names, page_urls = _prepare_pages(pages_df)
    topics = (
        [] if topics_df is None or topics_df.empty else topics_df["topic"].dropna().astype(str).str.strip().tolist()
    )

    _report("embed", 0.10)
    semantic_kw = semantic_page = semantic_topic = None
    tfidf_kw = tfidf_page = tfidf_topic = None

    if config.embedding.model_name == "tfidf" or config.similarity.mode in {"tfidf", "hybrid"}:
        corpus = keyword_texts + page_texts + topics
        vec, corpus_matrix = build_tfidf_vectorizer(corpus, preprocess_mode=config.preprocess_mode)
        tfidf_kw = vectorize_keywords_tfidf(keyword_texts, vec, preprocess_mode=config.preprocess_mode)
        n_kw = len(keywords)
        n_pg = len(page_texts)
        tfidf_page = (
            vectorize_keywords_tfidf(page_texts, vec, preprocess_mode=config.preprocess_mode) if page_texts else None
        )
        tfidf_topic = vectorize_keywords_tfidf(topics, vec, preprocess_mode=config.preprocess_mode) if topics else None
        if config.embedding.tfidf_svd_components and config.embedding.tfidf_svd_components > 0:
            # build_tfidf_vectorizer returns a sparse/dense matrix; both expose .shape at runtime.
            n_features = int(corpus_matrix.shape[1])  # type: ignore[attr-defined]
            svd_components = min(int(config.embedding.tfidf_svd_components), max(2, n_features - 1))
            if svd_components >= 2 and n_features > svd_components:
                svd = TruncatedSVD(n_components=svd_components, random_state=42)
                svd.fit(corpus_matrix)
                tfidf_kw = svd.transform(tfidf_kw)
                tfidf_page = svd.transform(tfidf_page) if tfidf_page is not None else None
                tfidf_topic = svd.transform(tfidf_topic) if tfidf_topic is not None else None

    if config.embedding.model_name != "tfidf":
        all_texts = keyword_texts + page_texts + topics
        all_vecs = vectorize_keywords_st_configured(all_texts, config.embedding)
        n_kw = len(keywords)
        n_pg = len(page_texts)
        semantic_kw = all_vecs[:n_kw]
        semantic_page = all_vecs[n_kw : n_kw + n_pg] if page_texts else None
        semantic_topic = all_vecs[n_kw + n_pg :] if topics else None

    require_similarity_inputs(config.similarity.mode, semantic_kw, tfidf_kw)
    _report("similarity", 0.40)

    # Declare with the broadest type the three branches can produce so mypy doesn't
    # narrow to ndarray on the first assignment and then reject the tfidf branch.
    kw_vectors: object | None
    page_vectors: object | None
    topic_vectors: object | None

    if config.similarity.mode == "hybrid":
        kw_vectors = compose_hybrid_feature_vectors(
            semantic_kw,
            tfidf_kw,
            semantic_weight=config.similarity.semantic_weight,
            tfidf_weight=config.similarity.tfidf_weight,
        )
        page_vectors = (
            compose_hybrid_feature_vectors(
                semantic_page,
                tfidf_page,
                semantic_weight=config.similarity.semantic_weight,
                tfidf_weight=config.similarity.tfidf_weight,
            )
            if page_texts
            else None
        )
        topic_vectors = (
            compose_hybrid_feature_vectors(
                semantic_topic,
                tfidf_topic,
                semantic_weight=config.similarity.semantic_weight,
                tfidf_weight=config.similarity.tfidf_weight,
            )
            if topics
            else None
        )
    elif config.similarity.mode == "semantic":
        kw_vectors = normalise_rows(semantic_kw)
        page_vectors = normalise_rows(semantic_page) if semantic_page is not None else None
        topic_vectors = normalise_rows(semantic_topic) if semantic_topic is not None else None
    else:
        kw_vectors = tfidf_kw
        page_vectors = tfidf_page
        topic_vectors = tfidf_topic

    if kw_vectors is None:
        raise ValueError("No keyword vectors were generated.")

    serp_overlap = compute_serp_overlap_matrix(df) if "serp_urls" in df.columns else None
    cluster_similarity = None
    if config.clustering.method == "graph":
        semantic_sim = (
            compute_similarity_matrix(normalise_rows(semantic_kw), normalise_rows(semantic_kw))
            if semantic_kw is not None
            else None
        )
        tfidf_sim = compute_similarity_matrix(tfidf_kw, tfidf_kw) if tfidf_kw is not None else None
        cluster_similarity = compose_hybrid_similarity_matrix(semantic_sim, tfidf_sim, serp_overlap, config.similarity)

    # When using transformer embeddings (L2-normalised cosine geometry), default Agglomerative to average+cosine
    # rather than ward+euclidean, which is wrong for unit-norm semantic vectors.
    clust_overrides: dict[str, object] = {"graph_similarity_matrix": cluster_similarity}
    using_semantic = config.embedding.model_name != "tfidf" and config.similarity.mode in {"semantic", "hybrid"}
    if (
        using_semantic
        and config.clustering.method == "agglomerative"
        and config.clustering.agglomerative_linkage == "ward"
        and config.clustering.agglomerative_metric == "euclidean"
    ):
        clust_overrides["agglomerative_linkage"] = "average"
        clust_overrides["agglomerative_metric"] = "cosine"
    clust_cfg = ClusteringConfig(**{**config.clustering.__dict__, **clust_overrides})
    _report("cluster", 0.55)
    labels = cluster_keywords(kw_vectors, config=clust_cfg)
    df["cluster_id"] = labels
    _report("label", 0.70)
    label_df = generate_cluster_labels(df, cfg=config.labeling, keyword_vectors=kw_vectors)
    df = apply_cluster_labels(df, label_df)

    if page_vectors is not None and page_texts:
        _report("page-mapping", 0.78)
        gap_cfg = GapThresholdConfig(
            mode=config.page_mapping.gap_threshold_mode, value=config.page_mapping.gap_threshold
        )
        df = map_keywords_to_pages(
            df,
            kw_vectors,
            page_vectors,
            page_names=page_names,
            page_urls=page_urls,
            gap_config=gap_cfg,
        )

    if topics and topic_vectors is not None:
        sim = compute_similarity_matrix(kw_vectors, topic_vectors)
        best_topic_idx = sim.argmax(axis=1)
        df["primary_topic"] = [topics[i] for i in best_topic_idx]
        df["topic_similarity_score"] = sim.max(axis=1).round(4).tolist()

    _report("score", 0.85)
    df = compute_opportunity_score(df, cfg=config.opportunity)
    df = add_notes(df)
    _report("reduce", 0.92)
    coords = reduce_dimensions(kw_vectors, n_components=3, method=config.reduction, umap_min_dist=config.umap_min_dist)
    quality_df = build_cluster_quality_report(df, keyword_vectors=kw_vectors)
    _report("done", 1.0)

    run_dir: str | None = None
    if config.run_history:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = uuid.uuid4().hex[:8]
        run_dir = os.path.join(config.run_output_root, f"{ts}_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "charts"), exist_ok=True)
        df.to_csv(os.path.join(run_dir, "clustered_keywords.csv"), index=False)
        quality_df.to_csv(os.path.join(run_dir, "cluster_quality_report.csv"), index=False)
        cluster_summary = (
            df.groupby(["cluster_id", "cluster_label"], dropna=False)
            .agg(
                keyword_count=("keyword", "count"),
                total_search_volume=("search_volume", "sum") if "search_volume" in df.columns else ("keyword", "count"),
                avg_opportunity_score=("opportunity_score", "mean")
                if "opportunity_score" in df.columns
                else ("cluster_id", "count"),
            )
            .reset_index()
        )
        cluster_summary.to_csv(os.path.join(run_dir, "cluster_summary.csv"), index=False)
        with open(os.path.join(run_dir, "input_schema.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "keyword_columns": sorted(keywords_df.columns.tolist()),
                    "pages_columns": sorted(pages_df.columns.tolist()) if pages_df is not None else [],
                    "topics_columns": sorted(topics_df.columns.tolist()) if topics_df is not None else [],
                    "row_counts": {
                        "keywords": int(len(keywords_df)),
                        "pages": int(len(pages_df)) if pages_df is not None else 0,
                        "topics": int(len(topics_df)) if topics_df is not None else 0,
                    },
                },
                f,
                indent=2,
            )
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding": config.embedding.__dict__,
                    "similarity": config.similarity.__dict__,
                    "clustering": config.clustering.__dict__,
                    "intent": config.intent.__dict__,
                    "page_mapping": config.page_mapping.__dict__,
                    "opportunity": config.opportunity.__dict__,
                    "labeling": config.labeling.__dict__,
                    "preprocess_mode": config.preprocess_mode,
                    "reduction": config.reduction,
                },
                f,
                indent=2,
            )
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_keywords": len(df),
                    "n_clusters": int(df["cluster_id"].nunique()),
                    "outliers": int((df["cluster_id"] == -1).sum()) if "cluster_id" in df.columns else 0,
                },
                f,
                indent=2,
            )

    return PipelineResult(
        df=df,
        keyword_vectors=kw_vectors,
        page_vectors=page_vectors,
        topic_vectors=topic_vectors,
        labels=labels,
        coords=coords,
        metrics={
            "n_keywords": len(df),
            "n_clusters": int(df["cluster_id"].nunique()),
            "similarity_mode": config.similarity.mode,
            "embedding_model": config.embedding.model_name,
            "opportunity_profile": config.opportunity.profile,
            "labeling_strategy": config.labeling.strategy,
        },
        run_dir=run_dir,
    )
