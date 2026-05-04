"""Tests for shared pipeline service."""

from pathlib import Path

import numpy as np
import pandas as pd

from keyword_clustering.clustering import ClusteringConfig
from keyword_clustering.pipeline import PageMappingConfig, PipelineConfig, run_keyword_clustering
from keyword_clustering.preprocessing import IntentConfig
from keyword_clustering.scoring import SimilarityConfig
from keyword_clustering.vectorization import EmbeddingConfig


def _sample_keywords() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "keyword": ["seo audit", "technical seo audit", "web design agency", "website redesign pricing"],
            "search_volume": [1000, 800, 1200, 700],
            "keyword_difficulty": [35, 40, 55, 60],
        }
    )


def _sample_pages() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "url": ["/seo-audit", "/web-design"],
            "page_name": ["SEO Audit", "Web Design Services"],
            "title": ["SEO Audit Services", "Web Design Agency Services"],
            "h1": ["Technical SEO Audit", "Website Design and Redesign"],
        }
    )


def test_pipeline_runs_with_tfidf():
    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="tfidf"),
        similarity=SimilarityConfig(mode="tfidf"),
        clustering=ClusteringConfig(method="kmeans", n_clusters=2),
        intent=IntentConfig(mode="rules"),
        page_mapping=PageMappingConfig(gap_threshold=0.2, gap_threshold_mode="fixed"),
        preprocess_mode="stem",
    )
    result = run_keyword_clustering(_sample_keywords(), _sample_pages(), None, cfg)
    assert "cluster_id" in result.df.columns
    assert "recommended_page" in result.df.columns
    assert "match_confidence" in result.df.columns


def test_pipeline_deterministic_for_same_config():
    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="tfidf"),
        similarity=SimilarityConfig(mode="tfidf"),
        clustering=ClusteringConfig(method="kmeans", n_clusters=2, random_state=42),
        intent=IntentConfig(mode="rules"),
        preprocess_mode="stem",
    )
    d1 = run_keyword_clustering(_sample_keywords(), _sample_pages(), None, cfg).df["cluster_id"].tolist()
    d2 = run_keyword_clustering(_sample_keywords(), _sample_pages(), None, cfg).df["cluster_id"].tolist()
    assert d1 == d2


def test_pipeline_hybrid_similarity_for_kmeans(monkeypatch):
    def fake_st_vectors(texts, cfg):
        # deterministic semantic vectors for pipeline hybrid path
        base = np.arange(len(texts), dtype=float).reshape(-1, 1)
        return np.hstack([base, np.ones((len(texts), 1))])

    monkeypatch.setattr("keyword_clustering.pipeline.vectorize_keywords_st_configured", fake_st_vectors)

    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2"),
        similarity=SimilarityConfig(mode="hybrid", semantic_weight=0.6, tfidf_weight=0.4),
        clustering=ClusteringConfig(method="kmeans", n_clusters=2, random_state=42),
        intent=IntentConfig(mode="rules"),
        preprocess_mode="light",
    )
    result = run_keyword_clustering(_sample_keywords(), _sample_pages(), None, cfg)
    assert "cluster_id" in result.df.columns
    assert result.keyword_vectors is not None


def test_run_history_writes_expected_artifacts(tmp_path: Path):
    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="tfidf"),
        similarity=SimilarityConfig(mode="tfidf"),
        clustering=ClusteringConfig(method="kmeans", n_clusters=2),
        intent=IntentConfig(mode="rules"),
        page_mapping=PageMappingConfig(gap_threshold=0.2, gap_threshold_mode="fixed"),
        preprocess_mode="stem",
        run_history=True,
        run_output_root=str(tmp_path / "runs"),
    )
    result = run_keyword_clustering(_sample_keywords(), _sample_pages(), None, cfg)
    assert result.run_dir is not None
    run_dir = Path(result.run_dir)
    for expected in (
        "clustered_keywords.csv",
        "cluster_quality_report.csv",
        "cluster_summary.csv",
        "config.json",
        "metrics.json",
        "input_schema.json",
    ):
        assert (run_dir / expected).exists()
