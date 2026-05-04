"""Branch coverage for keyword_clustering.pipeline that the audit explicitly enumerated.

Covers: missing keyword/url columns, topics path, similarity.mode='semantic' branch,
SVD branch, run_history content round-trip, progress callback contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from keyword_clustering.clustering import ClusteringConfig
from keyword_clustering.pipeline import (
    PageMappingConfig,
    PipelineConfig,
    run_keyword_clustering,
)
from keyword_clustering.preprocessing import IntentConfig
from keyword_clustering.scoring import SimilarityConfig
from keyword_clustering.vectorization import EmbeddingConfig


def _kw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "keyword": ["seo audit", "keyword research", "site speed", "blog writing", "internal linking"],
            "search_volume": [1000, 800, 600, 400, 300],
        }
    )


def _pages_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "url": ["/seo", "/blog"],
            "page_name": ["SEO", "Blog"],
            "title": ["SEO Services", "Blog Posts"],
        }
    )


def _topics_df() -> pd.DataFrame:
    return pd.DataFrame({"topic": ["SEO", "Content"]})


def test_missing_keyword_column_raises_value_error():
    df = pd.DataFrame({"oops": ["seo"]})
    with pytest.raises(ValueError, match="'keyword'"):
        run_keyword_clustering(df, None, None, PipelineConfig())


def test_missing_url_column_in_pages_raises_value_error():
    bad_pages = pd.DataFrame({"page_name": ["SEO"]})  # no `url`
    with pytest.raises(ValueError, match="'url'"):
        run_keyword_clustering(_kw_df(), bad_pages, None, PipelineConfig())


def test_topics_path_writes_primary_topic_and_similarity_columns():
    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="tfidf"),
        similarity=SimilarityConfig(mode="tfidf"),
        clustering=ClusteringConfig(method="kmeans", n_clusters=2),
        intent=IntentConfig(mode="rules"),
        page_mapping=PageMappingConfig(gap_threshold=0.25, gap_threshold_mode="fixed"),
    )
    result = run_keyword_clustering(_kw_df(), None, _topics_df(), cfg)
    assert "primary_topic" in result.df.columns
    assert "topic_similarity_score" in result.df.columns
    assert set(result.df["primary_topic"]).issubset({"SEO", "Content"})


def test_svd_branch_reduces_tfidf_dimensions(monkeypatch: pytest.MonkeyPatch):
    """When tfidf_svd_components > 0 and the corpus has more features than components,
    the SVD branch should fire and the resulting kw_vectors should be dense."""
    df = pd.DataFrame({"keyword": [f"keyword phrase number {i}" for i in range(20)]})
    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="tfidf", tfidf_svd_components=4),
        similarity=SimilarityConfig(mode="tfidf"),
        clustering=ClusteringConfig(method="kmeans", n_clusters=3),
        intent=IntentConfig(mode="rules"),
    )
    result = run_keyword_clustering(df, None, None, cfg)
    # After SVD, kw_vectors must be 2-D ndarray with the requested component count.
    assert hasattr(result.keyword_vectors, "shape")
    assert result.keyword_vectors.shape[1] == 4


def test_semantic_only_similarity_requires_transformer():
    """similarity=semantic with embedding=tfidf must raise via require_similarity_inputs."""
    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="tfidf"),
        similarity=SimilarityConfig(mode="semantic"),
        clustering=ClusteringConfig(method="kmeans", n_clusters=2),
        intent=IntentConfig(mode="rules"),
    )
    with pytest.raises(ValueError, match="similarity=semantic requires semantic"):
        run_keyword_clustering(_kw_df(), None, None, cfg)


def test_run_history_writes_round_trippable_metadata(tmp_path: Path):
    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="tfidf"),
        similarity=SimilarityConfig(mode="tfidf"),
        clustering=ClusteringConfig(method="kmeans", n_clusters=2),
        intent=IntentConfig(mode="rules"),
        run_history=True,
        run_output_root=str(tmp_path / "runs"),
    )
    result = run_keyword_clustering(_kw_df(), _pages_df(), None, cfg)
    assert result.run_dir is not None
    run_dir = Path(result.run_dir)

    import json as _json

    config = _json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["embedding"]["model_name"] == "tfidf"
    assert config["similarity"]["mode"] == "tfidf"
    metrics = _json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_keywords"] == len(_kw_df())
    assert metrics["n_clusters"] == int(result.df["cluster_id"].nunique())
    schema = _json.loads((run_dir / "input_schema.json").read_text(encoding="utf-8"))
    assert "keyword" in schema["keyword_columns"]
    assert "url" in schema["pages_columns"]


def test_progress_callback_fires_at_each_stage_in_order():
    """The progress_callback must receive every named stage in pipeline order with 0 ≤ frac ≤ 1."""
    captured: list[tuple[str, float]] = []

    def cb(stage: str, fraction: float) -> None:
        captured.append((stage, fraction))

    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="tfidf"),
        similarity=SimilarityConfig(mode="tfidf"),
        clustering=ClusteringConfig(method="kmeans", n_clusters=2),
        intent=IntentConfig(mode="rules"),
    )
    run_keyword_clustering(_kw_df(), _pages_df(), None, cfg, progress_callback=cb)

    stages = [s for s, _ in captured]
    fractions = [f for _, f in captured]
    expected_stages = ["load", "embed", "similarity", "cluster", "label", "page-mapping", "score", "reduce", "done"]
    for s in expected_stages:
        assert s in stages, f"expected stage {s} missing from progress events"
    # Fractions are non-decreasing.
    assert fractions == sorted(fractions)
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions[-1] == 1.0


def test_progress_callback_failure_does_not_break_pipeline():
    """A buggy callback must not propagate up — pipeline runs through anyway."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("oops")

    cfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name="tfidf"),
        similarity=SimilarityConfig(mode="tfidf"),
        clustering=ClusteringConfig(method="kmeans", n_clusters=2),
        intent=IntentConfig(mode="rules"),
    )
    # Should NOT raise.
    result = run_keyword_clustering(_kw_df(), None, None, cfg, progress_callback=boom)
    assert "cluster_id" in result.df.columns
