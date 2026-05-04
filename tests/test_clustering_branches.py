"""Branch coverage for keyword_clustering.clustering — auto-k strategies, UMAP, t-SNE,
and the n_clusters=1 warning that the audit explicitly enumerated."""

from __future__ import annotations

import logging

import pytest

from keyword_clustering.clustering import ClusteringConfig, cluster_keywords, reduce_dimensions
from keyword_clustering.vectorization import build_tfidf_vectorizer, vectorize_keywords_tfidf

KEYWORDS = [
    "SEO audit",
    "keyword research",
    "backlink analysis",
    "on-page SEO",
    "website design",
    "UI development",
    "web application",
    "responsive design",
    "content strategy",
    "blog writing",
    "copywriting",
    "editorial calendar",
]


def _vectors() -> object:
    vec, _ = build_tfidf_vectorizer(KEYWORDS)
    return vectorize_keywords_tfidf(KEYWORDS, vec)


def test_auto_k_calinski_harabasz_chooses_valid_labels():
    labels = cluster_keywords(
        _vectors(),
        config=ClusteringConfig(method="kmeans", n_clusters=8, auto_k="calinski_harabasz", k_min=2, k_max=5),
    )
    assert len(labels) == len(KEYWORDS)
    assert 2 <= len(set(labels)) <= 5


def test_n_clusters_one_emits_warning(caplog: pytest.LogCaptureFixture):
    """n_clusters=1 with kmeans must log a WARNING about cluster collapse."""
    with caplog.at_level(logging.WARNING, logger="keyword_clustering.clustering"):
        labels = cluster_keywords(_vectors(), method="kmeans", n_clusters=1)
    assert (labels == 0).all()
    assert any("n_clusters=1" in rec.message for rec in caplog.records)


def test_reduce_dimensions_umap_returns_correct_shape():
    pytest.importorskip("umap")
    coords = reduce_dimensions(_vectors(), n_components=3, method="umap")
    assert coords.shape == (len(KEYWORDS), 3)


def test_reduce_dimensions_tsne_returns_correct_shape():
    coords = reduce_dimensions(_vectors(), n_components=2, method="tsne")
    assert coords.shape == (len(KEYWORDS), 2)
    # t-SNE preserves the number of samples even when perplexity needs auto-clamping.
    assert coords.dtype.kind == "f"


def test_reduce_dimensions_handles_single_sample():
    """Edge case: a 1-sample input must not crash any reduction backend."""
    vec, _ = build_tfidf_vectorizer(["only one"])
    one = vectorize_keywords_tfidf(["only one"], vec)
    coords = reduce_dimensions(one, n_components=3, method="pca")
    assert coords.shape[0] == 1
    assert (coords == 0).all()
