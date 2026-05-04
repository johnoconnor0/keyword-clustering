"""Tests for clustering module."""

import numpy as np
import pytest

from keyword_clustering.clustering import cluster_keywords, reduce_dimensions
from keyword_clustering.vectorization import build_tfidf_vectorizer, vectorize_keywords_tfidf


KEYWORDS = [
    "SEO audit", "keyword research", "backlink analysis", "on-page SEO",
    "website design", "UI development", "web application", "responsive design",
    "content strategy", "blog writing", "copywriting", "editorial calendar",
]


def _get_vectors():
    vec, _ = build_tfidf_vectorizer(KEYWORDS)
    return vectorize_keywords_tfidf(KEYWORDS, vec)


def test_kmeans_label_count():
    vectors = _get_vectors()
    labels = cluster_keywords(vectors, method="kmeans", n_clusters=3, random_state=42)
    assert len(labels) == len(KEYWORDS)
    assert set(labels).issubset({0, 1, 2})


def test_kmeans_deterministic():
    vectors = _get_vectors()
    labels1 = cluster_keywords(vectors, method="kmeans", n_clusters=3, random_state=42)
    labels2 = cluster_keywords(vectors, method="kmeans", n_clusters=3, random_state=42)
    np.testing.assert_array_equal(labels1, labels2)


def test_agglomerative_label_count():
    vectors = _get_vectors()
    labels = cluster_keywords(vectors, method="agglomerative", n_clusters=3)
    assert len(labels) == len(KEYWORDS)
    unique = set(labels)
    assert len(unique) == 3


def test_unknown_method_raises():
    vectors = _get_vectors()
    with pytest.raises(ValueError, match="Unknown clustering method"):
        cluster_keywords(vectors, method="badmethod")


def test_n_clusters_capped_to_samples():
    tiny = ["seo", "web"]
    vec, _ = build_tfidf_vectorizer(tiny)
    vecs = vectorize_keywords_tfidf(tiny, vec)
    labels = cluster_keywords(vecs, method="kmeans", n_clusters=10)
    # Should not crash; number of clusters can't exceed n_samples
    assert len(labels) == 2


def test_pca_reduction_shape():
    vectors = _get_vectors()
    coords = reduce_dimensions(vectors, n_components=3, method="pca")
    assert coords.shape == (len(KEYWORDS), 3)


def test_pca_reduction_2d():
    vectors = _get_vectors()
    coords = reduce_dimensions(vectors, n_components=2, method="pca")
    assert coords.shape[1] == 2


def test_unknown_reduction_raises():
    vectors = _get_vectors()
    with pytest.raises(ValueError, match="Unknown reduction method"):
        reduce_dimensions(vectors, method="badreduction")


def test_hdbscan_returns_labels():
    pytest.importorskip("hdbscan")
    vectors = _get_vectors()
    labels = cluster_keywords(vectors, method="hdbscan")
    assert len(labels) == len(KEYWORDS)
    assert isinstance(labels, np.ndarray)
