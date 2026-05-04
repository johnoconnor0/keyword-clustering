"""Tests for clustering module."""

import numpy as np
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


def _get_vectors():
    vec, _ = build_tfidf_vectorizer(KEYWORDS)
    return vectorize_keywords_tfidf(KEYWORDS, vec)


def test_kmeans_label_count():
    vectors = _get_vectors()
    labels = cluster_keywords(vectors, method="kmeans", n_clusters=3, random_state=42)
    assert len(labels) == len(KEYWORDS)
    # All three clusters must actually be produced — not just label-set subset.
    assert set(labels) == {0, 1, 2}
    # Sanity: each cluster has at least one member.
    for cid in (0, 1, 2):
        assert (labels == cid).sum() >= 1


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


@pytest.mark.parametrize("n_components", [2, 3])
def test_pca_reduction_shape(n_components: int):
    vectors = _get_vectors()
    coords = reduce_dimensions(vectors, n_components=n_components, method="pca")
    assert coords.shape == (len(KEYWORDS), n_components)


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
    # HDBSCAN labels are non-negative integers, with -1 reserved for noise.
    assert labels.dtype.kind in "iu"
    assert (labels >= -1).all()
    # At least one of: a real cluster (>=0) or a noise marker (-1) is produced.
    assert set(labels).issubset(set(range(-1, len(KEYWORDS) + 1)))


def test_agglomerative_invalid_metric_linkage_combo():
    vectors = _get_vectors()
    with pytest.raises(ValueError, match="requires metric='euclidean'"):
        cluster_keywords(
            vectors,
            config=ClusteringConfig(
                method="agglomerative",
                n_clusters=3,
                agglomerative_metric="cosine",
                agglomerative_linkage="ward",
            ),
        )


def test_graph_method_runs():
    vectors = _get_vectors()
    labels = cluster_keywords(vectors, config=ClusteringConfig(method="graph", graph_k=4, graph_min_similarity=0.05))
    assert len(labels) == len(KEYWORDS)
    # Graph clustering should produce more than one community on this 12-keyword
    # corpus split across three semantic groups (SEO / web design / content).
    assert len(set(labels)) >= 2


def test_graph_method_ann_flag_falls_back():
    """When hnswlib isn't installed, ANN flag must silently fall back to sklearn kNN
    and produce labels equivalent in count to the non-ANN path."""
    vectors = _get_vectors()
    labels_no_ann = cluster_keywords(
        vectors, config=ClusteringConfig(method="graph", graph_k=4, graph_min_similarity=0.05, random_state=42)
    )
    labels_ann = cluster_keywords(
        vectors,
        config=ClusteringConfig(
            method="graph", graph_k=4, graph_min_similarity=0.05, graph_use_ann=True, random_state=42
        ),
    )
    assert len(labels_ann) == len(KEYWORDS)
    # Same number of communities under both paths (community membership may permute).
    assert len(set(labels_no_ann)) == len(set(labels_ann))


def test_auto_k_silhouette_chooses_valid_labels():
    vectors = _get_vectors()
    labels = cluster_keywords(
        vectors,
        config=ClusteringConfig(method="kmeans", n_clusters=8, auto_k="silhouette", k_min=2, k_max=5),
    )
    assert len(labels) == len(KEYWORDS)
    assert 2 <= len(set(labels)) <= 5
