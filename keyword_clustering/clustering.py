"""Clustering algorithms and dimensionality reduction."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from .vectorization import to_dense


def cluster_keywords(
    vectors: object,
    method: str = "kmeans",
    n_clusters: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    """
    Assign integer cluster labels to each keyword vector.

    method: 'kmeans' | 'agglomerative' | 'hdbscan'
    Returns array of shape (n_keywords,) with integer labels.
    """
    dense = to_dense(vectors)
    n_clusters = min(n_clusters, dense.shape[0])

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        return model.fit_predict(dense)

    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return model.fit_predict(dense)

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except ImportError as exc:
            raise ImportError("Install hdbscan: pip install hdbscan") from exc
        model = hdbscan.HDBSCAN(min_cluster_size=max(2, len(dense) // n_clusters))
        return model.fit_predict(dense)

    raise ValueError(f"Unknown clustering method: {method!r}. Choose kmeans, agglomerative, or hdbscan.")


def reduce_dimensions(
    vectors: object,
    n_components: int = 3,
    method: str = "pca",
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce high-dimensional vectors to n_components dimensions for plotting.

    method: 'pca' | 'umap'
    """
    dense = to_dense(vectors)
    n_samples = dense.shape[0]
    actual_components = min(n_components, n_samples, dense.shape[1])

    if method == "pca":
        return PCA(n_components=actual_components, random_state=random_state).fit_transform(dense)

    if method == "umap":
        try:
            from umap import UMAP  # type: ignore
        except ImportError as exc:
            raise ImportError("Install umap-learn: pip install umap-learn") from exc
        return UMAP(n_components=actual_components, random_state=random_state).fit_transform(dense)

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, n_samples - 1)
        return TSNE(
            n_components=actual_components,
            random_state=random_state,
            perplexity=perplexity,
        ).fit_transform(dense)

    raise ValueError(f"Unknown reduction method: {method!r}. Choose pca, umap, or tsne.")
