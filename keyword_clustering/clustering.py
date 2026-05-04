"""Clustering algorithms and dimensionality reduction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .vectorization import to_dense

logger = logging.getLogger(__name__)

_SILHOUETTE_SAMPLE_THRESHOLD = 5_000  # rows above which we subsample for speed
_SILHOUETTE_SAMPLE_SIZE = 2_000


@dataclass(frozen=True)
class ClusteringConfig:
    method: str = "kmeans"
    n_clusters: int = 8
    random_state: int = 42
    use_minibatch: bool = False
    auto_k: str | None = None  # one of: None, "silhouette", "calinski_harabasz"
    k_min: int = 4
    k_max: int = 40
    agglomerative_metric: str = "euclidean"
    agglomerative_linkage: str = "ward"
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 2
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"
    graph_k: int = 10
    graph_min_similarity: float = 0.55
    community_algorithm: str = "louvain"
    graph_use_ann: bool = False
    graph_ann_ef: int = 100
    graph_similarity_matrix: np.ndarray | None = None
    n_jobs: int = -1  # -1 = use all available cores; 1 = single-threaded; 0 → sklearn default
    umap_min_dist: float = 0.1  # BERTopic-style pipelines want 0.0; default matches umap-learn.


def _choose_k(dense: np.ndarray, cfg: ClusteringConfig) -> int:
    kmin = max(2, min(cfg.k_min, dense.shape[0] - 1))
    kmax = max(kmin, min(cfg.k_max, dense.shape[0] - 1))
    best_k = min(cfg.n_clusters, dense.shape[0])
    best_score = -np.inf
    method = (cfg.auto_k or "silhouette").lower()
    n = dense.shape[0]
    sample_size = _SILHOUETTE_SAMPLE_SIZE if n > _SILHOUETTE_SAMPLE_THRESHOLD else None
    for k in range(kmin, kmax + 1):
        model = KMeans(n_clusters=k, random_state=cfg.random_state, n_init="auto")
        labels = model.fit_predict(dense)
        if len(set(labels)) <= 1:
            continue
        if method == "calinski_harabasz":
            score = calinski_harabasz_score(dense, labels)
        else:
            score = silhouette_score(
                dense,
                labels,
                metric="cosine",
                sample_size=sample_size,
                random_state=cfg.random_state,
                n_jobs=cfg.n_jobs,
            )
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _graph_cluster(dense: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    import networkx as nx

    n = dense.shape[0]
    if n == 1:
        return np.array([0], dtype=int)
    k = min(max(2, cfg.graph_k), n - 1)

    g = nx.Graph()
    g.add_nodes_from(range(n))

    if cfg.graph_similarity_matrix is not None:
        # Pre-computed full N×N similarity matrix path (used by hybrid graph clustering).
        # Iterate the upper triangle once.
        sim = cfg.graph_similarity_matrix
        rows, cols = np.where(np.triu(sim >= cfg.graph_min_similarity, k=1))
        for i, j in zip(rows.tolist(), cols.tolist()):
            g.add_edge(int(i), int(j), weight=float(sim[i, j]))
    else:
        # Sparse kNN path — only iterate edges yielded by the nearest-neighbours search.
        distances = indices = None
        if cfg.graph_use_ann:
            try:
                import hnswlib  # type: ignore

                index = hnswlib.Index(space="cosine", dim=dense.shape[1])
                index.init_index(max_elements=n, ef_construction=max(100, cfg.graph_ann_ef), M=16)
                index.add_items(dense, np.arange(n))
                index.set_ef(max(k + 1, cfg.graph_ann_ef))
                ann_labels, dists = index.knn_query(dense, k=k + 1)
                indices = ann_labels
                distances = dists
            except (ImportError, RuntimeError):
                distances = indices = None
        if distances is None or indices is None:
            nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=cfg.n_jobs)
            nn.fit(dense)
            distances, indices = nn.kneighbors(dense)
        # Each edge appears at most twice (i→j and j→i); networkx dedupes via undirected Graph.
        threshold = cfg.graph_min_similarity
        for i in range(n):
            for dist, j in zip(distances[i][1:], indices[i][1:]):
                weight = 1.0 - float(dist)
                if weight < threshold or i == j:
                    continue
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                existing = g.get_edge_data(a, b)
                if existing is None or weight > existing.get("weight", 0.0):
                    g.add_edge(a, b, weight=weight)
    if g.number_of_edges() == 0:
        return np.arange(n, dtype=int)

    algo = cfg.community_algorithm.lower()
    if algo == "leiden":
        try:
            import igraph as ig  # type: ignore
            import leidenalg  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Leiden clustering requires igraph and leidenalg.") from exc
        ig_graph = ig.Graph.TupleList(g.edges(), directed=False, edge_attrs=["weight"])
        part = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=ig_graph.es["weight"],
        )
        labels = np.full(n, -1, dtype=int)
        for cid, members in enumerate(part):
            for node in members:
                labels[int(node)] = cid
        return labels

    try:
        communities = nx.community.louvain_communities(g, weight="weight", seed=cfg.random_state)
    except (ImportError, AttributeError):  # pragma: no cover
        # NetworkX builds without optional community-detection support fall back to greedy modularity.
        communities = list(nx.community.greedy_modularity_communities(g, weight="weight"))
    labels = np.full(n, -1, dtype=int)
    for cid, nodes in enumerate(communities):
        for node in nodes:
            labels[int(node)] = cid
    return labels


def cluster_keywords(
    vectors: object,
    method: str = "kmeans",
    n_clusters: int = 8,
    random_state: int = 42,
    config: ClusteringConfig | None = None,
) -> np.ndarray:
    """Assign integer cluster labels to each keyword vector."""
    dense = to_dense(vectors)
    cfg = config or ClusteringConfig(
        method=method,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    method = cfg.method
    n_clusters = min(max(1, cfg.n_clusters), dense.shape[0])
    if dense.shape[0] == 1:
        return np.array([0], dtype=int)
    if n_clusters == 1 and method in {"kmeans", "agglomerative"} and cfg.auto_k is None:
        logger.warning(
            "cluster_keywords called with n_clusters=1 (method=%s); every keyword will collapse "
            "into a single cluster. Set --clusters >= 2 or --auto-k to discover k.",
            method,
        )

    if (
        cfg.auto_k in {"silhouette", "calinski_harabasz"}
        and method in {"kmeans", "agglomerative"}
        and dense.shape[0] > 2
    ):
        n_clusters = _choose_k(dense, cfg)

    if method == "kmeans":
        if cfg.use_minibatch:
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=cfg.random_state, n_init="auto")
        else:
            model = KMeans(n_clusters=n_clusters, random_state=cfg.random_state, n_init="auto")
        return model.fit_predict(dense)

    if method == "agglomerative":
        if cfg.agglomerative_linkage == "ward" and cfg.agglomerative_metric != "euclidean":
            raise ValueError("Agglomerative linkage='ward' requires metric='euclidean'.")
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=cfg.agglomerative_linkage,
            metric=cfg.agglomerative_metric,
        )
        return model.fit_predict(dense)

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except ImportError as exc:
            raise ImportError("Install hdbscan: pip install hdbscan") from exc
        model = hdbscan.HDBSCAN(
            min_cluster_size=max(2, cfg.hdbscan_min_cluster_size),
            min_samples=max(1, cfg.hdbscan_min_samples),
            metric=cfg.hdbscan_metric,
            cluster_selection_method=cfg.hdbscan_cluster_selection_method,
        )
        return model.fit_predict(dense)

    if method == "graph":
        return _graph_cluster(dense, cfg)

    raise ValueError(f"Unknown clustering method: {method!r}. Choose kmeans, agglomerative, hdbscan, or graph.")


def reduce_dimensions(
    vectors: object,
    n_components: int = 3,
    method: str = "pca",
    random_state: int = 42,
    umap_min_dist: float = 0.1,
) -> np.ndarray:
    """Reduce high-dimensional vectors for plotting."""
    dense = to_dense(vectors)
    n_samples = dense.shape[0]
    n_features = dense.shape[1] if dense.ndim > 1 else 1
    actual_components = min(n_components, max(1, n_samples), max(1, n_features))

    if n_samples <= 1:
        return np.zeros((n_samples, actual_components))

    if method == "pca":
        return PCA(n_components=actual_components, random_state=random_state).fit_transform(dense)

    if method == "umap":
        try:
            from umap import UMAP  # type: ignore
        except ImportError as exc:
            raise ImportError("Install umap-learn: pip install umap-learn") from exc
        n_neighbors = min(15, max(2, n_samples - 1))
        return UMAP(
            n_components=actual_components,
            random_state=random_state,
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=umap_min_dist,
        ).fit_transform(dense)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(1, n_samples - 1))
        return TSNE(
            n_components=actual_components,
            random_state=random_state,
            perplexity=perplexity,
        ).fit_transform(dense)

    raise ValueError(f"Unknown reduction method: {method!r}. Choose pca, umap, or tsne.")
