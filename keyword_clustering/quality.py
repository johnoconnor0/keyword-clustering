"""Cluster-quality diagnostics that don't fit cleanly in scoring.py.

The bootstrap stability metric is the only resident here today; future additions
should be metrics that *evaluate* an existing clustering rather than produce or
score keywords.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_cluster_stability(
    vectors: object,
    base_labels: np.ndarray | pd.Series,
    method: str = "kmeans",
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> dict[int, float]:
    """Hennig-style bootstrap cluster stability.

    Subsample the rows (with replacement), re-cluster the subsample using the same
    method + n_clusters, and for each base cluster compute the maximum Jaccard
    overlap with any bootstrap cluster. The reported stability for cluster c is
    the mean of these max-Jaccard values across the bootstrap iterations.

    Returns a dict mapping cluster_id → stability score in [0, 1].
    """
    # Late import to avoid a clustering↔quality cycle at import time.
    from .clustering import ClusteringConfig, cluster_keywords
    from .vectorization import to_dense

    dense = to_dense(vectors)
    n = dense.shape[0]
    base = np.asarray(base_labels)
    rng = np.random.default_rng(random_state)
    cluster_ids = sorted({int(c) for c in base if int(c) != -1})
    if not cluster_ids:
        logger.warning(
            "compute_cluster_stability: no clusters to score (all labels are noise / -1). "
            "Returning an empty stability dict."
        )
        return {}
    if n < 4:
        logger.warning(
            "compute_cluster_stability: only %d sample(s) present; bootstrap resampling needs n >= 4. "
            "Returning NaN for every cluster.",
            n,
        )
        return {cid: float("nan") for cid in cluster_ids}
    if n_bootstrap <= 0:
        logger.warning(
            "compute_cluster_stability: n_bootstrap=%d is not positive; returning NaN for every cluster.",
            n_bootstrap,
        )
        return {cid: float("nan") for cid in cluster_ids}

    n_unique = len({int(c) for c in base if int(c) != -1})
    bootstrap_scores: dict[int, list[float]] = {cid: [] for cid in cluster_ids}
    for _ in range(n_bootstrap):
        idx = np.asarray(rng.choice(n, size=n, replace=True))
        boot_dense = dense[idx]
        try:
            boot_labels = cluster_keywords(
                boot_dense,
                config=ClusteringConfig(method=method, n_clusters=max(2, n_unique), random_state=random_state),
            )
        except (ValueError, RuntimeError):
            continue
        for cid in cluster_ids:
            base_members = set(np.where(base == cid)[0].tolist())
            best_jaccard = 0.0
            for boot_cid in {int(c) for c in boot_labels if int(c) != -1}:
                # Map boot indices back to original positions via idx[boot_member_pos].
                boot_member_pos = np.where(boot_labels == boot_cid)[0]
                boot_orig = set(idx[boot_member_pos].tolist())
                union = len(base_members | boot_orig)
                if union == 0:
                    continue
                jaccard = len(base_members & boot_orig) / union
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
            bootstrap_scores[cid].append(best_jaccard)
    return {
        cid: float(round(sum(scores) / len(scores), 4)) if scores else float("nan")
        for cid, scores in bootstrap_scores.items()
    }
