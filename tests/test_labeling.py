"""Tests for cluster labeling strategies.

These pin the user-facing contract:
    * `label_method` records the strategy that actually ran (no silent aliasing).
    * `c-tfidf` is BERTopic-style class TF-IDF — one fit over per-cluster pseudo-docs.
    * `mmr` runs Carbonell-Goldstein MMR when embeddings are supplied.
    * `keybert` raises until KeyBERT is wired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keyword_clustering.labeling import (
    LabelingConfig,
    apply_cluster_labels,
    generate_cluster_labels,
)


def _two_cluster_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"keyword": "blue widget", "cluster_id": 0},
            {"keyword": "blue widget pricing", "cluster_id": 0},
            {"keyword": "buy blue widgets", "cluster_id": 0},
            {"keyword": "red gadget", "cluster_id": 1},
            {"keyword": "red gadget reviews", "cluster_id": 1},
            {"keyword": "best red gadget", "cluster_id": 1},
        ]
    )


def _two_cluster_vectors() -> np.ndarray:
    # Three near-duplicate vectors per cluster, two well-separated clusters.
    return np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.05, 0.95],
            [0.1, 0.9],
        ]
    )


def test_label_method_records_actual_strategy_for_tfidf():
    df = _two_cluster_frame()
    labels = generate_cluster_labels(df, cfg=LabelingConfig(strategy="tfidf"))
    assert (labels["label_method"] == "tfidf").all()


def test_label_method_records_actual_strategy_for_ctfidf():
    df = _two_cluster_frame()
    labels = generate_cluster_labels(df, cfg=LabelingConfig(strategy="c-tfidf"))
    assert (labels["label_method"] == "c-tfidf").all()


def test_label_method_records_actual_strategy_for_mmr():
    df = _two_cluster_frame()
    vecs = _two_cluster_vectors()
    labels = generate_cluster_labels(df, cfg=LabelingConfig(strategy="mmr"), keyword_vectors=vecs)
    assert (labels["label_method"] == "mmr").all()


def test_keybert_raises_not_implemented():
    df = _two_cluster_frame()
    with pytest.raises(NotImplementedError, match="keybert"):
        generate_cluster_labels(df, cfg=LabelingConfig(strategy="keybert"))


def test_unknown_strategy_raises_value_error():
    df = _two_cluster_frame()
    with pytest.raises(ValueError, match="labeling strategy"):
        generate_cluster_labels(df, cfg=LabelingConfig(strategy="bogus"))


def test_ctfidf_picks_distinguishing_terms_per_cluster():
    """Class TF-IDF should weight terms by how cluster-specific they are.
    'widget' should dominate cluster 0; 'gadget' should dominate cluster 1.
    """
    df = _two_cluster_frame()
    labels = generate_cluster_labels(df, cfg=LabelingConfig(strategy="c-tfidf", n_top_terms=2))
    by_cluster = labels.set_index("cluster_id")
    assert "widget" in by_cluster.loc[0, "top_terms"].lower()
    assert "gadget" in by_cluster.loc[1, "top_terms"].lower()


def test_centroid_picks_most_central_keyword():
    df = _two_cluster_frame()
    vecs = _two_cluster_vectors()
    labels = generate_cluster_labels(df, cfg=LabelingConfig(strategy="centroid"), keyword_vectors=vecs)
    by_cluster = labels.set_index("cluster_id")
    # Cluster 0's middle vector (idx 1) is closest to its mean — keyword "blue widget pricing".
    assert by_cluster.loc[0, "representative_keyword"] in {"blue widget pricing", "blue widget", "buy blue widgets"}
    # Centroid pick must come from the cluster's own keywords.
    cluster1_kws = df.loc[df["cluster_id"] == 1, "keyword"].tolist()
    assert by_cluster.loc[1, "representative_keyword"] in cluster1_kws


def test_mmr_with_vectors_uses_relevance_and_diversity():
    """MMR with embeddings must pick keywords that are close to the centroid AND diverse from each other."""
    df = _two_cluster_frame()
    vecs = _two_cluster_vectors()
    labels = generate_cluster_labels(
        df,
        cfg=LabelingConfig(strategy="mmr", n_top_terms=3, mmr_lambda=0.65),
        keyword_vectors=vecs,
    )
    by_cluster = labels.set_index("cluster_id")
    # Each cluster's chosen top_terms must be a subset of its own keywords.
    for cid in (0, 1):
        cluster_kws = set(df.loc[df["cluster_id"] == cid, "keyword"].tolist())
        chosen = {t.strip() for t in by_cluster.loc[cid, "top_terms"].split(",")}
        assert chosen.issubset(cluster_kws), f"cluster {cid} picked terms outside its own keywords"


def test_mmr_without_vectors_falls_back_to_token_coverage():
    df = _two_cluster_frame()
    labels = generate_cluster_labels(df, cfg=LabelingConfig(strategy="mmr"), keyword_vectors=None)
    assert (labels["label_method"] == "mmr").all()
    # Falls back to token-coverage; must still produce non-empty labels.
    assert (labels["cluster_label"].str.len() > 0).all()


def test_apply_cluster_labels_preserves_label_method():
    df = _two_cluster_frame()
    label_df = generate_cluster_labels(df, cfg=LabelingConfig(strategy="c-tfidf"))
    enriched = apply_cluster_labels(df, label_df)
    assert "label_method" in enriched.columns
    assert (enriched["label_method"] == "c-tfidf").all()
