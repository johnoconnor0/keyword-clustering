"""Regression tests guarding against label-vs-positional index confusion.

The pipeline depends on positional indexing into NumPy arrays (embedding
matrices, similarity matrices, dimensionality-reduced coordinates) while
operating on pandas DataFrames whose index labels are not always 0..N-1.
These tests pin the contract: every entry point that fans out to numpy
must work correctly when the input DataFrame has a non-RangeIndex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keyword_clustering.labeling import LabelingConfig, generate_cluster_labels
from keyword_clustering.preprocessing import IntentConfig, enrich_keywords
from keyword_clustering.scoring import build_cluster_quality_report


def _shifted_index_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame whose index labels start at 1000 — a common shape after .query() / .filter()."""
    return pd.DataFrame(rows, index=range(1000, 1000 + len(rows)))


def test_enrich_keywords_handles_non_range_index_with_manual_intent():
    df = _shifted_index_df(
        [
            {"keyword": "buy seo tools", "intent": "transactional"},
            {"keyword": "best seo agency", "intent": None},
            {"keyword": "how to do seo", "intent": "informational"},
        ]
    )

    out = enrich_keywords(df.copy(), brand_terms=[], intent_config=IntentConfig(mode="rules"))

    assert list(out["intent"]) == ["transactional", "commercial", "informational"]
    # Manual values must be honoured for rows where they were supplied.
    assert list(out["intent_mode_used"]) == ["manual", "rules", "manual"]
    assert list(out["intent_confidence"]) == [1.0, 0.7, 1.0]


def test_enrich_keywords_handles_non_range_index_without_manual_column():
    df = _shifted_index_df(
        [
            {"keyword": "buy seo tools"},
            {"keyword": "best seo agency"},
            {"keyword": "how to do seo"},
        ]
    )

    out = enrich_keywords(df.copy(), brand_terms=["acme"], intent_config=IntentConfig(mode="rules"))

    assert list(out["intent"]) == ["transactional", "commercial", "informational"]
    assert list(out["branded"]) == [False, False, False]


def test_generate_cluster_labels_centroid_handles_non_range_index():
    df = _shifted_index_df(
        [
            {"keyword": "blue widgets", "cluster_id": 0},
            {"keyword": "blue widget pricing", "cluster_id": 0},
            {"keyword": "red gadget", "cluster_id": 1},
            {"keyword": "red gadget reviews", "cluster_id": 1},
        ]
    )
    # Vectors are positional. If labeling slices `dense` by labels (1000, 1001, ...) it crashes.
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ]
    )

    labels_df = generate_cluster_labels(df, cfg=LabelingConfig(strategy="centroid"), keyword_vectors=vectors)

    assert set(labels_df["cluster_id"]) == {0, 1}
    # Each cluster's representative must be one of its own keywords.
    for _, row in labels_df.iterrows():
        cluster_keywords = df.loc[df["cluster_id"] == row["cluster_id"], "keyword"].tolist()
        assert row["representative_keyword"] in cluster_keywords


def test_build_cluster_quality_report_handles_non_range_index():
    df = _shifted_index_df(
        [
            {"keyword": "blue widgets", "cluster_id": 0, "cluster_label": "blue"},
            {"keyword": "blue widget pricing", "cluster_id": 0, "cluster_label": "blue"},
            {"keyword": "red gadget", "cluster_id": 1, "cluster_label": "red"},
            {"keyword": "red gadget reviews", "cluster_id": 1, "cluster_label": "red"},
        ]
    )
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ]
    )

    report = build_cluster_quality_report(df, keyword_vectors=vectors)

    assert {0, 1}.issubset(set(report["cluster_id"]))
    # Intra-cluster similarity must be high (we constructed near-duplicates per cluster).
    intra = report.set_index("cluster_id")["avg_intra_cluster_similarity"]
    assert float(intra.loc[0]) > 0.9
    assert float(intra.loc[1]) > 0.9
