"""Coverage for keyword_clustering.vectorization helpers the audit flagged."""

from __future__ import annotations

import numpy as np
import pandas as pd

from keyword_clustering.vectorization import (
    compose_hybrid_feature_vectors,
    compute_serp_overlap_matrix,
    normalise_rows,
)


def test_compute_serp_overlap_matrix_jaccard_correctness():
    df = pd.DataFrame(
        {
            "keyword": ["a", "b", "c"],
            "serp_urls": [
                "https://x.com/1|https://x.com/2|https://x.com/3",
                "https://x.com/2|https://x.com/3|https://x.com/4",
                "https://x.com/9",
            ],
        }
    )
    mat = compute_serp_overlap_matrix(df)
    # Matrix is symmetric, diagonal is 1, all values in [0, 1].
    assert mat.shape == (3, 3)
    np.testing.assert_allclose(mat, mat.T)
    np.testing.assert_allclose(np.diag(mat), [1.0, 1.0, 1.0])
    # |a ∩ b| = 2, |a ∪ b| = 4 → 0.5 exactly.
    np.testing.assert_allclose(mat[0, 1], 0.5, atol=1e-6)
    # a vs c share 0 URLs → 0.0
    assert mat[0, 2] == 0.0
    # All values in valid Jaccard range
    assert (mat >= 0).all() and (mat <= 1).all()


def test_compute_serp_overlap_matrix_handles_missing_column():
    df = pd.DataFrame({"keyword": ["a", "b"]})
    mat = compute_serp_overlap_matrix(df)
    assert mat.shape == (2, 2)
    assert (mat == 0).all()


def test_compute_serp_overlap_matrix_handles_all_empty_strings():
    df = pd.DataFrame({"keyword": ["a", "b"], "serp_urls": ["", ""]})
    mat = compute_serp_overlap_matrix(df)
    assert mat.shape == (2, 2)
    assert (mat == 0).all()


def test_compute_serp_overlap_matrix_handles_nan_rows():
    df = pd.DataFrame({"keyword": ["a", "b"], "serp_urls": [None, "https://x.com/1"]})
    mat = compute_serp_overlap_matrix(df)
    assert mat.shape == (2, 2)
    # NaN row has empty url-set, so all overlaps with it are 0.
    assert mat[0, 1] == 0.0
    # b vs b is full self-overlap.
    assert mat[1, 1] == 1.0


def test_normalise_rows_unit_norm():
    raw = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    out = normalise_rows(raw)
    norms = np.linalg.norm(out, axis=1)
    # Zero-row stays at norm 0; the others become unit-norm.
    np.testing.assert_allclose([norms[0], norms[2]], [1.0, 1.0], atol=1e-6)
    assert norms[1] == 0.0


def test_compose_hybrid_feature_vectors_concatenates_weighted_blocks():
    sem = np.array([[1.0, 0.0]])
    tfi = np.array([[0.0, 1.0, 0.0]])
    out = compose_hybrid_feature_vectors(sem, tfi, semantic_weight=0.6, tfidf_weight=0.4)
    assert out.shape == (1, 5)  # 2 + 3 cols
    # Each block is L2-normalised then scaled by its weight.
    np.testing.assert_allclose(np.linalg.norm(out[0, :2]), 0.6, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out[0, 2:]), 0.4, atol=1e-6)


def test_compose_hybrid_feature_vectors_raises_when_no_inputs():
    import pytest

    with pytest.raises(ValueError, match="hybrid"):
        compose_hybrid_feature_vectors(None, None, semantic_weight=0.5, tfidf_weight=0.5)
