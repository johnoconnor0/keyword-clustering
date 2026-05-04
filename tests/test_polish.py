"""Regression tests for the polish-tier additions (gazetteer, bigram, per-text cache,
cluster stability, UMAP min_dist, sparse-hybrid feature vectors, defensive Streamlit
config builder) shipped in the second audit pass. The audit flagged these as
'implementation-without-test' debt.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sp

from keyword_clustering import preprocessing, vectorization
from keyword_clustering.preprocessing import classify_intent, set_local_intent_tokens
from keyword_clustering.vectorization import (
    EmbeddingConfig,
    _embedding_config_signature,
    _per_text_cache_path,
    compose_hybrid_feature_vectors,
    vectorize_keywords_st_configured,
)

# ─────────────────────────────────────────────────────────────────────────────
# M20: set_local_intent_tokens — region-specific gazetteer override
# ─────────────────────────────────────────────────────────────────────────────


def test_set_local_intent_tokens_override_then_reset():
    # Default state — Toronto isn't a built-in AU local token.
    assert classify_intent("plumber in toronto") != "local"

    set_local_intent_tokens(["toronto", "vancouver"])
    try:
        assert classify_intent("plumber in toronto") == "local"
        assert classify_intent("vancouver locksmith") == "local"
    finally:
        set_local_intent_tokens(None)

    # Reset — Toronto should once again NOT match.
    assert classify_intent("plumber in toronto") != "local"
    # AU defaults back in effect.
    assert classify_intent("plumber in sydney") == "local"


def test_set_local_intent_tokens_is_case_insensitive():
    set_local_intent_tokens(["LONDON", "Manchester"])
    try:
        assert classify_intent("london flat rental") == "local"
        assert classify_intent("manchester united tickets") == "local"
    finally:
        set_local_intent_tokens(None)


# ─────────────────────────────────────────────────────────────────────────────
# Bigram trigger pass in classify_intent
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("keyword", "expected"),
    [
        ("how to do seo", "informational"),
        ("what is canonicalization", "informational"),
        ("step by step migration guide", "informational"),
        ("sign in widget", "navigational"),
        ("log in to dashboard", "navigational"),
        ("plumber near me", "local"),
        ("free trial signup", "transactional"),
        ("widget for sale", "transactional"),
    ],
)
def test_bigram_triggers_classify_intent(keyword: str, expected: str):
    assert classify_intent(keyword) == expected


def test_bigram_triggers_take_precedence_over_unigrams():
    """A bigram trigger must beat an otherwise-stronger unigram fallthrough."""
    # 'best' is a unigram → commercial. But 'how to find the best agency' contains
    # 'how to' which is a stronger informational bigram.
    assert classify_intent("how to find the best agency") == "informational"


# ─────────────────────────────────────────────────────────────────────────────
# S24: per-text embedding cache + chunked-streaming hash
# ─────────────────────────────────────────────────────────────────────────────


def test_per_text_cache_path_separates_configs(tmp_path: Path):
    cfg_a = EmbeddingConfig(model_name="all-MiniLM-L6-v2", normalize_embeddings=True)
    cfg_b = EmbeddingConfig(model_name="all-mpnet-base-v2", normalize_embeddings=True)
    p_a = _per_text_cache_path("seo audit", cfg_a, tmp_path)
    p_b = _per_text_cache_path("seo audit", cfg_b, tmp_path)
    # Same text, different model → distinct cache paths (config signature differs).
    assert p_a != p_b
    # And the per-text root contains the config signature directory.
    assert _embedding_config_signature(cfg_a) in str(p_a)


def test_per_text_cache_path_stable_for_same_inputs(tmp_path: Path):
    cfg = EmbeddingConfig(model_name="all-MiniLM-L6-v2", normalize_embeddings=True)
    p1 = _per_text_cache_path("seo audit", cfg, tmp_path)
    p2 = _per_text_cache_path("seo audit", cfg, tmp_path)
    assert p1 == p2


def test_vectorize_keywords_st_configured_reuses_per_text_cache(tmp_path: Path):
    """Adding one keyword to a previously-cached corpus encodes only the delta."""
    cfg = EmbeddingConfig(
        model_name="stub-model",
        cache_dir=str(tmp_path),
        preprocess_mode="none",
        normalize_embeddings=False,
    )
    encode_call_log: list[list[str]] = []

    class _StubModel:
        def encode(self, texts, **kwargs):
            encode_call_log.append(list(texts))
            return np.array([[float(hash(t) % 1000) / 1000.0, 0.5] for t in texts], dtype=np.float32)

    with patch.object(vectorization, "_load_sentence_transformer", return_value=_StubModel()):
        # First run encodes all three keywords.
        first = vectorize_keywords_st_configured(["alpha", "beta", "gamma"], cfg)
        assert first.shape == (3, 2)
        assert encode_call_log[-1] == ["alpha", "beta", "gamma"]
        encode_call_log.clear()

        # Second run with one extra keyword — only "delta" should hit the model.
        second = vectorize_keywords_st_configured(["alpha", "beta", "gamma", "delta"], cfg)
        assert second.shape == (4, 2)
        assert encode_call_log[-1] == ["delta"], (
            f"expected only delta to be encoded; model received {encode_call_log[-1]}"
        )

        # Pre-existing rows must be byte-identical to first-run vectors.
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])


# ─────────────────────────────────────────────────────────────────────────────
# S24: compute_cluster_stability + input-validation guard
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_cluster_stability_returns_jaccard_scores():
    from keyword_clustering.scoring import compute_cluster_stability

    vectors = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]] * 5)
    labels = np.array([0, 0, 1, 1] * 5)
    out = compute_cluster_stability(vectors, labels, n_bootstrap=3, random_state=42)
    assert set(out.keys()) == {0, 1}
    for cid, score in out.items():
        assert 0.0 <= score <= 1.0, f"cluster {cid} stability {score} out of [0,1]"


def test_compute_cluster_stability_handles_too_few_samples():
    """n < 4 must short-circuit to NaN per cluster without bootstrapping."""
    from keyword_clustering.scoring import compute_cluster_stability

    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    out = compute_cluster_stability(vectors, labels, n_bootstrap=3)
    assert set(out.keys()) == {0, 1}
    assert all(np.isnan(v) for v in out.values())


def test_compute_cluster_stability_all_noise_labels_returns_empty(caplog: pytest.LogCaptureFixture):
    """All-noise (-1) labels must return an empty dict with a logged warning."""
    import logging

    from keyword_clustering.scoring import compute_cluster_stability

    vectors = np.array([[1.0, 0.0]] * 8)
    labels = np.full(8, -1, dtype=int)
    with caplog.at_level(logging.WARNING, logger="keyword_clustering.scoring"):
        out = compute_cluster_stability(vectors, labels, n_bootstrap=3)
    assert out == {}
    # The new guard logs an explanation.
    assert any("noise" in rec.message.lower() or "no clusters" in rec.message.lower() for rec in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# S24: UMAP min_dist parameter actually flows through
# ─────────────────────────────────────────────────────────────────────────────


def test_umap_min_dist_is_passed_to_umap():
    """The umap_min_dist kwarg must be forwarded to the UMAP constructor."""
    pytest.importorskip("umap")
    from keyword_clustering.clustering import reduce_dimensions

    captured: dict = {}

    class _StubUMAP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit_transform(self, x):
            n = x.shape[0]
            return np.zeros((n, captured.get("n_components", 3)))

    with patch("umap.UMAP", _StubUMAP):
        vec = np.random.RandomState(0).rand(10, 5)
        reduce_dimensions(vec, n_components=3, method="umap", umap_min_dist=0.0)

    assert captured["min_dist"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# H10: sparse compose_hybrid_feature_vectors path
# ─────────────────────────────────────────────────────────────────────────────


def test_compose_hybrid_feature_vectors_preserves_sparsity_when_tfidf_is_sparse():
    """When the TF-IDF block is a scipy.sparse matrix, the returned hybrid must be sparse too —
    no silent dense materialisation that would OOM on large vocabularies."""
    semantic = np.eye(3, dtype=np.float32)  # 3 × 3 dense
    tfidf = sp.csr_matrix(np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]))

    out = compose_hybrid_feature_vectors(semantic, tfidf, semantic_weight=0.6, tfidf_weight=0.4)

    assert sp.issparse(out), "hybrid output must remain sparse when TF-IDF input is sparse"
    # 3 + 4 columns
    assert out.shape == (3, 7)


def test_compose_hybrid_feature_vectors_returns_dense_when_both_inputs_dense():
    semantic = np.eye(3, dtype=np.float32)
    tfidf = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    out = compose_hybrid_feature_vectors(semantic, tfidf, semantic_weight=0.5, tfidf_weight=0.5)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Defensive Streamlit config builder — survives version skew between the deployed
# `app/streamlit_app.py` and the installed `keyword_clustering` package.
# ─────────────────────────────────────────────────────────────────────────────


def test_filter_supported_kwargs_drops_unknown_fields():
    """Simulate an older EmbeddingConfig (without `text_mode`) and verify the
    helper drops the unknown kwarg instead of raising TypeError."""
    from dataclasses import dataclass

    from app.helpers import _filter_supported_kwargs

    @dataclass
    class _OldEmbedding:
        model_name: str = "tfidf"
        preprocess_mode: str = "none"
        # no text_mode field

    kwargs = {
        "model_name": "tfidf",
        "preprocess_mode": "stem",
        "text_mode": "expanded",  # not in _OldEmbedding
        "chunk_size": 0,  # not in _OldEmbedding either
    }
    filtered = _filter_supported_kwargs(_OldEmbedding, kwargs)
    assert filtered == {"model_name": "tfidf", "preprocess_mode": "stem"}
    # Smoke: the filtered kwargs are constructible without TypeError.
    instance = _OldEmbedding(**filtered)
    assert instance.preprocess_mode == "stem"


def test_build_pipeline_config_smoke():
    """The Streamlit builder must produce a usable PipelineConfig with all current fields."""
    from app.helpers import build_pipeline_config

    cfg = build_pipeline_config(
        method="kmeans",
        n_clusters=4,
        embedding="tfidf",
        reduction="pca",
        similarity="tfidf",
        semantic_weight=0.5,
        tfidf_weight=0.5,
        serp_weight=0.0,
        preprocess="stem",
        gap_threshold=0.25,
        gap_threshold_mode="fixed",
        intent_mode="rules",
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=2,
        graph_k=10,
        graph_min_similarity=0.55,
        community_algorithm="louvain",
        graph_use_ann=False,
        graph_ann_ef=100,
        embedding_cache_dir=".cache/embeddings",
        embedding_text_mode="keyword",
        embedding_chunk_size=0,
        tfidf_svd_components=0,
        umap_min_dist=0.1,
        save_run_history=False,
    )
    assert cfg.embedding.model_name == "tfidf"
    assert cfg.clustering.method == "kmeans"
    assert cfg.preprocess_mode == "stem"
    assert cfg.reduction == "pca"


# ─────────────────────────────────────────────────────────────────────────────
# Restore-on-failure guard: ensure other tests don't see a leaked override.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_local_intent_tokens():
    yield
    preprocessing.set_local_intent_tokens(None)
