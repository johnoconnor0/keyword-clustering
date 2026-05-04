"""Tests for the semantic intent classifier added in Wave 8.

The audit (todo #10) flagged that classify_intent_embedding did no embeddings — only
token overlap. This file pins the new behaviour:
    * classify_intents_semantic returns sane labels with embeddings when available.
    * It falls back to classify_intent_overlap if sentence-transformers isn't importable.
    * enrich_keywords with mode='embedding' + an embedding_model uses the semantic batch.
    * The default mode='embedding' (no model name) keeps the old overlap behaviour.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from keyword_clustering.preprocessing import (
    IntentConfig,
    classify_intent_overlap,
    classify_intents_semantic,
    enrich_keywords,
)


def test_classify_intent_overlap_is_old_behaviour():
    label, conf = classify_intent_overlap("best seo agency")
    assert label == "commercial"
    assert 0.0 <= conf <= 1.0


def test_classify_intents_semantic_with_no_model_returns_per_keyword_overlap_results():
    """When the EmbeddingConfig import works but the model can't load, the function
    must fall back to classify_intent_overlap per keyword without crashing."""

    def boom(*args, **kwargs):
        raise OSError("model not available")

    with patch("keyword_clustering.vectorization.vectorize_keywords_st_configured", side_effect=boom):
        out = classify_intents_semantic(["best seo agency", "buy widgets"], model_name="missing-model")

    assert len(out) == 2
    assert out[0][0] == "commercial"
    assert out[1][0] == "transactional"


def test_classify_intents_semantic_with_stub_vectors_separates_classes():
    """Wire a deterministic stub vectoriser so the dot-product math is exercised
    without downloading a real model. The keyword vector is constructed to match
    the 'commercial' prototype; classifier should pick that label."""
    n_protos = 24  # sum of len() across _INTENT_PROTOTYPE_PHRASES

    def stub(texts, cfg):
        # Build orthogonal vectors per intent group, then a single keyword aligned with 'commercial'.
        out = np.zeros((len(texts), 5), dtype=np.float32)
        # Order in _INTENT_PROTOTYPE_PHRASES dict: informational, commercial, transactional, local, navigational
        proto_axes = {
            range(0, 6): 0,  # informational
            range(6, 12): 1,  # commercial
            range(12, 18): 2,  # transactional
            range(18, 22): 3,  # local
            range(22, 26): 4,  # navigational
        }
        for r, axis in proto_axes.items():
            for i in r:
                if i < n_protos:
                    out[i, axis] = 1.0
        # Following the prototype rows is a single keyword aligned with commercial.
        for i in range(n_protos, len(texts)):
            out[i, 1] = 1.0
        return out

    with patch("keyword_clustering.vectorization.vectorize_keywords_st_configured", side_effect=stub):
        out = classify_intents_semantic(["best seo agency"], model_name="stub-model")

    assert out[0][0] == "commercial"
    assert out[0][1] > 0.9


def test_enrich_keywords_embedding_mode_uses_semantic_when_model_set():
    """enrich_keywords with mode='embedding' AND embedding_model='...' must take the semantic batch path."""
    df = pd.DataFrame({"keyword": ["best seo agency", "how to do seo"]})

    fake_results = [("commercial", 0.92), ("informational", 0.88)]
    with patch(
        "keyword_clustering.preprocessing.classify_intents_semantic",
        return_value=fake_results,
    ) as mock_sem:
        out = enrich_keywords(df.copy(), intent_config=IntentConfig(mode="embedding", embedding_model="stub"))

    mock_sem.assert_called_once()
    assert list(out["intent"]) == ["commercial", "informational"]
    assert list(out["intent_mode_used"]) == ["embedding_semantic", "embedding_semantic"]


def test_enrich_keywords_embedding_mode_without_model_falls_back_to_overlap():
    """When embedding_model is empty (default), embedding mode must keep the overlap behaviour."""
    df = pd.DataFrame({"keyword": ["best seo agency", "buy widgets"]})

    with patch("keyword_clustering.preprocessing.classify_intents_semantic") as mock_sem:
        out = enrich_keywords(df.copy(), intent_config=IntentConfig(mode="embedding"))

    # Semantic path must NOT have been called when no model is configured.
    mock_sem.assert_not_called()
    # The mode-used column should reflect the overlap path.
    used = set(out["intent_mode_used"])
    assert used.issubset({"embedding_overlap", "rules_fallback"})


def test_enrich_keywords_embedding_mode_low_confidence_falls_back_to_rules():
    """Even when semantic mode runs, scores below semantic_threshold should fall back to rules."""
    df = pd.DataFrame({"keyword": ["something ambiguous"]})
    with patch(
        "keyword_clustering.preprocessing.classify_intents_semantic",
        return_value=[("local", 0.05)],  # below default threshold of 0.2
    ):
        out = enrich_keywords(
            df.copy(),
            intent_config=IntentConfig(mode="embedding", embedding_model="stub", semantic_threshold=0.2),
        )

    assert out["intent_mode_used"].iloc[0] == "rules_fallback"


def test_classify_intents_semantic_handles_empty_input():
    out = classify_intents_semantic([], model_name="stub")
    assert out == []
