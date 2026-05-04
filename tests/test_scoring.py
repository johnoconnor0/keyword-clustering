"""Tests for scoring module."""

import numpy as np
import pandas as pd
import pytest

from keyword_clustering.scoring import (
    GapThresholdConfig,
    OpportunityConfig,
    SimilarityConfig,
    build_cluster_quality_report,
    compose_hybrid_similarity_matrix,
    compute_opportunity_score,
    detect_cannibalization,
    detect_content_gaps,
    map_keywords_to_pages,
    require_similarity_inputs,
    score_keywords_against_corpus,
)
from keyword_clustering.vectorization import build_tfidf_vectorizer, vectorize_keywords_tfidf


def _make_vectors(texts: list[str], corpus: list[str] | None = None):
    corpus = corpus or texts
    vec, _ = build_tfidf_vectorizer(corpus)
    return vectorize_keywords_tfidf(texts, vec), vec


def test_score_keywords_returns_correct_shape():
    keywords = ["SEO", "web design", "content marketing"]
    pages = ["SEO Services", "Website Development", "Blog"]
    kw_vecs, vec = _make_vectors(keywords, keywords + pages)
    page_vecs = vectorize_keywords_tfidf(pages, vec)
    names, scores = score_keywords_against_corpus(kw_vecs, page_vecs, pages)
    assert len(names) == len(keywords)
    assert len(scores) == len(keywords)
    assert all(0 <= s <= 1 for s in scores)


def test_score_keywords_self_similarity():
    texts = ["SEO services optimisation"]
    vecs, vec = _make_vectors(texts)
    names, scores = score_keywords_against_corpus(vecs, vecs, texts)
    assert scores[0] == pytest.approx(1.0, abs=1e-6)


def test_opportunity_score_range():
    df = pd.DataFrame(
        {
            "keyword": ["seo", "web design", "marketing"],
            "search_volume": [1000, 500, 2000],
            "keyword_difficulty": [30, 60, 80],
            "rank": [5, 25, 50],
        }
    )
    result = compute_opportunity_score(df)
    assert "opportunity_score" in result.columns
    assert result["opportunity_score"].between(-1, 1).all()


def test_opportunity_score_formula_coefficients():
    """Verify the actual coefficient math, not just column existence.

    Construct deterministic min/max inputs so each normalised component is exactly
    0.0 (low row) or 1.0 (high row), then verify the weighted sum matches the
    'balanced' profile coefficients.
    """
    # Two rows with min/max values so normalization is deterministic:
    # vol_norm=[0,1], diff_norm=[0,1], rank_gap=[0,1]
    df = pd.DataFrame(
        {
            "keyword": ["low", "high"],
            "search_volume": [0, 1000],  # norm: 0.0, 1.0
            "keyword_difficulty": [0, 100],  # norm: 0.0, 1.0
            "rank": [10, 100],  # gap: 0.0, 1.0
        }
    )
    result = compute_opportunity_score(df)

    # The high row's components should match the 'balanced' profile coefficients
    # exactly, since each normalised input is 1.0.
    high = result.iloc[1]
    assert pytest.approx(float(high["opportunity_components_search_volume_score"]), abs=1e-6) == 0.25
    # Difficulty has weight -0.05 — high keyword_difficulty hurts the score.
    assert pytest.approx(float(high["opportunity_components_keyword_difficulty_score"]), abs=1e-6) == -0.05
    # rank_improvement_score weight = 0.20 in the balanced profile.
    assert pytest.approx(float(high["opportunity_components_rank_improvement_score"]), abs=1e-6) == 0.20

    # The low row should have all-zero numerical components (volume/difficulty/rank all min).
    low = result.iloc[0]
    assert pytest.approx(float(low["opportunity_components_search_volume_score"]), abs=1e-6) == 0.0
    assert pytest.approx(float(low["opportunity_components_keyword_difficulty_score"]), abs=1e-6) == 0.0
    assert pytest.approx(float(low["opportunity_components_rank_improvement_score"]), abs=1e-6) == 0.0


def test_opportunity_score_no_metrics_collapses_to_constant_with_warning(caplog: pytest.LogCaptureFixture):
    """When no signal columns are present, every row gets the same opportunity_score
    and the function must log a WARNING + emit 'insufficient signal' as the reason."""
    import logging

    df = pd.DataFrame({"keyword": ["seo", "web design"]})
    with caplog.at_level(logging.WARNING, logger="keyword_clustering.scoring"):
        result = compute_opportunity_score(df)
    assert "opportunity_score" in result.columns
    assert any("opportunity_score will collapse" in rec.message for rec in caplog.records)
    # Both rows must get the explicit constant-fallback reason, not a verbose breakdown.
    assert (result["opportunity_reason"] == "insufficient signal: opportunity_score is a constant fallback").all()
    # All rows collapse to the same value.
    assert result["opportunity_score"].nunique() == 1


def test_detect_cannibalization_found():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1],
            "cluster_label": ["SEO", "SEO", "Design", "Design"],
            "recommended_page": ["SEO Services", "Blog", "Home", "Home"],
            "keyword": ["seo audit", "seo basics", "web design", "ui design"],
        }
    )
    result = detect_cannibalization(df)
    assert not result.empty
    assert "mapping_conflict" in result["cannibalization_type"].values


def test_detect_cannibalization_none():
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1],
            "cluster_label": ["SEO", "SEO", "Design"],
            "recommended_page": ["SEO Services", "SEO Services", "Home"],
            "keyword": ["seo", "seo audit", "web design"],
        }
    )
    result = detect_cannibalization(df)
    assert result.empty


def test_opportunity_profile_switch():
    df = pd.DataFrame(
        {
            "keyword": ["a", "b"],
            "search_volume": [100, 1000],
            "keyword_difficulty": [10, 80],
            "rank": [12, 50],
        }
    )
    base = compute_opportunity_score(df, OpportunityConfig(profile="balanced"))
    wins = compute_opportunity_score(df, OpportunityConfig(profile="quick-wins"))
    assert not base["opportunity_score"].equals(wins["opportunity_score"])


def test_quality_report_shape():
    df = pd.DataFrame(
        {
            "keyword": ["seo", "seo audit", "web design"],
            "cluster_id": [0, 0, 1],
            "cluster_label": ["SEO", "SEO", "Design"],
            "primary_intent": ["informational", "informational", "commercial"],
            "recommended_page": ["SEO Services", "SEO Services", "Design Services"],
        }
    )
    rep = build_cluster_quality_report(df)
    assert "cluster_size" in rep.columns
    assert len(rep) == 2


def test_quality_report_extended_metrics():
    df = pd.DataFrame(
        {
            "keyword": ["seo", "seo audit", "web design", "ui design"],
            "cluster_id": [0, 0, 1, 1],
            "cluster_label": ["SEO", "SEO", "Design", "Design"],
            "primary_intent": ["informational", "informational", "commercial", "commercial"],
            "recommended_page": ["SEO Services", "SEO Services", "Design Services", "Design Services"],
            "serp_urls": ["a|b|c", "a|b|d", "x|y", "x|z"],
            "match_confidence": ["weak_match", "strong_match", "acceptable_match", "weak_match"],
        }
    )
    vectors = np.array(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.0, 0.1],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )
    rep = build_cluster_quality_report(df, keyword_vectors=vectors)
    for col in (
        "avg_intra_cluster_similarity",
        "avg_nearest_cluster_similarity",
        "serp_overlap_mean",
        "weakly_matched_percentage",
    ):
        assert col in rep.columns


def test_detect_content_gaps():
    df = pd.DataFrame(
        {
            "keyword": ["seo", "quantum computing"],
            "cluster_label": ["SEO", "Tech"],
            "primary_intent": ["informational", "informational"],
            "search_volume": [1000, 500],
            "keyword_difficulty": [30, 20],
            "opportunity_score": [0.5, 0.8],
            "page_similarity_score": [0.4, 0.01],
            "content_gap": [False, True],
        }
    )
    gaps = detect_content_gaps(df)
    assert len(gaps) == 1
    assert "quantum computing" in gaps["keyword"].values


def test_hybrid_similarity_weight_validation():
    mat = np.eye(3)
    with pytest.raises(ValueError):
        compose_hybrid_similarity_matrix(
            mat, mat, None, SimilarityConfig(mode="hybrid", semantic_weight=0, tfidf_weight=0)
        )


def test_similarity_input_requirements():
    with pytest.raises(ValueError):
        require_similarity_inputs("semantic", semantic_vectors=None, tfidf_vectors=np.ones((2, 2)))
    with pytest.raises(ValueError):
        require_similarity_inputs("tfidf", semantic_vectors=np.ones((2, 2)), tfidf_vectors=None)


def test_gap_threshold_modes():
    df = pd.DataFrame({"keyword": ["a", "b", "c"]})
    keywords = ["seo audit", "web design", "content plan"]
    pages = ["seo page", "web page", "content page"]
    vec, _ = build_tfidf_vectorizer(keywords + pages)
    kw = vectorize_keywords_tfidf(keywords, vec)
    pg = vectorize_keywords_tfidf(pages, vec)
    out = map_keywords_to_pages(
        df,
        kw,
        pg,
        pages,
        ["/a", "/b", "/c"],
        gap_config=GapThresholdConfig(mode="percentile", percentile=50),
    )
    assert "gap_threshold_used" in out.columns
    assert "match_confidence" in out.columns
