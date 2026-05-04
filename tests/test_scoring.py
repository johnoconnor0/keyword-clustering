"""Tests for scoring module."""

import numpy as np
import pandas as pd
import pytest

from keyword_clustering.scoring import (
    compute_opportunity_score,
    detect_cannibalization,
    detect_content_gaps,
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
    df = pd.DataFrame({
        "keyword": ["seo", "web design", "marketing"],
        "search_volume": [1000, 500, 2000],
        "keyword_difficulty": [30, 60, 80],
        "rank": [5, 25, 50],
    })
    result = compute_opportunity_score(df)
    assert "opportunity_score" in result.columns
    assert result["opportunity_score"].between(-1, 1).all()


def test_opportunity_score_formula_coefficients():
    # Two rows with min/max values so normalization is deterministic:
    # vol_norm=[0,1], diff_norm=[0,1], rank_gap=[0,1]
    df = pd.DataFrame({
        "keyword": ["low", "high"],
        "search_volume": [0, 1000],       # norm: 0.0, 1.0
        "keyword_difficulty": [0, 100],   # norm: 0.0, 1.0
        "rank": [10, 100],                # gap: 0.0, 1.0
    })
    result = compute_opportunity_score(df)
    # Row 0: 0.0*0.4 - 0.0*0.3 + 0.0*0.3 = 0.0
    assert result["opportunity_score"].iloc[0] == pytest.approx(0.0, abs=1e-4)
    # Row 1: 1.0*0.4 - 1.0*0.3 + 1.0*0.3 = 0.4
    assert result["opportunity_score"].iloc[1] == pytest.approx(0.4, abs=1e-4)


def test_opportunity_score_no_metrics():
    df = pd.DataFrame({"keyword": ["seo", "web design"]})
    result = compute_opportunity_score(df)
    assert "opportunity_score" in result.columns


def test_detect_cannibalization_found():
    df = pd.DataFrame({
        "cluster_id": [0, 0, 1, 1],
        "cluster_label": ["SEO", "SEO", "Design", "Design"],
        "recommended_page": ["SEO Services", "Blog", "Home", "Home"],
        "keyword": ["seo audit", "seo basics", "web design", "ui design"],
    })
    result = detect_cannibalization(df)
    assert len(result) == 1
    assert "SEO" in result["cluster_label"].values


def test_detect_cannibalization_none():
    df = pd.DataFrame({
        "cluster_id": [0, 0, 1],
        "cluster_label": ["SEO", "SEO", "Design"],
        "recommended_page": ["SEO Services", "SEO Services", "Home"],
        "keyword": ["seo", "seo audit", "web design"],
    })
    result = detect_cannibalization(df)
    assert result.empty


def test_detect_content_gaps():
    df = pd.DataFrame({
        "keyword": ["seo", "quantum computing"],
        "cluster_label": ["SEO", "Tech"],
        "primary_intent": ["informational", "informational"],
        "search_volume": [1000, 500],
        "keyword_difficulty": [30, 20],
        "opportunity_score": [0.5, 0.8],
        "page_similarity_score": [0.4, 0.01],
        "content_gap": [False, True],
    })
    gaps = detect_content_gaps(df)
    assert len(gaps) == 1
    assert "quantum computing" in gaps["keyword"].values
