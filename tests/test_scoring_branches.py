"""Branch coverage for keyword_clustering.scoring boundaries the audit flagged."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from keyword_clustering.scoring import (
    GapThresholdConfig,
    OpportunityConfig,
    compute_opportunity_score,
    map_keywords_to_pages,
    score_confidence_band,
)


def test_score_confidence_band_boundaries():
    # Boundary semantics are <0.20 / <0.40 / <0.65 / else.
    assert score_confidence_band(0.0) == "poor_match"
    assert score_confidence_band(0.1999) == "poor_match"
    assert score_confidence_band(0.20) == "weak_match"
    assert score_confidence_band(0.3999) == "weak_match"
    assert score_confidence_band(0.40) == "acceptable_match"
    assert score_confidence_band(0.6499) == "acceptable_match"
    assert score_confidence_band(0.65) == "strong_match"
    assert score_confidence_band(1.0) == "strong_match"


def test_compute_opportunity_score_invalid_profile_raises():
    df = pd.DataFrame({"keyword": ["x"], "search_volume": [100]})
    with pytest.raises(ValueError, match="opportunity profile"):
        compute_opportunity_score(df, cfg=OpportunityConfig(profile="bogus"))


def test_compute_opportunity_score_warns_when_no_signal_columns_present(caplog: pytest.LogCaptureFixture):
    df = pd.DataFrame({"keyword": ["x", "y"]})
    with caplog.at_level(logging.WARNING, logger="keyword_clustering.scoring"):
        out = compute_opportunity_score(df)
    assert any("opportunity_score will collapse" in rec.message for rec in caplog.records)
    # All rows get the explicit "insufficient signal" reason instead of the verbose breakdown.
    assert (out["opportunity_reason"] == "insufficient signal: opportunity_score is a constant fallback").all()


def test_map_keywords_to_pages_adaptive_mode_respects_floor():
    """Adaptive mode = max(adaptive_floor, percentile). When the percentile is below the floor,
    the floor wins; when above, the percentile wins."""
    df = pd.DataFrame({"keyword": ["a", "b", "c", "d"]})
    # All similarities are 0.10 so the 25th percentile is 0.10 — well below adaptive_floor of 0.30.
    keyword_vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    page_vectors = np.array([[0.10, 0.99]])
    cfg = GapThresholdConfig(mode="adaptive", percentile=25.0, adaptive_floor=0.30)
    mapped = map_keywords_to_pages(
        df,
        keyword_vectors,
        page_vectors,
        page_names=["Some page"],
        page_urls=["/page"],
        gap_config=cfg,
    )
    assert mapped["gap_threshold_used"].iloc[0] == 0.30


def test_map_keywords_to_pages_invalid_mode_raises():
    df = pd.DataFrame({"keyword": ["a"]})
    keyword_vectors = np.array([[1.0, 0.0]])
    page_vectors = np.array([[1.0, 0.0]])
    bad = GapThresholdConfig(mode="bogus")
    with pytest.raises(ValueError, match="gap_threshold_mode"):
        map_keywords_to_pages(
            df,
            keyword_vectors,
            page_vectors,
            page_names=["p"],
            page_urls=["/p"],
            gap_config=bad,
        )


def test_map_keywords_to_pages_default_config_is_fixed_25():
    """When no gap_config is passed, the default is fixed-mode at value=0.25."""
    df = pd.DataFrame({"keyword": ["a"]})
    keyword_vectors = np.array([[1.0, 0.0]])
    page_vectors = np.array([[1.0, 0.0]])
    mapped = map_keywords_to_pages(
        df,
        keyword_vectors,
        page_vectors,
        page_names=["p"],
        page_urls=["/p"],
    )
    assert mapped["gap_threshold_used"].iloc[0] == 0.25
