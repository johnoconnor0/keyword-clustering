"""Branch coverage for keyword_clustering.preprocessing — classify_intent_serp,
load_pages, load_topics, mode='serp', and mode validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from keyword_clustering.preprocessing import (
    IntentConfig,
    classify_intent_serp,
    enrich_keywords,
    load_pages,
    load_topics,
)


def test_classify_intent_serp_local_pack_dominates():
    row = pd.Series({"keyword": "plumber sydney", "local_pack": 1, "featured_snippet": 1})
    label, conf = classify_intent_serp(row)
    assert label == "local"
    assert conf == pytest.approx(0.9)


def test_classify_intent_serp_featured_snippet_signals_informational():
    row = pd.Series({"keyword": "what is seo", "local_pack": 0, "featured_snippet": 1, "people_also_ask": 0})
    label, conf = classify_intent_serp(row)
    assert label == "informational"
    assert conf == pytest.approx(0.75)


def test_classify_intent_serp_falls_back_to_rules_when_no_serp_signal():
    row = pd.Series({"keyword": "buy widgets", "local_pack": 0, "featured_snippet": 0, "people_also_ask": 0})
    label, conf = classify_intent_serp(row)
    assert label == "transactional"
    assert conf == pytest.approx(0.6)


def test_load_pages_returns_records_for_known_columns(tmp_path: Path):
    src = tmp_path / "pages.csv"
    pd.DataFrame({"url": ["/a", "/b"], "page_name": ["A", "B"], "ignore": [1, 2]}).to_csv(src, index=False)
    out = load_pages(str(src))
    assert out == [{"url": "/a", "page_name": "A"}, {"url": "/b", "page_name": "B"}]


def test_load_pages_returns_empty_for_none_path():
    assert load_pages(None) == []


def test_load_pages_missing_required_columns_raises(tmp_path: Path):
    src = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["A"]}).to_csv(src, index=False)
    with pytest.raises(ValueError, match="'url' and 'page_name'"):
        load_pages(str(src))


def test_load_topics_returns_strings_only(tmp_path: Path):
    src = tmp_path / "topics.csv"
    pd.DataFrame({"topic": [" SEO ", None, "Web Design"]}).to_csv(src, index=False)
    out = load_topics(str(src))
    assert out == ["SEO", "Web Design"]


def test_load_topics_returns_empty_for_none_path():
    assert load_topics(None) == []


def test_load_topics_missing_column_raises(tmp_path: Path):
    src = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["A"]}).to_csv(src, index=False)
    with pytest.raises(ValueError, match="'topic'"):
        load_topics(str(src))


def test_enrich_keywords_invalid_mode_raises():
    df = pd.DataFrame({"keyword": ["seo"]})
    with pytest.raises(ValueError, match="intent mode must be"):
        enrich_keywords(df, intent_config=IntentConfig(mode="bogus"))


def test_enrich_keywords_serp_mode_uses_serp_signals():
    df = pd.DataFrame(
        {
            "keyword": ["plumber sydney", "what is seo", "buy widgets"],
            "local_pack": [1, 0, 0],
            "featured_snippet": [0, 1, 0],
            "people_also_ask": [0, 0, 0],
        }
    )
    out = enrich_keywords(df, intent_config=IntentConfig(mode="serp"))
    assert list(out["intent"]) == ["local", "informational", "transactional"]
    assert list(out["intent_mode_used"]) == ["serp"] * 3


def test_enrich_keywords_rules_path_is_vectorised_and_correct():
    """Vectorised rules path: every row goes through Series.map(classify_intent)."""
    df = pd.DataFrame({"keyword": ["best seo agency", "buy widgets", "how to do seo", "seo near me"]})
    out = enrich_keywords(df.copy(), intent_config=IntentConfig(mode="rules"))
    assert list(out["intent"]) == ["commercial", "transactional", "informational", "local"]
    assert (out["intent_confidence"] == 0.7).all()
    assert (out["intent_mode_used"] == "rules").all()
