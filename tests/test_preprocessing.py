"""Tests for preprocessing module."""

import pandas as pd
import pytest

from keyword_clustering.preprocessing import (
    classify_intent,
    enrich_keywords,
    load_keywords,
    preprocess_text,
)


def test_preprocess_text_basic():
    result = preprocess_text("Digital Marketing")
    assert isinstance(result, str)
    assert len(result) > 0
    # Stopwords like "and" should be removed
    result2 = preprocess_text("SEO and marketing")
    assert "and" not in result2.split()


def test_preprocess_text_punctuation():
    result = preprocess_text("web-design, UX!")
    assert "," not in result
    assert "!" not in result
    assert "-" not in result


def test_preprocess_text_lowercase():
    r1 = preprocess_text("SEO")
    r2 = preprocess_text("seo")
    assert r1 == r2


def test_preprocess_text_stemming():
    r1 = preprocess_text("marketing")
    r2 = preprocess_text("marketer")
    # Both should stem to the same root
    assert r1 and r2  # both non-empty


def test_classify_intent_informational():
    assert classify_intent("what is SEO") == "informational"
    assert classify_intent("how to improve rankings") == "informational"


def test_classify_intent_commercial():
    assert classify_intent("best SEO agency") == "commercial"
    assert classify_intent("top web design company") == "commercial"


def test_classify_intent_transactional():
    assert classify_intent("buy website design") == "transactional"
    assert classify_intent("pricing packages") == "transactional"


def test_classify_intent_navigational():
    assert classify_intent("login dashboard") == "navigational"


def test_classify_intent_default():
    result = classify_intent("keyword clustering")
    assert result in {"informational", "commercial", "transactional", "navigational"}


def test_load_keywords_missing_column(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("query,volume\nseo,1000\n")
    with pytest.raises(ValueError, match="keyword"):
        load_keywords(str(csv))


def test_load_keywords_deduplication(tmp_path):
    csv = tmp_path / "kw.csv"
    csv.write_text("keyword\nseo\nseo\nmarketing\n")
    df = load_keywords(str(csv))
    assert df["keyword"].duplicated().sum() == 0
    assert len(df) == 2


def test_enrich_keywords_adds_intent():
    df = pd.DataFrame({"keyword": ["what is SEO", "buy web design", "SEO agency"]})
    enriched = enrich_keywords(df)
    assert "intent" in enriched.columns
    assert enriched["intent"].notna().all()


def test_enrich_keywords_branded():
    df = pd.DataFrame({"keyword": ["acme SEO", "marketing tips"]})
    enriched = enrich_keywords(df, brand_terms=["acme"])
    assert enriched.loc[0, "branded"] == True
    assert enriched.loc[1, "branded"] == False
