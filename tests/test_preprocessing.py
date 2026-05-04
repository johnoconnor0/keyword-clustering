"""Tests for preprocessing module."""

import pandas as pd
import pytest

from keyword_clustering.preprocessing import (
    classify_intent,
    classify_intent_embedding,
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
    r1 = preprocess_text("marketing", mode="stem")
    r2 = preprocess_text("marketer", mode="stem")
    # Porter stemmer maps both to the same root.
    assert r1 == r2 == "market"


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
    # 'keyword clustering' has no transactional/commercial/navigational/local triggers,
    # so it must fall through to the informational fallback (not just any label).
    assert classify_intent("keyword clustering") == "informational"


def test_classify_intent_local():
    assert classify_intent("seo agency near me") == "local"


def test_preprocess_modes():
    src = "Running faster in Brisbane"
    # mode="none": untouched.
    assert preprocess_text(src, mode="none") == src
    # mode="light": lowercase + collapse whitespace, stopwords NOT removed.
    assert preprocess_text(src, mode="light") == "running faster in brisbane"
    # mode="stem": stopwords ("in") removed; remaining words Porter-stemmed.
    stemmed = preprocess_text(src, mode="stem")
    assert "in" not in stemmed.split()
    assert stemmed.split() == ["run", "faster", "brisban"]
    # mode="lemmatize": stopwords removed; words lemmatised (no -ing collapse for default POS=n).
    lemmatised = preprocess_text(src, mode="lemmatize")
    assert "in" not in lemmatised.split()
    assert "running" in lemmatised.split()  # WordNet leaves verb-form -ing alone with default POS


def test_embedding_intent_classifier_returns_confidence():
    label, conf = classify_intent_embedding("best seo agency comparison")
    assert label in {"informational", "commercial", "transactional", "local", "navigational"}
    assert 0.0 <= conf <= 1.0


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
