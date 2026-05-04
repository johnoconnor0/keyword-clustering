"""Text preprocessing and CSV ingestion."""

from __future__ import annotations

import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()
_stop_words: set[str] | None = None


def _get_stop_words() -> set[str]:
    global _stop_words
    if _stop_words is None:
        try:
            _stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _stop_words = set(stopwords.words("english"))
    return _stop_words


def preprocess_text(text: str) -> str:
    """Lowercase, strip non-word chars, stem each token."""
    text = text.lower()
    text = re.sub(r"\W+", " ", text).strip()
    stop = _get_stop_words()
    tokens = [_stemmer.stem(w) for w in text.split() if w not in stop]
    return " ".join(tokens)


def load_keywords(path: str) -> pd.DataFrame:
    """Load keywords CSV. Required column: keyword. All others optional."""
    df = pd.read_csv(path)
    if "keyword" not in df.columns:
        raise ValueError(f"Keywords CSV must have a 'keyword' column. Found: {list(df.columns)}")
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["keyword"]).reset_index(drop=True)
    numeric_cols = [
        "search_volume", "keyword_difficulty", "cpc", "rank", "clicks", "impressions", "ctr",
        # SERP feature presence flags (1/0 or True/False from export tools)
        "featured_snippet", "local_pack", "people_also_ask", "image_pack", "video_result",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pages(path: str | None) -> list[dict]:
    """Load pages CSV with columns url, page_name. Returns list of dicts."""
    if path is None:
        return []
    df = pd.read_csv(path)
    for col in ("url", "page_name"):
        if col not in df.columns:
            raise ValueError(f"Pages CSV must have columns 'url' and 'page_name'. Found: {list(df.columns)}")
    return df[["url", "page_name"]].dropna().to_dict(orient="records")


def load_topics(path: str | None) -> list[str]:
    """Load topics CSV with column topic. Returns list of topic strings."""
    if path is None:
        return []
    df = pd.read_csv(path)
    if "topic" not in df.columns:
        raise ValueError(f"Topics CSV must have a 'topic' column. Found: {list(df.columns)}")
    return df["topic"].dropna().astype(str).str.strip().tolist()


def classify_intent(keyword: str) -> str:
    """Rule-based search intent classifier."""
    kw = keyword.lower()
    transactional = {"buy", "purchase", "order", "price", "pricing", "cost", "cheap", "deal", "discount", "hire", "get"}
    commercial = {"best", "top", "review", "compare", "vs", "service", "agency", "company", "tool", "software", "platform"}
    navigational = {"login", "sign in", "account", "dashboard", "portal", "app"}
    informational = {"what", "how", "why", "when", "where", "guide", "tutorial", "learn", "explained", "basics", "tips"}

    words = set(kw.split())
    if words & navigational:
        return "navigational"
    if words & transactional:
        return "transactional"
    if words & commercial:
        return "commercial"
    if words & informational:
        return "informational"
    return "informational"


def enrich_keywords(df: pd.DataFrame, brand_terms: list[str] | None = None) -> pd.DataFrame:
    """Add intent and branded columns if not already present."""
    if "intent" not in df.columns or df["intent"].isna().any():
        df["intent"] = df["intent"].combine_first(df["keyword"].apply(classify_intent)) \
            if "intent" in df.columns else df["keyword"].apply(classify_intent)

    brand_terms = [t.lower() for t in (brand_terms or [])]
    df["branded"] = df["keyword"].str.lower().apply(
        lambda kw: any(b in kw for b in brand_terms) if brand_terms else False
    )
    return df
