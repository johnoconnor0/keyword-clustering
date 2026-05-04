"""Text preprocessing, intent classification, and CSV ingestion."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stop_words: set[str] | None = None
_stop_words_lock = threading.Lock()
_VALID_PREPROCESS_MODES = {"none", "light", "stem", "lemmatize"}
_INTENT_LABELS = ("informational", "commercial", "transactional", "local", "navigational")


def _get_stop_words() -> set[str]:
    global _stop_words
    # Double-checked locking — safe under concurrent Streamlit sessions.
    if _stop_words is None:
        with _stop_words_lock:
            if _stop_words is None:
                try:
                    _stop_words = set(stopwords.words("english"))
                except LookupError:
                    nltk.download("stopwords", quiet=True)
                    _stop_words = set(stopwords.words("english"))
    return _stop_words


def _normalise_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\W+", " ", text).strip()
    return text


def preprocess_text(text: str, mode: str = "stem") -> str:
    """Normalize text according to a preprocessing mode."""
    if mode not in _VALID_PREPROCESS_MODES:
        raise ValueError(f"Unknown preprocess mode: {mode!r}. Choose none, light, stem, or lemmatize.")
    if mode == "none":
        return str(text)

    text = _normalise_text(text)
    if mode == "light":
        return " ".join(text.split())

    stop = _get_stop_words()
    if mode == "stem":
        tokens = [_stemmer.stem(w) for w in text.split() if w not in stop]
    else:
        try:
            tokens = [_lemmatizer.lemmatize(w) for w in text.split() if w not in stop]
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            tokens = [_lemmatizer.lemmatize(w) for w in text.split() if w not in stop]
    return " ".join(tokens)


def load_keywords(path: str) -> pd.DataFrame:
    """Load keywords CSV. Required column: keyword. All others optional."""
    df = pd.read_csv(path)
    if "keyword" not in df.columns:
        raise ValueError(f"Keywords CSV must have a 'keyword' column. Found: {list(df.columns)}")
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["keyword"]).reset_index(drop=True)
    numeric_cols = [
        "search_volume",
        "keyword_difficulty",
        "cpc",
        "rank",
        "clicks",
        "impressions",
        "ctr",
        # SERP feature presence flags (1/0 or True/False from export tools)
        "featured_snippet",
        "local_pack",
        "people_also_ask",
        "image_pack",
        "video_result",
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


_DEFAULT_LOCAL_TOKENS: frozenset[str] = frozenset(
    {"near", "nearby", "local", "brisbane", "sydney", "melbourne", "perth", "adelaide", "gold coast"}
)
_local_tokens_override: set[str] | None = None


def set_local_intent_tokens(tokens: list[str] | set[str] | None) -> None:
    """Replace the local-intent token set used by `classify_intent`.

    Pass ``None`` to revert to the built-in (Australia-centric) defaults. The intended
    use is at app/CLI startup so end-users in other regions (UK, US, IN, ...) can
    inject their own gazetteer without code changes.
    """
    global _local_tokens_override
    if tokens is None:
        _local_tokens_override = None
    else:
        _local_tokens_override = {t.lower() for t in tokens}


def _local_tokens() -> frozenset[str]:
    if _local_tokens_override is None:
        return _DEFAULT_LOCAL_TOKENS
    return frozenset(_local_tokens_override)


_BIGRAM_TRIGGERS: dict[str, str] = {
    # Bigram phrases that should trigger an intent class even when the constituent
    # unigrams don't appear in the per-class token sets.
    "sign in": "navigational",
    "log in": "navigational",
    "log out": "navigational",
    "near me": "local",
    "how to": "informational",
    "what is": "informational",
    "step by": "informational",  # "step by step"
    "side effects": "informational",
    "vs.": "commercial",
    "free trial": "transactional",
    "for sale": "transactional",
    "money back": "transactional",
}


def _bigrams_in_text(text: str) -> set[str]:
    tokens = text.split()
    return {f"{a} {b}" for a, b in zip(tokens, tokens[1:])}


def classify_intent(keyword: str) -> str:
    """Rule-based search intent classifier."""
    kw = keyword.lower()
    transactional = {"buy", "purchase", "order", "price", "pricing", "cost", "cheap", "deal", "discount", "hire", "get"}
    commercial = {
        "best",
        "top",
        "review",
        "compare",
        "vs",
        "service",
        "agency",
        "company",
        "tool",
        "software",
        "platform",
    }
    navigational = {"login", "sign in", "account", "dashboard", "portal", "app"}
    informational = {"what", "how", "why", "when", "where", "guide", "tutorial", "learn", "explained", "basics", "tips"}
    local = _local_tokens()

    words = set(kw.split())
    bigrams = _bigrams_in_text(kw)

    # Bigram triggers run first — phrases like "sign in" / "near me" / "how to"
    # are stronger signals than the constituent unigrams.
    for bigram in bigrams:
        if bigram in _BIGRAM_TRIGGERS:
            return _BIGRAM_TRIGGERS[bigram]
    if "near me" in kw or words & local:
        return "local"
    if words & navigational:
        return "navigational"
    if words & transactional:
        return "transactional"
    if words & commercial:
        return "commercial"
    if words & informational:
        return "informational"
    return "informational"


@dataclass(frozen=True)
class IntentConfig:
    mode: str = "rules"
    semantic_threshold: float = 0.2
    # Sentence-transformer model to use when mode='embedding'. Empty string → token-overlap fallback.
    embedding_model: str = ""


# Prototype keywords for each intent class. Used by both the overlap-only and the
# transformer-backed classifiers — the latter encodes these phrases and compares
# each keyword's embedding against the per-class centroid.
_INTENT_PROTOTYPE_PHRASES: dict[str, tuple[str, ...]] = {
    "informational": ("how to do this", "what is this", "tutorial", "guide", "checklist", "explained"),
    "commercial": ("best service", "top tools", "comparison", "reviews", "agency comparison", "vs alternative"),
    "transactional": ("buy now", "pricing", "request a quote", "book online", "hire", "cost"),
    "local": ("near me", "local service", "in this city", "nearby"),
    "navigational": ("brand login", "user portal", "dashboard", "account page"),
}

# Token-overlap prototypes (legacy; cheap, no model required).
_INTENT_PROTOTYPE_TOKENS: dict[str, set[str]] = {
    "informational": {"how", "guide", "what", "tutorial", "checklist"},
    "commercial": {"best", "top", "comparison", "reviews", "agency"},
    "transactional": {"buy", "pricing", "quote", "book", "hire"},
    "local": {"near", "near me", "local", "city", "suburb"},
    "navigational": {"brand", "login", "portal", "dashboard", "account"},
}


def classify_intent_overlap(keyword: str) -> tuple[str, float]:
    """Prototype similarity via token overlap. Cheap, no model dependency, weak signal."""
    kw = _normalise_text(keyword)
    tokens = set(kw.split())
    best_label = "informational"
    best_score = 0.0
    for label, proto in _INTENT_PROTOTYPE_TOKENS.items():
        overlap = len(tokens & proto)
        score = overlap / max(1, len(proto))
        if score > best_score:
            best_label, best_score = label, score
    return best_label, float(round(best_score, 4))


# Backwards-compatible alias — earlier code and tests refer to this name; the implementation
# was always token-overlap, never embeddings (which is why the audit flagged it).
classify_intent_embedding = classify_intent_overlap


def classify_intents_semantic(keywords: list[str], model_name: str) -> list[tuple[str, float]]:
    """Sentence-transformer-backed intent classification.

    Encodes the keyword list and the per-class prototype phrases once, then assigns each
    keyword the label of its nearest prototype centroid by cosine similarity. Falls back
    to per-keyword `classify_intent_overlap` if the optional 'sentence-transformers'
    dependency is missing — so callers can opt in safely.
    """
    if not keywords:
        return []
    try:
        from .vectorization import EmbeddingConfig, vectorize_keywords_st_configured
    except ImportError:
        return [classify_intent_overlap(kw) for kw in keywords]

    cfg = EmbeddingConfig(model_name=model_name, cache_dir=None, preprocess_mode="light")
    try:
        proto_labels = list(_INTENT_PROTOTYPE_PHRASES.keys())
        proto_phrases = [phrase for phrases in _INTENT_PROTOTYPE_PHRASES.values() for phrase in phrases]
        proto_groups: list[slice] = []
        cursor = 0
        for phrases in _INTENT_PROTOTYPE_PHRASES.values():
            proto_groups.append(slice(cursor, cursor + len(phrases)))
            cursor += len(phrases)

        # Single encode call covers prototypes + every keyword to amortise model load.
        all_vectors = vectorize_keywords_st_configured(proto_phrases + list(keywords), cfg)
        proto_block = all_vectors[: len(proto_phrases)]
        kw_block = all_vectors[len(proto_phrases) :]

        import numpy as np

        proto_centroids = np.vstack([proto_block[s].mean(axis=0) for s in proto_groups])
        # Normalise both sides so a dot product equals cosine similarity.
        proto_norms = np.linalg.norm(proto_centroids, axis=1, keepdims=True)
        proto_norms[proto_norms == 0] = 1.0
        proto_centroids = proto_centroids / proto_norms
        kw_norms = np.linalg.norm(kw_block, axis=1, keepdims=True)
        kw_norms[kw_norms == 0] = 1.0
        kw_block = kw_block / kw_norms
        sims = kw_block @ proto_centroids.T
        best_idx = sims.argmax(axis=1)
        best_scores = sims.max(axis=1)
        return [(proto_labels[int(i)], float(round(float(s), 4))) for i, s in zip(best_idx, best_scores)]
    except (ImportError, OSError, RuntimeError):
        return [classify_intent_overlap(kw) for kw in keywords]


def classify_intent_serp(row: pd.Series) -> tuple[str, float]:
    """Infer intent from SERP feature flags and fallback to lexical rule-based logic."""
    if bool(row.get("local_pack", 0)):
        return "local", 0.9
    if bool(row.get("featured_snippet", 0)) or bool(row.get("people_also_ask", 0)):
        return "informational", 0.75
    kw = str(row.get("keyword", ""))
    return classify_intent(kw), 0.6


def enrich_keywords(
    df: pd.DataFrame,
    brand_terms: list[str] | None = None,
    intent_config: IntentConfig | None = None,
) -> pd.DataFrame:
    """Add intent and branded columns if not already present.

    Intent computation is vectorised: the `rules` path uses `Series.map(classify_intent)`,
    `embedding` runs as a single batch through `classify_intents_semantic`, and the
    manual-override layer is applied via `Series.where`. Earlier per-row `iterrows`
    was the slowest single op in the pipeline on large inputs.
    """
    cfg = intent_config or IntentConfig()
    mode = cfg.mode
    if mode not in {"rules", "serp", "embedding", "manual"}:
        raise ValueError("intent mode must be one of: rules, serp, embedding, manual")

    keywords = df["keyword"].astype(str)
    manual = df["intent"] if "intent" in df.columns else pd.Series([None] * len(df), index=df.index)
    manual_norm = manual.astype(str).str.strip().str.lower()
    manual_mask = manual.notna() & manual_norm.isin(_INTENT_LABELS)

    # Compute mode-specific labels for every row (we'll mask manual rows back at the end).
    if mode == "rules":
        mode_labels = keywords.map(classify_intent).astype(str)
        mode_conf = pd.Series(0.7, index=df.index)
        mode_used = pd.Series("rules", index=df.index)
    elif mode == "manual":
        # Manual mode: rows without a valid manual label fall back to rules with low conf.
        mode_labels = keywords.map(classify_intent).astype(str)
        mode_conf = pd.Series(0.5, index=df.index)
        mode_used = pd.Series("rules_fallback", index=df.index)
    elif mode == "serp":
        serp_results = df.apply(classify_intent_serp, axis=1)
        mode_labels = serp_results.map(lambda t: t[0]).astype(str)
        mode_conf = serp_results.map(lambda t: float(t[1]))
        mode_used = pd.Series("serp", index=df.index)
    else:  # mode == "embedding"
        if cfg.embedding_model:
            scored = classify_intents_semantic(keywords.tolist(), cfg.embedding_model)
            embed_used_label = "embedding_semantic"
        else:
            scored = [classify_intent_overlap(kw) for kw in keywords.tolist()]
            embed_used_label = "embedding_overlap"
        mode_labels = pd.Series([t[0] for t in scored], index=df.index, dtype=str)
        mode_conf = pd.Series([float(t[1]) for t in scored], index=df.index)
        mode_used = pd.Series(embed_used_label, index=df.index)

        # Low-confidence rows fall back to rules.
        rules_fallback_mask = mode_conf < cfg.semantic_threshold
        if rules_fallback_mask.any():
            fallback_labels = keywords[rules_fallback_mask].map(classify_intent).astype(str)
            mode_labels.loc[rules_fallback_mask] = fallback_labels
            mode_conf.loc[rules_fallback_mask] = mode_conf.loc[rules_fallback_mask].clip(lower=0.45)
            mode_used.loc[rules_fallback_mask] = "rules_fallback"

    # Manual override layer: rows with a valid pre-existing intent take precedence.
    final_labels = mode_labels.mask(manual_mask, manual_norm)
    final_conf = mode_conf.mask(manual_mask, 1.0)
    final_used = mode_used.mask(manual_mask, "manual")

    df["intent"] = final_labels
    df["intent_confidence"] = final_conf.round(4)
    df["intent_mode_used"] = final_used

    brand_terms = [t.lower() for t in (brand_terms or [])]
    df["branded"] = (
        df["keyword"]
        .astype(str)
        .str.lower()
        .apply(lambda kw: any(b in kw for b in brand_terms) if brand_terms else False)
    )
    return df
