"""Cluster label generation strategies.

Strategies:
    tfidf      – per-cluster TF-IDF over that cluster's keywords.
    c-tfidf    – BERTopic-style class TF-IDF: one fit over per-cluster pseudo-documents.
    centroid   – pick the keyword whose embedding is closest to the cluster mean.
    mmr        – Carbonell-Goldstein MMR over the cluster's keywords (requires embeddings).
                 Falls back to a diversity-by-token-coverage heuristic when no vectors are passed.
    keybert    – not yet wired; requires the optional `keybert` package. Raises NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_VALID_STRATEGIES = {"tfidf", "c-tfidf", "centroid", "mmr", "keybert"}


@dataclass(frozen=True)
class LabelingConfig:
    strategy: str = "tfidf"
    n_top_terms: int = 3
    mmr_lambda: float = 0.65


def _tfidf_terms(keywords: list[str], n_top_terms: int) -> tuple[str, list[str], float]:
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=200, stop_words="english")
    try:
        matrix = vec.fit_transform(keywords)
    except ValueError:
        return keywords[0].title(), [keywords[0]], 0.2
    scores = matrix.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    top_idx = scores.argsort()[::-1][:n_top_terms]
    top_terms = [terms[i] for i in top_idx]
    label = " / ".join(top_terms).title() if top_terms else keywords[0].title()
    conf = float(np.clip(scores[top_idx[0]] if len(top_idx) else 0.5, 0.0, 1.0))
    return label, top_terms, round(conf, 4)


def _build_ctfidf_top_terms(df: pd.DataFrame, n_top_terms: int) -> dict[object, tuple[str, list[str], float]]:
    """BERTopic-style class TF-IDF.

    Each cluster contributes a single pseudo-document — the space-joined keywords
    of that cluster. A single TfidfVectorizer is fit across those pseudo-documents,
    so the resulting term weights are class-level rather than instance-level.
    """
    grouped = df.groupby("cluster_id")["keyword"].apply(lambda kws: " ".join(map(str, kws)))
    cluster_ids = grouped.index.tolist()
    documents = grouped.tolist()
    out: dict[object, tuple[str, list[str], float]] = {}
    if not documents:
        return out
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words="english")
    try:
        matrix = vec.fit_transform(documents)
    except ValueError:
        # Empty vocabulary (e.g. all stopwords). Fall back to a per-cluster trivial label.
        for cid, doc in zip(cluster_ids, documents):
            first_word = doc.split()[0] if doc.strip() else f"Cluster {cid}"
            out[cid] = (first_word.title(), [first_word], 0.2)
        return out
    terms = vec.get_feature_names_out()
    for row_idx, cid in enumerate(cluster_ids):
        row = matrix.getrow(row_idx)
        if row.nnz == 0:
            out[cid] = (f"Cluster {cid}", [], 0.0)
            continue
        scores = row.toarray().ravel()
        top_idx = scores.argsort()[::-1][:n_top_terms]
        top_terms = [terms[j] for j in top_idx if scores[j] > 0]
        label = " / ".join(top_terms).title() if top_terms else f"Cluster {cid}"
        conf = float(np.clip(scores[top_idx[0]] if top_idx.size else 0.5, 0.0, 1.0))
        out[cid] = (label, top_terms, round(conf, 4))
    return out


def _centroid_keyword(keywords: list[str], vectors: np.ndarray | None) -> tuple[str, str, float]:
    if vectors is None or len(vectors) == 0:
        return keywords[0].title(), keywords[0], 0.5
    centroid = vectors.mean(axis=0, keepdims=True)
    sim = cosine_similarity(vectors, centroid).ravel()
    idx = int(sim.argmax())
    rep = keywords[idx]
    return rep.title(), rep, round(float(np.clip(sim[idx], 0.0, 1.0)), 4)


def _mmr_token_coverage(keywords: list[str], n_top_terms: int) -> tuple[str, list[str], float]:
    """Diversity-by-token-coverage fallback for when no embeddings are available."""
    token_sets = [set(k.lower().split()) for k in keywords]
    chosen: list[int] = []
    coverage: set[str] = set()
    for _ in range(min(n_top_terms, len(keywords))):
        best_idx = None
        best_gain = -1
        for i, tokens in enumerate(token_sets):
            if i in chosen:
                continue
            gain = len(tokens - coverage)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx is None:
            break
        chosen.append(best_idx)
        coverage |= token_sets[best_idx]
    terms = [keywords[i] for i in chosen] or [keywords[0]]
    return (
        " / ".join(terms[:n_top_terms]).title(),
        terms[:n_top_terms],
        round(min(1.0, len(coverage) / 10), 4),
    )


def _mmr_terms(
    keywords: list[str],
    cluster_vectors: np.ndarray | None,
    n_top_terms: int,
    mmr_lambda: float,
) -> tuple[str, list[str], float]:
    """Carbonell-Goldstein MMR — relevance to centroid balanced against diversity."""
    if cluster_vectors is None or len(cluster_vectors) == 0:
        return _mmr_token_coverage(keywords, n_top_terms)

    centroid = cluster_vectors.mean(axis=0, keepdims=True)
    sim_to_centroid = cosine_similarity(cluster_vectors, centroid).ravel()
    pairwise = cosine_similarity(cluster_vectors)
    selected: list[int] = []
    remaining = list(range(len(keywords)))
    target = min(n_top_terms, len(keywords))
    while remaining and len(selected) < target:
        if not selected:
            pick = int(np.argmax(sim_to_centroid[remaining]))
        else:
            best_score = -np.inf
            pick = 0
            for r_idx, c in enumerate(remaining):
                relevance = float(sim_to_centroid[c])
                redundancy = float(max(pairwise[c, s] for s in selected))
                score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
                if score > best_score:
                    best_score = score
                    pick = r_idx
        chosen = remaining.pop(pick)
        selected.append(chosen)
    chosen_keywords = [keywords[i] for i in selected]
    label = " / ".join(chosen_keywords[:n_top_terms]).title() if chosen_keywords else keywords[0].title()
    conf = round(float(sim_to_centroid[selected[0]]) if selected else 0.5, 4)
    return label, chosen_keywords[:n_top_terms], conf


def generate_cluster_labels(
    df: pd.DataFrame,
    cfg: LabelingConfig | None = None,
    keyword_vectors: object | None = None,
) -> pd.DataFrame:
    """
    Return DataFrame with one row per cluster:
    cluster_id, cluster_label, representative_keyword, top_terms, label_confidence, label_method
    """
    config = cfg or LabelingConfig()
    strategy = config.strategy
    if strategy == "keybert":
        raise NotImplementedError(
            "labeling strategy 'keybert' is not yet wired. Install the optional 'keybert' "
            "package and pick a different strategy: tfidf, c-tfidf, centroid, or mmr."
        )
    if strategy not in _VALID_STRATEGIES:
        raise ValueError("labeling strategy must be one of tfidf, c-tfidf, centroid, mmr, keybert")

    dense = None
    if keyword_vectors is not None:
        dense = np.asarray(keyword_vectors.toarray() if hasattr(keyword_vectors, "toarray") else keyword_vectors)

    # Reset the index so group.index gives positional offsets that align with `dense`.
    df = df.reset_index(drop=True)

    ctfidf_terms = _build_ctfidf_top_terms(df, config.n_top_terms) if strategy == "c-tfidf" else {}

    rows: list[dict[str, object]] = []
    for cluster_id, group in df.groupby("cluster_id"):
        keywords = group["keyword"].astype(str).tolist()
        if not keywords:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": f"cluster_{cluster_id}",
                    "representative_keyword": "",
                    "top_terms": "",
                    "label_confidence": 0.0,
                    "label_method": strategy,
                }
            )
            continue
        try:
            if strategy == "tfidf":
                label, top_terms, conf = _tfidf_terms(keywords, config.n_top_terms)
                rep = keywords[0]
            elif strategy == "c-tfidf":
                label, top_terms, conf = ctfidf_terms.get(cluster_id, (keywords[0].title(), [keywords[0]], 0.2))
                rep = keywords[0]
            elif strategy == "centroid":
                cluster_vectors = dense[group.index.to_numpy()] if dense is not None else None
                label, rep, conf = _centroid_keyword(keywords, cluster_vectors)
                top_terms = [rep]
            else:  # mmr
                cluster_vectors = dense[group.index.to_numpy()] if dense is not None else None
                label, top_terms, conf = _mmr_terms(keywords, cluster_vectors, config.n_top_terms, config.mmr_lambda)
                rep = top_terms[0] if top_terms else keywords[0]
        except (ValueError, IndexError, ZeroDivisionError):
            label, rep, top_terms, conf = keywords[0].title(), keywords[0], [keywords[0]], 0.2

        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": label,
                "representative_keyword": rep,
                "top_terms": ", ".join(top_terms),
                "label_confidence": conf,
                "label_method": strategy,
            }
        )
    return pd.DataFrame(rows)


def apply_cluster_labels(df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    enriched = out.merge(label_df, on="cluster_id", how="left")
    enriched["cluster_label"] = enriched["cluster_label"].fillna("unlabeled")
    enriched["representative_keyword"] = enriched["representative_keyword"].fillna("")
    enriched["top_terms"] = enriched["top_terms"].fillna("")
    enriched["label_confidence"] = enriched["label_confidence"].fillna(0.0)
    enriched["label_method"] = enriched["label_method"].fillna("tfidf")
    return enriched
