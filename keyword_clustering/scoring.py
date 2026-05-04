"""Similarity scoring, page mapping, opportunity scoring, gap and cannibalization detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .vectorization import compute_similarity_matrix


def _normalise_series(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]; returns 0.5 for constant series."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def score_keywords_against_corpus(
    keyword_vectors: object,
    corpus_vectors: object,
    corpus_names: list[str],
) -> tuple[list[str], list[float]]:
    """
    For each keyword vector, find the most similar item in corpus.

    Returns (best_name_per_keyword, best_score_per_keyword).
    """
    sim = compute_similarity_matrix(keyword_vectors, corpus_vectors)
    best_idx = sim.argmax(axis=1)
    best_scores = sim.max(axis=1).tolist()
    best_names = [corpus_names[i] for i in best_idx]
    return best_names, best_scores


def map_keywords_to_pages(
    df: pd.DataFrame,
    keyword_vectors: object,
    page_vectors: object,
    page_names: list[str],
    page_urls: list[str],
    gap_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Assign recommended_page and recommended_url to each keyword.
    Keywords with max page similarity at or below gap_threshold are flagged as content gaps.
    """
    sim = compute_similarity_matrix(keyword_vectors, page_vectors)
    best_idx = sim.argmax(axis=1)
    best_scores = sim.max(axis=1).tolist()

    df = df.copy()
    df["recommended_page"] = [page_names[i] for i in best_idx]
    df["recommended_url"] = [page_urls[i] for i in best_idx]
    df["page_similarity_score"] = [round(s, 4) for s in best_scores]
    df["content_gap"] = df["page_similarity_score"] <= gap_threshold

    if "current_url" in df.columns:
        url_to_page = dict(zip(page_urls, page_names))
        df["current_page"] = df["current_url"].map(url_to_page).fillna("")

    return df


def compute_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opportunity_score column.

    opportunity = (volume_norm * 0.4) - (difficulty_norm * 0.3) + (rank_gap_norm * 0.3)
    All inputs are optional; missing columns default to 0.5 (neutral contribution).
    """
    df = df.copy()

    vol_norm = _normalise_series(df["search_volume"].fillna(df["search_volume"].median())) \
        if "search_volume" in df.columns else pd.Series(0.5, index=df.index)
    diff_norm = _normalise_series(df["keyword_difficulty"].fillna(50)) \
        if "keyword_difficulty" in df.columns else pd.Series(0.5, index=df.index)

    if "rank" in df.columns:
        rank_gap: pd.Series = df["rank"].fillna(100).clip(upper=100).apply(
            lambda r: max(0.0, float(r) - 10) / 90
        )
    else:
        rank_gap = pd.Series(0.5, index=df.index)

    df["opportunity_score"] = (
        vol_norm * 0.4 - diff_norm * 0.3 + rank_gap * 0.3
    ).round(4)
    return df


def detect_cannibalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find clusters where 2+ distinct pages are mapped as recommended_page.

    Returns a summary DataFrame with cluster_id, cluster_label, competing_pages.
    """
    if "cluster_id" not in df.columns or "recommended_page" not in df.columns:
        return pd.DataFrame()

    rows = []
    for cluster_id, group in df.groupby("cluster_id"):
        pages = group["recommended_page"].dropna().unique().tolist()
        if len(pages) > 1:
            label = group["cluster_label"].iloc[0] if "cluster_label" in group.columns else str(cluster_id)
            rows.append({
                "cluster_id": cluster_id,
                "cluster_label": label,
                "competing_pages": ", ".join(pages),
                "keyword_count": len(group),
                "keywords_sample": ", ".join(group["keyword"].head(5).tolist()),
            })
    return pd.DataFrame(rows)


def add_notes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a human-readable notes column summarising key observations.

    Flags: page mismatches, content gaps, high-opportunity keywords.
    """
    df = df.copy()
    notes = pd.Series([""] * len(df), index=df.index, dtype=str)

    if "current_page" in df.columns and "recommended_page" in df.columns:
        curr = df["current_page"].fillna("").astype(str)
        rec = df["recommended_page"].fillna("").astype(str)
        mismatch = (curr != "") & (curr != rec)
        notes[mismatch] = (
            "Page mismatch — currently: " + curr[mismatch] + ", recommended: " + rec[mismatch]
        )

    if "content_gap" in df.columns:
        gap = df["content_gap"].fillna(False).astype(bool)
        gap_note = "Content gap — no suitable page found"
        notes[gap] = notes[gap].apply(lambda n: (n + "; " + gap_note) if n else gap_note)

    if "opportunity_score" in df.columns:
        hi = df["opportunity_score"].fillna(0) > 0.3
        hi_note = "High opportunity"
        notes[hi] = notes[hi].apply(lambda n: (n + "; " + hi_note) if n else hi_note)

    df["notes"] = notes
    return df


def detect_content_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Return keywords flagged as content gaps (no good page match)."""
    if "content_gap" not in df.columns:
        return pd.DataFrame()
    cols = ["keyword", "cluster_label", "primary_intent", "search_volume",
            "keyword_difficulty", "opportunity_score", "page_similarity_score"]
    cols = [c for c in cols if c in df.columns]
    return df[df["content_gap"]][cols].reset_index(drop=True)
