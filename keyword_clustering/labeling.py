"""Automatic cluster label generation from top TF-IDF terms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import preprocess_text


def generate_cluster_labels(
    df: pd.DataFrame,
    n_top_terms: int = 3,
) -> dict[int, str]:
    """
    For each cluster_id in df, extract the top TF-IDF terms from the
    keywords in that cluster using original (unstemmed) text for readability.

    Returns dict mapping cluster_id -> label string.
    """
    labels: dict[int, str] = {}

    for cluster_id, group in df.groupby("cluster_id"):
        keywords = group["keyword"].tolist()
        if not keywords:
            labels[cluster_id] = f"cluster_{cluster_id}"
            continue

        # Use original keywords (not stemmed) so labels are human-readable
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100, stop_words="english")
            matrix = vec.fit_transform(keywords)
            scores = matrix.sum(axis=0).A1
            terms = vec.get_feature_names_out()
            top_idx = scores.argsort()[::-1][:n_top_terms]
            top_terms = [terms[i] for i in top_idx]
            labels[cluster_id] = " / ".join(top_terms).title()
        except ValueError:
            labels[cluster_id] = keywords[0].title()

    return labels


def apply_cluster_labels(df: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    df["cluster_label"] = df["cluster_id"].map(labels).fillna("unlabeled")
    return df
