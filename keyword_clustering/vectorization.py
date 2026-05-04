"""TF-IDF and Sentence Transformer vectorization."""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess_text


def build_tfidf_vectorizer(corpus: list[str]) -> tuple[TfidfVectorizer, object]:
    """Fit TF-IDF on preprocessed corpus. Returns (vectorizer, matrix)."""
    processed = [preprocess_text(t) for t in corpus]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(processed)
    return vec, matrix


def vectorize_keywords_tfidf(
    keywords: list[str],
    vectorizer: TfidfVectorizer,
) -> object:
    """Transform keywords using a fitted TF-IDF vectorizer."""
    processed = [preprocess_text(kw) for kw in keywords]
    return vectorizer.transform(processed)


def vectorize_keywords_st(keywords: list[str], model_name: str) -> np.ndarray:
    """Encode keywords with a Sentence Transformer model."""
    from sentence_transformers import SentenceTransformer  # lazy import

    model = SentenceTransformer(model_name)
    return model.encode(keywords, show_progress_bar=True, convert_to_numpy=True)


def compute_similarity_matrix(
    query_vectors: object,
    corpus_vectors: object,
) -> np.ndarray:
    """Cosine similarity: (n_queries, n_corpus)."""
    return cosine_similarity(query_vectors, corpus_vectors)


def to_dense(matrix: object) -> np.ndarray:
    if issparse(matrix):
        return matrix.toarray()
    return np.array(matrix)
