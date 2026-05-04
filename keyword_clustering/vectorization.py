"""TF-IDF and Sentence Transformer vectorization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess_text

ST_PRESETS = {
    "tfidf": "tfidf",
    "mini": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2",
    "e5-small": "intfloat/e5-small-v2",
    "e5-base": "intfloat/e5-base-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
}


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "tfidf"
    batch_size: int = 64
    device: str = "auto"
    normalize_embeddings: bool = True
    cache_dir: str | None = ".cache/embeddings"
    preprocess_mode: str = "none"
    chunk_size: int = 0
    tfidf_svd_components: int = 0
    text_mode: str = "keyword"

    @property
    def resolved_model_name(self) -> str:
        return ST_PRESETS.get(self.model_name, self.model_name)


def build_tfidf_vectorizer(
    corpus: list[str],
    preprocess_mode: str = "stem",
) -> tuple[TfidfVectorizer, object]:
    """Fit TF-IDF on a preprocessed corpus. Returns (vectorizer, matrix).

    Stopword handling: when preprocess_mode removes stopwords (`stem` / `lemmatize`),
    we skip the vectorizer-level pass to avoid double removal. For `none` / `light`
    we still want the vectorizer to drop English stopwords.
    """
    processed = [preprocess_text(t, mode=preprocess_mode) for t in corpus]
    vec_stop_words: str | None = None if preprocess_mode in {"stem", "lemmatize"} else "english"
    vec = TfidfVectorizer(stop_words=vec_stop_words, ngram_range=(1, 2))
    matrix = vec.fit_transform(processed)
    return vec, matrix


def vectorize_keywords_tfidf(
    keywords: list[str],
    vectorizer: TfidfVectorizer,
    preprocess_mode: str = "stem",
) -> object:
    """Transform keywords using a fitted TF-IDF vectorizer."""
    processed = [preprocess_text(kw, mode=preprocess_mode) for kw in keywords]
    return vectorizer.transform(processed)


def _embedding_config_signature(cfg: EmbeddingConfig) -> str:
    """Hash for the cfg slice that affects embedding outputs (independent of texts)."""
    payload = {
        "model": cfg.resolved_model_name,
        "device": cfg.device,
        "normalize": cfg.normalize_embeddings,
        "preprocess": cfg.preprocess_mode,
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _embedding_cache_key(texts: list[str], cfg: EmbeddingConfig) -> str:
    """Whole-corpus cache key. Streams the texts through sha256 so we don't
    JSON-encode and hold a multi-MB payload in memory for 10k+ inputs."""
    h = hashlib.sha256()
    h.update(_embedding_config_signature(cfg).encode("utf-8"))
    for chunk in texts:
        h.update(b"\n")
        h.update(chunk.encode("utf-8"))
    return h.hexdigest()


def _per_text_cache_path(text: str, cfg: EmbeddingConfig, cache_dir: Path) -> Path:
    """Per-text cache path. Adding one keyword to a corpus invalidates only that
    keyword's cache entry, not the whole bundle — incremental runs reuse most work."""
    sig = _embedding_config_signature(cfg)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return cache_dir / "text" / sig / f"{text_hash}.npy"


def _resolve_device(device: str) -> str:
    if device in {"cpu", "cuda", "mps"}:
        return device
    if device != "auto":
        raise ValueError("embedding device must be one of: auto, cpu, cuda, mps")
    try:
        import torch  # type: ignore
    except (ImportError, AttributeError):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=6)
def _load_sentence_transformer(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer  # lazy import

    return SentenceTransformer(model_name, device=device)


def vectorize_keywords_st_configured(keywords: list[str], cfg: EmbeddingConfig) -> np.ndarray:
    """Encode keywords with a Sentence Transformer model using full config and cache.

    Cache strategy (when ``cfg.cache_dir`` is set):
      1. Whole-corpus key (`<dir>/<sha>.npy`) — fast hit when re-running with
         identical keyword list and config.
      2. Per-text cache (`<dir>/text/<sig>/<hash>.npy`) — populated alongside
         the corpus cache so that adding/removing one keyword only re-encodes
         the delta, not the whole bundle.
    """
    texts = [preprocess_text(kw, mode=cfg.preprocess_mode) for kw in keywords]
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    corpus_cache_path: Path | None = None
    text_cache_root: Path | None = None
    if cfg.cache_dir:
        cache_dir = Path(cfg.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        corpus_cache_path = cache_dir / f"{_embedding_cache_key(texts, cfg)}.npy"
        if corpus_cache_path.exists():
            return np.load(corpus_cache_path)
        text_cache_root = cache_dir / "text" / _embedding_config_signature(cfg)
        text_cache_root.mkdir(parents=True, exist_ok=True)

    # Per-text cache lookup — collect hits, encode only the misses.
    cached_vectors: dict[int, np.ndarray] = {}
    misses_idx: list[int] = []
    misses_texts: list[str] = []
    if text_cache_root is not None and cfg.cache_dir is not None:
        cache_root_path = Path(cfg.cache_dir)
        for i, t in enumerate(texts):
            p = _per_text_cache_path(t, cfg, cache_root_path)
            if p.exists():
                cached_vectors[i] = np.load(p)
            else:
                misses_idx.append(i)
                misses_texts.append(t)
    else:
        misses_idx = list(range(len(texts)))
        misses_texts = list(texts)

    encoded_misses: np.ndarray | None = None
    if misses_texts:
        device = _resolve_device(cfg.device)
        model = _load_sentence_transformer(cfg.resolved_model_name, device)
        if cfg.chunk_size and cfg.chunk_size > 0 and len(misses_texts) > cfg.chunk_size:
            chunks: list[np.ndarray] = []
            for start in range(0, len(misses_texts), cfg.chunk_size):
                part = misses_texts[start : start + cfg.chunk_size]
                part_vec = model.encode(
                    part,
                    batch_size=cfg.batch_size,
                    normalize_embeddings=cfg.normalize_embeddings,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )
                chunks.append(np.asarray(part_vec))
            encoded_misses = np.vstack(chunks)
        else:
            encoded_misses = model.encode(
                misses_texts,
                batch_size=cfg.batch_size,
                normalize_embeddings=cfg.normalize_embeddings,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        # Populate per-text cache for the misses.
        if text_cache_root is not None and cfg.cache_dir is not None:
            cache_root_path = Path(cfg.cache_dir)
            for n_pos, t in enumerate(misses_texts):
                p = _per_text_cache_path(t, cfg, cache_root_path)
                np.save(p, encoded_misses[n_pos])

    # Reassemble in original order.
    if cached_vectors and encoded_misses is not None:
        sample_dim = encoded_misses.shape[1]
        embeddings = np.empty((len(texts), sample_dim), dtype=np.float32)
        for i, vec in cached_vectors.items():
            embeddings[i] = vec
        for n, i in enumerate(misses_idx):
            embeddings[i] = encoded_misses[n]
    elif cached_vectors:  # all-cache hit (shouldn't happen — corpus cache would have hit first)
        sample_dim = next(iter(cached_vectors.values())).shape[0]
        embeddings = np.empty((len(texts), sample_dim), dtype=np.float32)
        for i, vec in cached_vectors.items():
            embeddings[i] = vec
    else:
        embeddings = encoded_misses if encoded_misses is not None else np.empty((0, 0), dtype=np.float32)

    if corpus_cache_path is not None:
        np.save(corpus_cache_path, embeddings)
    return embeddings


def compute_similarity_matrix(
    query_vectors: object,
    corpus_vectors: object,
) -> np.ndarray:
    """Cosine similarity: (n_queries, n_corpus)."""
    return cosine_similarity(query_vectors, corpus_vectors)


def compute_serp_overlap_matrix(keywords_df) -> np.ndarray:
    """Compute Jaccard overlap of SERP URL sets between keywords.

    Vectorised: builds a sparse boolean keyword × url indicator matrix M, then
    intersection counts come from M @ M.T and union counts from row sums minus
    intersections. This is O(n · avg_urls) instead of O(n²) Python set ops.
    """
    if "serp_urls" not in keywords_df.columns:
        n = len(keywords_df)
        return np.zeros((n, n), dtype=float)

    from sklearn.feature_extraction.text import CountVectorizer

    def _tokenise(raw: object) -> str:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return ""
        return " ".join(p.strip() for p in str(raw).split("|") if p.strip())

    docs = [_tokenise(v) for v in keywords_df["serp_urls"].tolist()]
    n = len(docs)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if all(not d for d in docs):
        return np.zeros((n, n), dtype=float)

    # Use a token pattern that keeps full URLs intact (anything non-whitespace).
    vec = CountVectorizer(token_pattern=r"\S+", lowercase=False, binary=True)
    try:
        m = vec.fit_transform(docs).astype(bool).astype(np.float32)
    except ValueError:
        return np.zeros((n, n), dtype=float)
    intersections = (m @ m.T).toarray()
    row_sums = np.asarray(m.sum(axis=1)).ravel()
    unions = row_sums[:, None] + row_sums[None, :] - intersections
    out = np.zeros((n, n), dtype=float)
    nonzero = unions > 0
    out[nonzero] = intersections[nonzero] / unions[nonzero]
    return out


def to_dense(matrix: object) -> np.ndarray:
    if issparse(matrix):
        # mypy doesn't narrow `object` through issparse, but the runtime guarantee holds.
        return np.asarray(matrix.toarray())  # type: ignore[attr-defined]
    return np.asarray(matrix)


def normalise_rows(matrix: object) -> np.ndarray:
    """L2-normalize row vectors for cosine-friendly downstream algorithms."""
    dense = to_dense(matrix).astype(np.float32, copy=False)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms


def compose_hybrid_feature_vectors(
    semantic_vectors: object | None,
    tfidf_vectors: object | None,
    semantic_weight: float,
    tfidf_weight: float,
) -> object:
    """Build weighted feature vectors by concatenating L2-normalised channels.

    Preserves sparsity: if the TF-IDF block is sparse, the result is a
    `scipy.sparse.csr_matrix`; otherwise it is a dense `np.ndarray`. Returning
    sparse avoids materialising a `(n, vocab_size)` dense matrix when the TF-IDF
    vocabulary is large — the previous version OOM'd on 50k-row + 200k-vocab inputs.

    Downstream callers must handle both shapes; `to_dense()` is the unified
    entrypoint for code that needs `np.ndarray`.
    """
    from scipy.sparse import hstack as sparse_hstack
    from scipy.sparse import issparse
    from sklearn.preprocessing import normalize

    semantic_block: object | None = None
    tfidf_block: object | None = None

    if semantic_vectors is not None and semantic_weight > 0:
        # Semantic vectors are typically dense; normalise + scale in float32.
        sem = np.asarray(semantic_vectors).astype(np.float32, copy=False)
        sem = normalize(sem, norm="l2", axis=1) * float(semantic_weight)
        semantic_block = sem

    if tfidf_vectors is not None and tfidf_weight > 0:
        if issparse(tfidf_vectors):
            # Keep TF-IDF sparse — normalize + scalar multiply preserve sparsity.
            tfidf_block = normalize(tfidf_vectors, norm="l2", axis=1) * float(tfidf_weight)
        else:
            tfi = np.asarray(tfidf_vectors).astype(np.float32, copy=False)
            tfidf_block = normalize(tfi, norm="l2", axis=1) * float(tfidf_weight)

    blocks = [b for b in (semantic_block, tfidf_block) if b is not None]
    if not blocks:
        raise ValueError("Cannot compose hybrid vectors without at least one weighted vector source.")
    if len(blocks) == 1:
        return blocks[0]

    # Mixed dense + sparse: convert everything to sparse for hstack to preserve memory.
    if any(issparse(b) for b in blocks):
        from scipy.sparse import csr_matrix

        return sparse_hstack([b if issparse(b) else csr_matrix(b) for b in blocks], format="csr")
    # All blocks are dense ndarrays here.
    dense_blocks = [np.asarray(b) for b in blocks]
    return np.concatenate(dense_blocks, axis=1)
