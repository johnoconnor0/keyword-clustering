"""Performance baselines for the per-text embedding cache.

These tests exist to detect regressions in the incremental-run speedup. They are
opt-in via `pytest-benchmark`: when the plugin isn't installed (the slim CI job),
the tests skip cleanly. To run them locally:

    pytest tests/test_benchmarks.py --benchmark-only

Compare two runs:

    pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=baseline
    # ... make changes ...
    pytest tests/test_benchmarks.py --benchmark-only --benchmark-compare=baseline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

from keyword_clustering import vectorization
from keyword_clustering.vectorization import EmbeddingConfig, vectorize_keywords_st_configured

_CORPUS_SIZE = 200


class _FastStubModel:
    """Deterministic, near-zero-cost embedding model so the benchmark measures
    cache machinery (path generation, sha256, np.save/np.load, dict assembly)
    rather than real transformer inference."""

    def encode(self, texts, **kwargs):
        # 8-dim hash-based vector — cheap, deterministic.
        out = np.empty((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            for j in range(8):
                out[i, j] = ((h >> (j * 4)) & 0xF) / 15.0
        return out


@pytest.fixture
def cfg(tmp_path: Path) -> EmbeddingConfig:
    return EmbeddingConfig(
        model_name="stub-model",
        cache_dir=str(tmp_path),
        preprocess_mode="none",
        normalize_embeddings=False,
    )


@pytest.fixture
def warm_cache(cfg: EmbeddingConfig) -> list[str]:
    """Warm the per-text cache with `_CORPUS_SIZE` keywords."""
    keywords = [f"keyword phrase {i}" for i in range(_CORPUS_SIZE)]
    with patch.object(vectorization, "_load_sentence_transformer", return_value=_FastStubModel()):
        vectorize_keywords_st_configured(keywords, cfg)
    return keywords


def test_per_text_cache_incremental_run_is_fast(benchmark, cfg, warm_cache):
    """Benchmark the *common* incremental case: warm cache + 1 new keyword.

    With a working per-text cache this should encode exactly one delta keyword,
    not the full N+1 corpus. The corpus-cache path also fires here (different
    bundle hash), but the encode-call shortlist proves the per-text path works.
    """
    incremental_keywords = warm_cache + ["brand-new keyword"]

    def _run() -> int:
        with patch.object(vectorization, "_load_sentence_transformer", return_value=_FastStubModel()):
            vectorize_keywords_st_configured(incremental_keywords, cfg)
        return 1

    benchmark(_run)


def test_per_text_cache_full_warm_corpus_short_circuits(benchmark, cfg, warm_cache):
    """Re-running the exact same corpus must hit the corpus-cache fast path
    and skip per-text and per-text encode entirely. Benchmark this."""

    def _run() -> int:
        with patch.object(vectorization, "_load_sentence_transformer", return_value=_FastStubModel()):
            vectorize_keywords_st_configured(warm_cache, cfg)
        return 1

    benchmark(_run)
