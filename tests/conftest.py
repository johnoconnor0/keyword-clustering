"""Shared pytest fixtures for the keyword-clustering test suite."""

from __future__ import annotations

import nltk
import pytest


@pytest.fixture(autouse=True, scope="session")
def _download_nltk_data():
    """Ensure NLTK corpora used by preprocessing are present.

    On CI the workflow handles this explicitly, but a fresh local clone may
    not. This fixture downloads quietly on demand so `pytest tests/` works
    without manual setup.
    """
    for name in ("stopwords", "wordnet", "omw-1.4"):
        try:
            if name == "stopwords":
                nltk.data.find("corpora/stopwords")
            else:
                nltk.data.find(f"corpora/{name}")
        except LookupError:
            nltk.download(name, quiet=True)
    yield
