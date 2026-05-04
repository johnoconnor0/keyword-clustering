FROM python:3.11-slim

WORKDIR /app

# Build essentials for numpy/scipy/scikit-learn wheels that may need to compile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Bring in the source first so `pip install .` sees the real package layout.
# `requirements.txt` was removed in v0.2.0 — `pyproject.toml` is now the single
# source of truth for dependencies. The `[app]` extra ships streamlit + plotly +
# kaleido, `[semantic]` adds sentence-transformers, `[advanced]` adds hdbscan +
# umap-learn + hnswlib.
COPY . .

RUN pip install --no-cache-dir ".[app,semantic,advanced]" \
    && python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
