"""Tests for connector and crawl integrations."""

from __future__ import annotations

import pandas as pd

from keyword_clustering.integrations import crawl_page, enrich_pages_dataframe, normalize_connector_csv


class _FakeResponse:
    def __init__(self, html_text: str, status: int = 200, content_type: str = "text/html") -> None:
        self.status = status
        self._bytes = html_text.encode("utf-8")
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_normalize_connector_csv():
    gsc = pd.DataFrame({"query": ["seo audit"], "clicks": [10]})
    out = normalize_connector_csv("gsc", gsc)
    assert "keyword" in out.columns


def test_crawl_page_parses_fields(monkeypatch):
    html_doc = """
    <html>
      <head>
        <title>SEO Audit Services</title>
        <meta name="description" content="Technical SEO audits and recommendations.">
        <link rel="canonical" href="/seo-audit" />
      </head>
      <body>
        <h1>Technical SEO Audit</h1>
        <h2>What is included</h2>
        <h2>Pricing</h2>
        <p>We audit crawlability and indexation.</p>
      </body>
    </html>
    """

    monkeypatch.setattr("keyword_clustering.integrations.urlopen", lambda req, timeout=20: _FakeResponse(html_doc))
    result = crawl_page("https://example.com/seo-audit")
    assert result["status_code"] == 200
    assert "SEO Audit Services" in str(result["title"])
    assert "Technical SEO Audit" in str(result["h1"])
    assert "What is included" in str(result["headings"])


def test_enrich_pages_dataframe(monkeypatch):
    monkeypatch.setattr(
        "keyword_clustering.integrations.crawl_page",
        lambda url, timeout=20: {
            "url": url,
            "status_code": 200,
            "title": "OK",
            "meta_description": "",
            "h1": "",
            "headings": "",
            "body_excerpt": "",
            "canonical_url": url,
            "word_count": 1,
            "error": "",
        },
    )
    pages = pd.DataFrame({"url": ["https://example.com/a"], "page_name": ["A"]})
    out = enrich_pages_dataframe(pages, timeout=5)
    assert len(out) == 1
    assert out.loc[0, "status_code"] == 200
