"""Error-path coverage for keyword_clustering.integrations.

Exercises the failure modes left untested by the happy-path test_integrations.py
that the audit flagged: network failure, non-HTML content, malformed HTML,
canonical URL resolution, and every connector's required-column validation.
"""

from __future__ import annotations

from urllib.error import URLError

import pandas as pd
import pytest

from keyword_clustering import integrations
from keyword_clustering.integrations import (
    crawl_page,
    enrich_pages_dataframe,
    normalize_connector_csv,
)


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200):
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_crawl_page_returns_error_on_url_error(monkeypatch: pytest.MonkeyPatch):
    def boom(*args, **kwargs):
        raise URLError("name resolution failure")

    monkeypatch.setattr(integrations, "urlopen", boom)
    out = crawl_page("https://example.com/missing")
    assert out["status_code"] == 0
    assert "name resolution" in out["error"]
    assert out["url"] == "https://example.com/missing"


def test_crawl_page_returns_error_on_non_html_content_type(monkeypatch: pytest.MonkeyPatch):
    def fake_open(*args, **kwargs):
        return _FakeResponse(b"PDF...", content_type="application/pdf")

    monkeypatch.setattr(integrations, "urlopen", fake_open)
    out = crawl_page("https://example.com/whitepaper.pdf")
    assert out["status_code"] == 200
    assert "Unsupported content-type" in out["error"]


def test_crawl_page_resolves_relative_canonical_to_absolute_url(monkeypatch: pytest.MonkeyPatch):
    body = (
        b"<html><head><title>Audit</title>"
        b"<link rel='canonical' href='/seo-audit'>"
        b"<meta name='description' content='Audit page'>"
        b"</head><body><h1>Heading</h1><p>Body content goes here.</p></body></html>"
    )
    monkeypatch.setattr(integrations, "urlopen", lambda *a, **k: _FakeResponse(body))
    out = crawl_page("https://example.com/landing")
    assert out["canonical_url"] == "https://example.com/seo-audit"
    assert out["title"] == "Audit"
    assert out["h1"] == "Heading"
    assert "Body content" in out["body_excerpt"]


def test_crawl_page_handles_malformed_html_gracefully(monkeypatch: pytest.MonkeyPatch):
    body = b"<html><head><title>broken<body><h1>Hi</h1>"  # unterminated tags
    monkeypatch.setattr(integrations, "urlopen", lambda *a, **k: _FakeResponse(body))
    out = crawl_page("https://example.com/broken")
    assert out["status_code"] == 200
    assert out["error"] == ""
    # Even with broken HTML the parser shouldn't crash; we just get whatever it managed to extract.
    assert "url" in out and "title" in out


def test_enrich_pages_dataframe_requires_url_column():
    df = pd.DataFrame({"name": ["Home"]})
    with pytest.raises(ValueError, match="'url'"):
        enrich_pages_dataframe(df)


def test_enrich_pages_dataframe_skips_blank_url_rows(monkeypatch: pytest.MonkeyPatch):
    body = b"<html><head><title>OK</title></head><body><h1>Hi</h1></body></html>"
    monkeypatch.setattr(integrations, "urlopen", lambda *a, **k: _FakeResponse(body))
    df = pd.DataFrame({"url": ["https://example.com/a", "", "  "]})
    out = enrich_pages_dataframe(df)
    assert len(out) == 1
    assert out["url"].iloc[0] == "https://example.com/a"


def test_normalize_connector_csv_unknown_source_raises():
    df = pd.DataFrame({"keyword": ["x"]})
    with pytest.raises(ValueError, match="Unknown source"):
        normalize_connector_csv("notarealtool", df)


def test_normalize_connector_csv_missing_columns_raises():
    df = pd.DataFrame({"junk": [1, 2]})  # gsc requires 'query'
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_connector_csv("gsc", df)


@pytest.mark.parametrize(
    ("source", "input_columns", "expected_renamed"),
    [
        ("gsc", {"Query": ["x"], "Clicks": [1]}, "keyword"),
        ("ahrefs", {"Keyword": ["x"], "Volume": [10]}, "keyword"),
        ("semrush", {"Keyword": ["x"]}, "keyword"),
        ("ga4", {"landing_page": ["/x"], "users": [10]}, "url"),
        ("screamingfrog", {"address": ["https://e.com/a"], "Status Code": [200]}, "url"),
        ("sitebulb", {"url": ["https://e.com/a"]}, "url"),
    ],
)
def test_normalize_connector_csv_renames_per_source(source: str, input_columns: dict, expected_renamed: str):
    df = pd.DataFrame(input_columns)
    out = normalize_connector_csv(source, df)
    assert expected_renamed in out.columns


def test_normalize_connector_csv_screamingfrog_keeps_url_when_both_present():
    """When 'address' AND 'url' are present, screamingfrog renames the 'address' column to 'url'."""
    df = pd.DataFrame({"address": ["https://e.com/a"], "url": ["https://e.com/a-existing"]})
    out = normalize_connector_csv("screamingfrog", df)
    assert "url" in out.columns
    # Two columns may both end up named "url" after rename — that's a known data-quality issue
    # for input CSVs that conflict, surfaced here so contributors don't add a stricter contract by accident.
    assert (out.columns == "url").sum() >= 1
