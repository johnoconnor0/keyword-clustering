"""CSV connector adapters for SEO tool exports."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd

REQUIRED_BY_SOURCE = {
    "gsc": ["query"],
    "ga4": ["landing_page"],
    "ahrefs": ["keyword"],
    "semrush": ["keyword"],
    "screamingfrog": ["address"],
    "sitebulb": ["url"],
}


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._in_h1 = False
        self._in_h2 = False
        self._in_body = False
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.h2_parts: list[str] = []
        self.body_parts: list[str] = []
        self.meta_description = ""
        self.canonical_url = ""

    def handle_starttag(self, tag: str, attrs):
        attrs_map = {k.lower(): v for k, v in attrs}
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        elif tag == "h1":
            self._in_h1 = True
        elif tag == "h2":
            self._in_h2 = True
        elif tag == "body":
            self._in_body = True
        elif tag == "meta" and attrs_map.get("name", "").lower() == "description":
            self.meta_description = attrs_map.get("content", "") or self.meta_description
        elif tag == "link" and attrs_map.get("rel", "").lower() == "canonical":
            self.canonical_url = attrs_map.get("href", "") or self.canonical_url

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False
        elif tag == "h2":
            self._in_h2 = False
        elif tag == "body":
            self._in_body = False

    def handle_data(self, data: str):
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        if self._in_h1:
            self.h1_parts.append(text)
        if self._in_h2:
            self.h2_parts.append(text)
        if self._in_body:
            self.body_parts.append(text)


def crawl_page(url: str, timeout: int = 20) -> dict[str, object]:
    req = Request(
        url,
        headers={"User-Agent": "keyword-clustering-bot/0.1 (+https://github.com/johnoconnor0/keyword-clustering)"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:  # nosec B310 - user-directed crawl utility
            status_code = int(getattr(resp, "status", 200))
            content_type = str(resp.headers.get("Content-Type", ""))
            html_bytes = resp.read()
    except URLError as exc:
        return {"status_code": 0, "error": str(exc), "url": url}
    except Exception as exc:  # pragma: no cover
        return {"status_code": 0, "error": str(exc), "url": url}

    if "text/html" not in content_type.lower():
        return {
            "status_code": status_code,
            "error": f"Unsupported content-type: {content_type}",
            "url": url,
        }

    raw_html = html_bytes.decode("utf-8", errors="ignore")
    parser = _PageParser()
    parser.feed(raw_html)
    title = " ".join(parser.title_parts).strip()
    h1 = " ".join(parser.h1_parts).strip()
    headings = " | ".join(parser.h2_parts[:12]).strip()
    body_text = " ".join(parser.body_parts).strip()
    body_text = re.sub(r"\s+", " ", html.unescape(body_text))
    excerpt = body_text[:500]
    canonical = parser.canonical_url.strip()
    if canonical and not canonical.startswith(("http://", "https://")):
        canonical = urljoin(url, canonical)

    return {
        "url": url,
        "status_code": status_code,
        "error": "",
        "title": title,
        "meta_description": parser.meta_description.strip(),
        "h1": h1,
        "headings": headings,
        "body_excerpt": excerpt,
        "canonical_url": canonical or url,
        "word_count": len(body_text.split()) if body_text else 0,
    }


def enrich_pages_dataframe(pages_df: pd.DataFrame, timeout: int = 20) -> pd.DataFrame:
    if "url" not in pages_df.columns:
        raise ValueError("Pages data must include a 'url' column.")
    rows = []
    for _, row in pages_df.iterrows():
        url = str(row.get("url", "")).strip()
        if not url:
            continue
        crawled = crawl_page(url, timeout=timeout)
        merged = row.to_dict()
        merged.update(crawled)
        rows.append(merged)
    return pd.DataFrame(rows)


def normalize_connector_csv(source: str, df: pd.DataFrame) -> pd.DataFrame:
    key = source.lower()
    if key not in REQUIRED_BY_SOURCE:
        raise ValueError(f"Unknown source: {source}")
    missing = [c for c in REQUIRED_BY_SOURCE[key] if c not in df.columns.str.lower().tolist()]
    if missing:
        raise ValueError(f"{source} file is missing required columns: {', '.join(missing)}")

    renamed = df.copy()
    lower_map = {c.lower(): c for c in renamed.columns}
    if key == "gsc":
        renamed = renamed.rename(columns={lower_map.get("query", "query"): "keyword"})
    elif key in {"ahrefs", "semrush"}:
        renamed = renamed.rename(columns={lower_map.get("keyword", "keyword"): "keyword"})
    elif key == "ga4":
        renamed = renamed.rename(columns={lower_map.get("landing_page", "landing_page"): "url"})
    elif key in {"screamingfrog", "sitebulb"}:
        src = lower_map.get("address") or lower_map.get("url")
        renamed = renamed.rename(columns={src: "url"})
    return renamed
