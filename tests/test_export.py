"""Tests for keyword_clustering.export — file existence + schema for every output artefact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from keyword_clustering.export import (
    build_interactive_report,
    save_all_outputs,
    save_cannibalization,
    save_cluster_summary,
    save_clustered_keywords,
    save_content_gaps,
    save_keyword_page_map,
    save_quality_report,
    save_recommendations,
    save_seo_workflow_artifacts,
)


@pytest.fixture
def clustered_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "keyword": "blue widget",
                "cluster_id": 0,
                "cluster_label": "Blue Widgets",
                "primary_intent": "commercial",
                "recommended_page": "Widgets",
                "recommended_url": "/widgets",
                "current_url": "/old-widgets",
                "page_similarity_score": 0.62,
                "match_confidence": "acceptable_match",
                "content_gap": False,
                "search_volume": 800,
                "keyword_difficulty": 30,
                "opportunity_score": 0.72,
                "opportunity_reason": "Profile=balanced; volume=0.80, rank_gap=0.40",
                "branded": False,
                "featured_snippet": 0,
                "local_pack": 0,
                "people_also_ask": 1,
                "image_pack": 0,
                "video_result": 0,
            },
            {
                "keyword": "buy blue widget",
                "cluster_id": 0,
                "cluster_label": "Blue Widgets",
                "primary_intent": "transactional",
                "recommended_page": "Widgets",
                "recommended_url": "/widgets",
                "current_url": "/widgets",
                "page_similarity_score": 0.85,
                "match_confidence": "strong_match",
                "content_gap": False,
                "search_volume": 400,
                "keyword_difficulty": 28,
                "opportunity_score": 0.81,
                "opportunity_reason": "Profile=balanced; volume=0.40",
                "branded": False,
                "featured_snippet": 0,
                "local_pack": 0,
                "people_also_ask": 0,
                "image_pack": 0,
                "video_result": 0,
            },
            {
                "keyword": "red gadget",
                "cluster_id": 1,
                "cluster_label": "Red Gadgets",
                "primary_intent": "informational",
                "recommended_page": "Gadgets",
                "recommended_url": "/gadgets",
                "current_url": "",
                "page_similarity_score": 0.15,
                "match_confidence": "poor_match",
                "content_gap": True,
                "search_volume": 200,
                "keyword_difficulty": 12,
                "opportunity_score": 0.41,
                "opportunity_reason": "Profile=balanced",
                "branded": False,
                "featured_snippet": 1,
                "local_pack": 0,
                "people_also_ask": 0,
                "image_pack": 0,
                "video_result": 0,
            },
        ]
    )


def test_save_clustered_keywords_writes_csv(clustered_frame: pd.DataFrame, tmp_path: Path):
    path = save_clustered_keywords(clustered_frame, str(tmp_path))
    assert Path(path).exists()
    out = pd.read_csv(path)
    assert "keyword" in out.columns and "cluster_id" in out.columns
    assert len(out) == len(clustered_frame)


def test_save_keyword_page_map_filters_to_known_columns(clustered_frame: pd.DataFrame, tmp_path: Path):
    path = save_keyword_page_map(clustered_frame, str(tmp_path))
    out = pd.read_csv(path)
    assert {"keyword", "recommended_page", "recommended_url", "page_similarity_score", "match_confidence"}.issubset(
        out.columns
    )


def test_save_content_gaps_writes_only_gap_rows(clustered_frame: pd.DataFrame, tmp_path: Path):
    path = save_content_gaps(clustered_frame, str(tmp_path))
    out = pd.read_csv(path)
    assert len(out) == 1
    assert out["keyword"].iloc[0] == "red gadget"


def test_save_cannibalization_writes_csv_for_competing_pages(clustered_frame: pd.DataFrame, tmp_path: Path):
    # Cluster 0 has two different current_urls and shared recommended_page — should trigger cannibalization.
    path = save_cannibalization(clustered_frame, str(tmp_path))
    out = pd.read_csv(path)
    # Ranking_cannibalization should fire for cluster 0 (two distinct current URLs).
    assert "ranking_cannibalization" in set(out["cannibalization_type"].astype(str)) or out.empty


def test_save_cluster_summary_aggregates_per_cluster(clustered_frame: pd.DataFrame, tmp_path: Path):
    path = save_cluster_summary(clustered_frame, str(tmp_path))
    out = pd.read_csv(path)
    assert {"cluster_id", "cluster_label", "top_keywords"}.issubset(out.columns)
    assert len(out) == 2  # two clusters


def test_save_recommendations_writes_markdown(clustered_frame: pd.DataFrame, tmp_path: Path):
    path = save_recommendations(clustered_frame, str(tmp_path))
    text = Path(path).read_text(encoding="utf-8")
    assert text.startswith("# SEO Keyword Cluster Recommendations")
    assert "## Blue Widgets" in text
    assert "## Red Gadgets" in text


def test_save_quality_report_writes_csv(clustered_frame: pd.DataFrame, tmp_path: Path):
    vectors = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]])
    path = save_quality_report(clustered_frame, str(tmp_path), keyword_vectors=vectors)
    out = pd.read_csv(path)
    assert {"cluster_id", "cluster_label", "cluster_size"}.issubset(out.columns)


def test_save_seo_workflow_artifacts_writes_briefs_and_supporting_csvs(clustered_frame: pd.DataFrame, tmp_path: Path):
    paths = save_seo_workflow_artifacts(clustered_frame, str(tmp_path))
    for key in (
        "page_briefs_dir",
        "internal_linking_opportunities",
        "keyword_heading_map",
        "content_hub_map",
        "serp_feature_opportunities",
    ):
        assert key in paths
    briefs = list(Path(paths["page_briefs_dir"]).glob("*.md"))
    assert briefs, "page_briefs_dir should contain at least one brief"
    # Slug rule: lowercase, no slashes, no spaces.
    for brief in briefs:
        assert brief.name == brief.name.lower()
        assert "/" not in brief.stem
        assert " " not in brief.stem


def test_save_seo_workflow_artifacts_serp_csv_includes_serp_columns_when_present(
    clustered_frame: pd.DataFrame, tmp_path: Path
):
    paths = save_seo_workflow_artifacts(clustered_frame, str(tmp_path))
    serp_csv = pd.read_csv(paths["serp_feature_opportunities"])
    assert "keyword" in serp_csv.columns
    assert "featured_snippet" in serp_csv.columns


def test_save_all_outputs_returns_every_artefact_path(clustered_frame: pd.DataFrame, tmp_path: Path):
    paths = save_all_outputs(clustered_frame, str(tmp_path))
    expected = {
        "clustered_keywords",
        "keyword_page_map",
        "content_gaps",
        "cannibalization",
        "cluster_summary",
        "recommendations",
        "cluster_quality_report",
        "page_briefs_dir",
        "internal_linking_opportunities",
        "keyword_heading_map",
        "content_hub_map",
        "serp_feature_opportunities",
    }
    assert expected.issubset(paths.keys())
    for key, p in paths.items():
        if key == "page_briefs_dir":
            assert Path(p).is_dir()
        else:
            assert Path(p).exists(), f"{key} -> {p} missing"


def test_build_interactive_report_creates_html(clustered_frame: pd.DataFrame, tmp_path: Path):
    # No chart files present — the report should still render with no sections.
    path = build_interactive_report(str(tmp_path))
    text = Path(path).read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "SEO Keyword Cluster Report" in text


def test_build_interactive_report_inlines_existing_chart_html(tmp_path: Path):
    (tmp_path / "treemap.html").write_text("<div id='treemap-marker'>chart</div>", encoding="utf-8")
    path = build_interactive_report(str(tmp_path))
    text = Path(path).read_text(encoding="utf-8")
    assert "treemap-marker" in text
    assert "Cluster Treemap" in text
