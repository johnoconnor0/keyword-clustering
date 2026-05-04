"""CSV and report export helpers."""

from __future__ import annotations

import os

import pandas as pd

from .scoring import build_cluster_quality_report, detect_cannibalization, detect_content_gaps


def _top_keywords(x: pd.Series) -> str:
    return ", ".join(x.head(5).tolist())


def _safe_mode(series: pd.Series, default: str) -> str:
    """`series.mode().iloc[0]` raises `IndexError` when every value is NaN.
    Return a default in that case so external callers don't crash on edge data.
    """
    if series is None:
        return default
    modes = series.dropna().mode()
    if modes.empty:
        return default
    return str(modes.iloc[0])


def save_clustered_keywords(df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clustered_keywords.csv")
    df.to_csv(path, index=False)
    return path


def save_keyword_page_map(df: pd.DataFrame, output_dir: str) -> str:
    cols = [
        c
        for c in [
            "keyword",
            "recommended_page",
            "recommended_url",
            "page_similarity_score",
            "match_confidence",
            "cluster_label",
            "primary_intent",
        ]
        if c in df.columns
    ]
    path = os.path.join(output_dir, "keyword_page_map.csv")
    df[cols].to_csv(path, index=False)
    return path


def save_content_gaps(df: pd.DataFrame, output_dir: str) -> str:
    gaps = detect_content_gaps(df)
    path = os.path.join(output_dir, "content_gap_report.csv")
    gaps.to_csv(path, index=False)
    return path


def save_cannibalization(df: pd.DataFrame, output_dir: str) -> str:
    cannibal = detect_cannibalization(df)
    path = os.path.join(output_dir, "cannibalization_report.csv")
    cannibal.to_csv(path, index=False)
    return path


def save_cluster_summary(df: pd.DataFrame, output_dir: str) -> str:
    agg: dict = {"keyword": ["count", _top_keywords]}
    if "search_volume" in df.columns:
        agg["search_volume"] = ["sum", "mean"]
    if "keyword_difficulty" in df.columns:
        agg["keyword_difficulty"] = "mean"
    if "opportunity_score" in df.columns:
        agg["opportunity_score"] = "mean"

    summary = df.groupby(["cluster_id", "cluster_label"]).agg(agg)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.rename(columns={"keyword__top_keywords": "top_keywords"})
    path = os.path.join(output_dir, "cluster_summary.csv")
    summary.reset_index().to_csv(path, index=False)
    return path


def save_recommendations(df: pd.DataFrame, output_dir: str) -> str:
    lines = ["# SEO Keyword Cluster Recommendations\n"]
    for _, group in df.groupby("cluster_id"):
        label = group["cluster_label"].iloc[0] if "cluster_label" in group.columns else ""
        intent = _safe_mode(group.get("primary_intent"), "mixed")
        page = _safe_mode(group.get("recommended_page"), "unknown")
        lines += [
            f"\n## {label}",
            f"- **Recommended page:** {page}",
            f"- **Primary intent:** {intent}",
            f"- **Top keywords:** {', '.join(group['keyword'].head(5).tolist())}",
            f"- **Opportunity rationale:** {group['opportunity_reason'].iloc[0] if 'opportunity_reason' in group.columns else 'n/a'}",
            "",
        ]
    path = os.path.join(output_dir, "recommendations.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def save_quality_report(df: pd.DataFrame, output_dir: str, keyword_vectors: object | None = None) -> str:
    report = build_cluster_quality_report(df, keyword_vectors)
    path = os.path.join(output_dir, "cluster_quality_report.csv")
    report.to_csv(path, index=False)
    return path


def save_seo_workflow_artifacts(df: pd.DataFrame, output_dir: str) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    briefs_dir = os.path.join(output_dir, "page_briefs")
    os.makedirs(briefs_dir, exist_ok=True)
    for _, group in df.groupby("cluster_id"):
        label = str(group["cluster_label"].iloc[0]).replace("/", "-").replace(" ", "-").lower()
        fname = os.path.join(briefs_dir, f"{label[:80]}.md")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"# {group['cluster_label'].iloc[0]}\n\n")
            f.write("## Recommended page\n")
            f.write(f"- {_safe_mode(group.get('recommended_page'), 'Unknown')}\n\n")
            f.write("## Target keywords\n")
            for kw in group["keyword"].head(20):
                f.write(f"- {kw}\n")
    linking = (
        df[["keyword", "cluster_label", "recommended_page"]]
        if all(c in df.columns for c in ["keyword", "cluster_label", "recommended_page"])
        else pd.DataFrame()
    )
    linking_path = os.path.join(output_dir, "internal_linking_opportunities.csv")
    linking.to_csv(linking_path, index=False)
    heading_path = os.path.join(output_dir, "keyword_to_heading_map.csv")
    if "recommended_page" in df.columns:
        pd.DataFrame({"keyword": df["keyword"], "suggested_h2": df["keyword"], "page": df["recommended_page"]}).to_csv(
            heading_path, index=False
        )
    else:
        pd.DataFrame({"keyword": df["keyword"], "suggested_h2": df["keyword"]}).to_csv(heading_path, index=False)
    hub_path = os.path.join(output_dir, "content_hub_map.html")
    pd.DataFrame({"cluster": df.get("cluster_label", pd.Series([""] * len(df))), "keyword": df["keyword"]}).head(
        2000
    ).to_html(hub_path, index=False)
    serp_path = os.path.join(output_dir, "serp_feature_opportunities.csv")
    serp_cols = [
        c
        for c in ["featured_snippet", "local_pack", "people_also_ask", "image_pack", "video_result"]
        if c in df.columns
    ]
    if serp_cols:
        df[["keyword"] + serp_cols].to_csv(serp_path, index=False)
    else:
        pd.DataFrame({"keyword": df["keyword"]}).to_csv(serp_path, index=False)
    return {
        "page_briefs_dir": briefs_dir,
        "internal_linking_opportunities": linking_path,
        "keyword_heading_map": heading_path,
        "content_hub_map": hub_path,
        "serp_feature_opportunities": serp_path,
    }


def build_interactive_report(output_dir: str) -> str:
    charts = [
        ("cluster_map_3d.html", "3D Cluster Map"),
        ("cluster_map_2d.html", "2D Topic Map"),
        ("treemap.html", "Cluster Treemap"),
        ("heatmap.html", "Similarity Heatmap"),
        ("opportunity_matrix.html", "Opportunity Matrix"),
        ("sankey.html", "Page to Cluster Sankey"),
        ("network_graph.html", "Keyword Network"),
    ]
    sections = []
    for filename, title in charts:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                sections.append(f"<h2>{title}</h2><div>{f.read()}</div>")
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>SEO Keyword Cluster Report</title>
<style>body{{font-family:sans-serif;padding:20px}}h1{{border-bottom:2px solid #333}}h2{{margin-top:40px}}</style></head>
<body><h1>SEO Keyword Cluster Report</h1>{"".join(sections)}</body></html>"""
    path = os.path.join(output_dir, "interactive_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def save_all_outputs(
    df: pd.DataFrame, output_dir: str, keyword_vectors: object | None = None, run_dir: str | None = None
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    out = {
        "clustered_keywords": save_clustered_keywords(df, output_dir),
        "keyword_page_map": save_keyword_page_map(df, output_dir),
        "content_gaps": save_content_gaps(df, output_dir),
        "cannibalization": save_cannibalization(df, output_dir),
        "cluster_summary": save_cluster_summary(df, output_dir),
        "recommendations": save_recommendations(df, output_dir),
        "cluster_quality_report": save_quality_report(df, output_dir, keyword_vectors),
    }
    out.update(save_seo_workflow_artifacts(df, output_dir))
    if run_dir:
        out["run_dir"] = run_dir
    return out
