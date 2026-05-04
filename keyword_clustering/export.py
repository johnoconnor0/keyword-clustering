"""CSV and report export helpers."""

from __future__ import annotations

import os

import pandas as pd

from .scoring import detect_cannibalization, detect_content_gaps


def _top_keywords(x: pd.Series) -> str:
    return ", ".join(x.head(5).tolist())


def save_clustered_keywords(df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "clustered_keywords.csv")
    df.to_csv(path, index=False)
    return path


def save_keyword_page_map(df: pd.DataFrame, output_dir: str) -> str:
    cols = [c for c in ["keyword", "recommended_page", "recommended_url",
                         "page_similarity_score", "cluster_label", "primary_intent"] if c in df.columns]
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

    group_cols = ["cluster_id", "cluster_label"] if "cluster_label" in df.columns else ["cluster_id"]
    summary = df.groupby(group_cols).agg(agg)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.rename(columns={"keyword__top_keywords": "top_keywords"})
    path = os.path.join(output_dir, "cluster_summary.csv")
    summary.reset_index().to_csv(path, index=False)
    return path


def save_recommendations(df: pd.DataFrame, output_dir: str) -> str:
    lines = ["# SEO Keyword Cluster Recommendations\n"]

    for cluster_id, group in df.groupby("cluster_id"):
        label = group["cluster_label"].iloc[0] if "cluster_label" in group.columns else f"Cluster {cluster_id}"
        intent = group["primary_intent"].mode().iloc[0] if "primary_intent" in group.columns else "mixed"
        page = group["recommended_page"].mode().iloc[0] if "recommended_page" in group.columns else "unknown"
        top_kws = ", ".join(group["keyword"].head(5).tolist())
        vol = int(group["search_volume"].sum()) if "search_volume" in group.columns else "N/A"
        opp = round(group["opportunity_score"].mean(), 3) if "opportunity_score" in group.columns else "N/A"

        lines += [
            f"\n## {label}",
            f"- **Recommended page:** {page}",
            f"- **Primary intent:** {intent}",
            f"- **Total search volume:** {vol}",
            f"- **Avg opportunity score:** {opp}",
            f"- **Top keywords:** {top_kws}",
            "",
            "**Recommended actions:**",
            f"- Ensure `{page}` is optimised for this keyword group.",
            "- Add supporting FAQ sections for informational long-tail variants." if intent == "informational" else "- Strengthen commercial CTAs and trust signals.",
            "- Review internal linking from related pages to concentrate authority.",
            "",
        ]

    path = os.path.join(output_dir, "recommendations.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def build_interactive_report(output_dir: str) -> str:
    """Bundle all saved HTML charts into a single interactive_report.html."""
    charts = [
        ("cluster_map_3d.html", "3D Cluster Map"),
        ("cluster_map_2d.html", "2D Topic Map"),
        ("treemap.html", "Cluster Treemap"),
        ("heatmap.html", "Similarity Heatmap"),
        ("opportunity_matrix.html", "Opportunity Matrix"),
        ("sankey.html", "Page → Cluster Sankey"),
        ("network_graph.html", "Keyword Network"),
    ]

    sections = []
    for filename, title in charts:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                content = f.read()
            sections.append(f"<h2>{title}</h2>\n<div>{content}</div>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>SEO Keyword Cluster Report</title>
<style>body{{font-family:sans-serif;padding:20px}} h1{{border-bottom:2px solid #333}} h2{{margin-top:40px}}</style>
</head>
<body>
<h1>SEO Keyword Cluster Report</h1>
{"".join(sections)}
</body></html>"""

    path = os.path.join(output_dir, "interactive_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def save_all_outputs(df: pd.DataFrame, output_dir: str) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    return {
        "clustered_keywords": save_clustered_keywords(df, output_dir),
        "keyword_page_map": save_keyword_page_map(df, output_dir),
        "content_gaps": save_content_gaps(df, output_dir),
        "cannibalization": save_cannibalization(df, output_dir),
        "cluster_summary": save_cluster_summary(df, output_dir),
        "recommendations": save_recommendations(df, output_dir),
    }
