"""Interactive Plotly chart functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

from .vectorization import to_dense


def _safe_size_col(df: pd.DataFrame) -> pd.Series | None:
    if "search_volume" in df.columns:
        vol = df["search_volume"].fillna(0)
        if vol.max() > 0:
            return vol
    return None


def plot_3d_clusters(df: pd.DataFrame, coords: np.ndarray) -> go.Figure:
    """3D scatter plot coloured by cluster, sized by search_volume."""
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["z"] = coords[:, 2] if coords.shape[1] > 2 else np.zeros(len(df))

    hover = [c for c in ["keyword", "recommended_page", "primary_intent", "opportunity_score",
                          "search_volume", "keyword_difficulty", "cluster_label"] if c in plot_df.columns]
    size_col = _safe_size_col(plot_df)

    fig = px.scatter_3d(
        plot_df,
        x="x", y="y", z="z",
        color="cluster_label" if "cluster_label" in plot_df.columns else None,
        size=size_col,
        size_max=20,
        hover_data=hover,
        title="3D Keyword Cluster Map",
        labels={"cluster_label": "Cluster"},
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig


def plot_2d_clusters(df: pd.DataFrame, coords: np.ndarray) -> go.Figure:
    """2D scatter plot coloured by cluster."""
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    hover = [c for c in ["keyword", "recommended_page", "primary_intent", "opportunity_score",
                          "search_volume", "keyword_difficulty"] if c in plot_df.columns]
    size_col = _safe_size_col(plot_df)

    fig = px.scatter(
        plot_df,
        x="x", y="y",
        color="cluster_label" if "cluster_label" in plot_df.columns else None,
        size=size_col,
        size_max=20,
        hover_data=hover,
        title="2D Keyword Topic Map",
        labels={"cluster_label": "Cluster"},
    )
    fig.update_layout(xaxis_title="Component 1", yaxis_title="Component 2")
    return fig


def plot_treemap(df: pd.DataFrame) -> go.Figure:
    """Cluster treemap sized by keyword count, coloured by total search volume."""
    if "cluster_label" not in df.columns:
        return go.Figure()

    agg = df.groupby("cluster_label").agg(
        keyword_count=("keyword", "count"),
        total_volume=("search_volume", "sum") if "search_volume" in df.columns else ("keyword", "count"),
    ).reset_index()

    fig = px.treemap(
        agg,
        path=["cluster_label"],
        values="keyword_count",
        color="total_volume",
        color_continuous_scale="Blues",
        title="Cluster Treemap — Size: keyword count, Colour: total search volume",
        hover_data=["keyword_count", "total_volume"],
    )
    return fig


def plot_similarity_heatmap(vectors: object, labels: list[str]) -> go.Figure:
    """Cosine similarity heatmap between all items (topics or cluster centroids)."""
    dense = to_dense(vectors)
    sim = cosine_similarity(dense)
    fig = px.imshow(
        sim,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        zmin=0, zmax=1,
        title="Keyword Similarity Heatmap",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_sankey(df: pd.DataFrame) -> go.Figure:
    """Sankey diagram: page → cluster keyword flow."""
    if "recommended_page" not in df.columns or "cluster_label" not in df.columns:
        return go.Figure()

    pages = df["recommended_page"].unique().tolist()
    clusters = df["cluster_label"].unique().tolist()
    all_nodes = pages + clusters
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    links: dict[tuple, int] = {}
    for _, row in df.iterrows():
        key = (row["recommended_page"], row["cluster_label"])
        links[key] = links.get(key, 0) + 1

    source = [node_idx[k[0]] for k in links]
    target = [node_idx[k[1]] for k in links]
    value = list(links.values())

    fig = go.Figure(go.Sankey(
        node=dict(label=all_nodes, pad=15, thickness=20),
        link=dict(source=source, target=target, value=value),
    ))
    fig.update_layout(title="Page → Cluster Keyword Flow")
    return fig


def plot_opportunity_matrix(df: pd.DataFrame) -> go.Figure:
    """SERP opportunity scatter: x=difficulty, y=opportunity_score, size=volume."""
    cols_needed = {"opportunity_score", "keyword_difficulty"}
    if not cols_needed.issubset(df.columns):
        return go.Figure()

    plot_df = df.dropna(subset=["opportunity_score", "keyword_difficulty"]).copy()
    size_col = _safe_size_col(plot_df)
    hover = [c for c in ["keyword", "recommended_page", "primary_intent", "cluster_label",
                          "search_volume", "rank"] if c in plot_df.columns]

    fig = px.scatter(
        plot_df,
        x="keyword_difficulty",
        y="opportunity_score",
        color="cluster_label" if "cluster_label" in plot_df.columns else None,
        size=size_col,
        size_max=25,
        hover_data=hover,
        title="SERP Opportunity Matrix — x: Difficulty, y: Opportunity Score",
        labels={"keyword_difficulty": "Keyword Difficulty", "opportunity_score": "Opportunity Score"},
    )
    return fig


def plot_network_graph(df: pd.DataFrame, vectors: object, top_n_edges: int = 80) -> go.Figure:
    """Keyword network graph: nodes=keywords, edges=high cosine similarity pairs."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from sklearn.decomposition import PCA

    dense = to_dense(vectors)
    sim = cos_sim(dense)
    coords = PCA(n_components=2, random_state=42).fit_transform(dense)
    keywords = df["keyword"].tolist()
    cluster_labels = df["cluster_label"].tolist() if "cluster_label" in df.columns else [""] * len(keywords)

    np.fill_diagonal(sim, 0)
    flat = sim.flatten()
    threshold = np.sort(flat)[-min(top_n_edges * 2, len(flat))]

    edge_x, edge_y = [], []
    rows, cols = np.where(sim >= threshold)
    for r, c in zip(rows, cols):
        if r < c:
            edge_x += [coords[r, 0], coords[c, 0], None]
            edge_y += [coords[r, 1], coords[c, 1], None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.5, color="#aaa"), hoverinfo="none")

    hover_text = [f"{kw}<br>Cluster: {cl}" for kw, cl in zip(keywords, cluster_labels)]
    node_trace = go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers",
        marker=dict(size=8, color=df["cluster_id"].tolist() if "cluster_id" in df.columns else None,
                    colorscale="Viridis", showscale=False),
        text=hover_text, hoverinfo="text",
    )

    fig = go.Figure([edge_trace, node_trace],
                    layout=go.Layout(title="Keyword Similarity Network",
                                     showlegend=False,
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig


def save_all_charts(
    df: pd.DataFrame,
    vectors: object,
    coords: np.ndarray,
    output_dir: str,
    topic_vectors: object | None = None,
    topic_labels: list[str] | None = None,
) -> None:
    """Save all charts as HTML to output_dir (CDN plotly.js keeps files small).

    Pass topic_vectors and topic_labels to also generate heatmap.html.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    opts = dict(include_plotlyjs="cdn")
    plot_3d_clusters(df, coords).write_html(os.path.join(output_dir, "cluster_map_3d.html"), **opts)
    plot_2d_clusters(df, coords).write_html(os.path.join(output_dir, "cluster_map_2d.html"), **opts)
    plot_treemap(df).write_html(os.path.join(output_dir, "treemap.html"), **opts)
    plot_opportunity_matrix(df).write_html(os.path.join(output_dir, "opportunity_matrix.html"), **opts)
    plot_sankey(df).write_html(os.path.join(output_dir, "sankey.html"), **opts)
    plot_network_graph(df, vectors).write_html(os.path.join(output_dir, "network_graph.html"), **opts)

    if topic_vectors is not None and topic_labels:
        plot_similarity_heatmap(topic_vectors, topic_labels).write_html(
            os.path.join(output_dir, "heatmap.html"), **opts
        )
