"""Command-line entry point for keyword-cluster."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product

import pandas as pd

from .clustering import ClusteringConfig
from .integrations import enrich_pages_dataframe, normalize_connector_csv
from .labeling import LabelingConfig
from .pipeline import PageMappingConfig, PipelineConfig, run_keyword_clustering
from .preprocessing import IntentConfig, set_local_intent_tokens
from .scoring import OpportunityConfig, SimilarityConfig
from .vectorization import EmbeddingConfig


def _add_common_run_args(run: argparse.ArgumentParser) -> None:
    run.add_argument("--keywords", required=True, help="CSV with 'keyword' column.")
    run.add_argument("--pages", default=None, help="CSV with pages. Requires url,page_name columns.")
    run.add_argument("--topics", default=None, help="CSV with topic column.")
    run.add_argument("--serp-file", default=None, help="Optional SERP CSV (keyword,position,url,title).")
    run.add_argument("--output", default="outputs", help="Output directory.")
    run.add_argument("--run-history", action="store_true", default=False)
    run.add_argument("--run-output-root", default="outputs/runs")

    run.add_argument("--method", default="kmeans", choices=["kmeans", "agglomerative", "hdbscan", "graph"])
    run.add_argument("--clusters", type=int, default=8)
    run.add_argument("--auto-k", default="none", choices=["none", "silhouette"])
    run.add_argument("--k-min", type=int, default=4)
    run.add_argument("--k-max", type=int, default=40)
    run.add_argument("--use-minibatch", action="store_true")
    run.add_argument("--agglomerative-linkage", default="ward")
    run.add_argument("--distance-metric", default="euclidean")
    run.add_argument("--min-cluster-size", type=int, default=5)
    run.add_argument("--min-samples", type=int, default=2)
    run.add_argument("--cluster-selection-method", default="eom")
    run.add_argument("--hdbscan-metric", default="euclidean")
    run.add_argument("--graph-k", type=int, default=10)
    run.add_argument("--graph-min-similarity", type=float, default=0.55)
    run.add_argument("--community-algorithm", choices=["louvain", "leiden"], default="louvain")
    run.add_argument("--graph-use-ann", action="store_true", default=False)
    run.add_argument("--graph-ann-ef", type=int, default=100)

    run.add_argument("--embedding-model", dest="embedding_model", default="tfidf")
    run.add_argument("--embedding", dest="embedding_model", help="Alias for --embedding-model.")
    run.add_argument("--embedding-batch-size", type=int, default=64)
    run.add_argument("--embedding-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    # Default matches EmbeddingConfig.normalize_embeddings=True. Users can disable with --no-normalize-embeddings.
    run.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--embedding-cache", default=".cache/embeddings")
    run.add_argument("--embedding-chunk-size", type=int, default=0)
    run.add_argument("--tfidf-svd-components", type=int, default=0)
    run.add_argument("--embedding-text-mode", default="keyword", choices=["keyword", "expanded"])
    run.add_argument("--preprocess", default="stem", choices=["none", "light", "stem", "lemmatize"])

    run.add_argument("--similarity", default="tfidf", choices=["semantic", "tfidf", "hybrid"])
    run.add_argument("--semantic-weight", type=float, default=0.65)
    run.add_argument("--tfidf-weight", type=float, default=0.35)
    run.add_argument("--serp-weight", type=float, default=0.0)

    run.add_argument("--gap-threshold", type=float, default=0.25)
    run.add_argument("--gap-threshold-mode", default="fixed", choices=["fixed", "percentile", "adaptive"])
    run.add_argument("--intent-mode", default="rules", choices=["rules", "serp", "embedding", "manual"])
    run.add_argument("--reduction", default="pca", choices=["pca", "umap", "tsne"])
    run.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (set 0.0 for BERTopic-style tight clusters).",
    )
    run.add_argument("--brand", nargs="*", default=[])
    run.add_argument(
        "--local-intent-tokens",
        nargs="*",
        default=None,
        help=(
            "Override the local-intent gazetteer (defaults to AU cities). "
            "Pass region-specific tokens (e.g. --local-intent-tokens near nearby london manchester edinburgh) "
            "to localise the rule-based intent classifier."
        ),
    )
    run.add_argument(
        "--opportunity-profile", default="balanced", choices=["balanced", "quick-wins", "growth", "commercial"]
    )
    run.add_argument("--labeling", default="tfidf", choices=["tfidf", "c-tfidf", "centroid", "mmr", "keybert"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="keyword-cluster")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full traceback on error.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Cluster keywords and generate outputs.")
    _add_common_run_args(run)

    compare = sub.add_parser("compare", help="Compare two historical runs.")
    compare.add_argument("--run-a", required=True)
    compare.add_argument("--run-b", required=True)
    compare.add_argument("--output", default="outputs/compare")

    tune = sub.add_parser("tune", help="Auto-parameter tuning with bounded grid search.")
    _add_common_run_args(tune)
    tune.add_argument("--methods", nargs="+", default=["kmeans", "agglomerative", "hdbscan", "graph"])
    tune.add_argument("--clusters-grid", nargs="+", type=int, default=[6, 8, 10])
    tune.add_argument("--embedding-models", nargs="+", default=["tfidf", "all-MiniLM-L6-v2"])
    tune.add_argument("--reductions", nargs="+", default=["pca", "umap"])
    tune.add_argument("--semantic-weights", nargs="+", type=float, default=[0.45, 0.55, 0.65])
    tune.add_argument("--tfidf-weights", nargs="+", type=float, default=[0.55, 0.45, 0.35])
    tune.add_argument("--hdbscan-min-cluster-sizes", nargs="+", type=int, default=[5, 10])
    tune.add_argument("--hdbscan-min-samples-grid", nargs="+", type=int, default=[1, 2, 4])

    for source in ("gsc", "ga4", "ahrefs", "semrush", "screamingfrog", "sitebulb"):
        imp = sub.add_parser(f"import-{source}", help=f"Normalize {source.upper()} CSV.")
        imp.add_argument("--file", required=True)
        imp.add_argument("--output", default=f"outputs/{source}_normalized.csv")

    crawl = sub.add_parser("crawl", help="Crawl/enrich pages CSV with title/meta/headings/body excerpt.")
    crawl.add_argument("--pages", required=True, help="Pages CSV with at least a url column.")
    crawl.add_argument("--output", default="outputs/enriched_pages.csv")
    crawl.add_argument("--timeout", type=int, default=20)

    enrich = sub.add_parser("enrich-pages", help="Alias for crawl.")
    enrich.add_argument("--pages", required=True, help="Pages CSV with at least a url column.")
    enrich.add_argument("--output", default="outputs/enriched_pages.csv")
    enrich.add_argument("--timeout", type=int, default=20)
    return parser


def _load_serp(df: pd.DataFrame, serp_file: str | None) -> pd.DataFrame:
    if not serp_file:
        return df
    serp = pd.read_csv(serp_file)
    req = {"keyword", "url"}
    if not req.issubset(set(serp.columns)):
        raise ValueError("SERP CSV must include at least columns: keyword,url")
    serp_urls = (
        serp.dropna(subset=["keyword", "url"])
        .groupby("keyword")["url"]
        .apply(lambda s: "|".join(dict.fromkeys(s.astype(str).tolist())))
        .reset_index(name="serp_urls")
    )
    return df.merge(serp_urls, on="keyword", how="left")


def _config_from_args(
    args: argparse.Namespace,
    method_override: str | None = None,
    cluster_override: int | None = None,
    sem_w: float | None = None,
    tfidf_w: float | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        embedding=EmbeddingConfig(
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
            device=args.embedding_device,
            normalize_embeddings=args.normalize_embeddings,
            cache_dir=args.embedding_cache,
            chunk_size=args.embedding_chunk_size,
            tfidf_svd_components=args.tfidf_svd_components,
            text_mode=args.embedding_text_mode,
            # The user picks a single preprocess mode; honour it on the embedding side too.
            # Earlier code clamped to none|light, which silently downgraded stem/lemmatize when
            # combined with a transformer model.
            preprocess_mode=args.preprocess,
        ),
        similarity=SimilarityConfig(
            mode=args.similarity,
            semantic_weight=args.semantic_weight if sem_w is None else sem_w,
            tfidf_weight=args.tfidf_weight if tfidf_w is None else tfidf_w,
            serp_weight=args.serp_weight,
        ),
        clustering=ClusteringConfig(
            method=args.method if method_override is None else method_override,
            n_clusters=args.clusters if cluster_override is None else cluster_override,
            auto_k=None if args.auto_k == "none" else args.auto_k,
            k_min=args.k_min,
            k_max=args.k_max,
            use_minibatch=args.use_minibatch,
            agglomerative_metric=args.distance_metric,
            agglomerative_linkage=args.agglomerative_linkage,
            hdbscan_min_cluster_size=args.min_cluster_size,
            hdbscan_min_samples=args.min_samples,
            hdbscan_cluster_selection_method=args.cluster_selection_method,
            hdbscan_metric=args.hdbscan_metric,
            graph_k=args.graph_k,
            graph_min_similarity=args.graph_min_similarity,
            community_algorithm=args.community_algorithm,
            graph_use_ann=args.graph_use_ann,
            graph_ann_ef=args.graph_ann_ef,
        ),
        intent=IntentConfig(mode=args.intent_mode),
        page_mapping=PageMappingConfig(gap_threshold=args.gap_threshold, gap_threshold_mode=args.gap_threshold_mode),
        opportunity=OpportunityConfig(profile=args.opportunity_profile),
        labeling=LabelingConfig(strategy=args.labeling),
        preprocess_mode=args.preprocess,
        reduction=args.reduction,
        umap_min_dist=getattr(args, "umap_min_dist", 0.1),
        brand_terms=args.brand,
        run_history=args.run_history,
        run_output_root=args.run_output_root,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    from .export import build_interactive_report, save_all_outputs
    from .visualization import save_all_charts

    if getattr(args, "local_intent_tokens", None):
        set_local_intent_tokens(args.local_intent_tokens)
    df = _load_serp(pd.read_csv(args.keywords), args.serp_file)
    pages_df = pd.read_csv(args.pages) if args.pages else None
    topics_df = pd.read_csv(args.topics) if args.topics else None
    config = _config_from_args(args)
    result = run_keyword_clustering(df, pages_df, topics_df, config)
    saved = save_all_outputs(result.df, args.output, keyword_vectors=result.keyword_vectors, run_dir=result.run_dir)
    save_all_charts(
        result.df,
        result.keyword_vectors,
        result.coords,
        args.output,
        topic_vectors=result.topic_vectors,
        topic_labels=topics_df["topic"].dropna().astype(str).tolist()
        if topics_df is not None and "topic" in topics_df.columns
        else None,
    )
    if result.run_dir:
        save_all_outputs(result.df, result.run_dir, keyword_vectors=result.keyword_vectors, run_dir=result.run_dir)
        save_all_charts(
            result.df,
            result.keyword_vectors,
            result.coords,
            os.path.join(result.run_dir, "charts"),
            topic_vectors=result.topic_vectors,
            topic_labels=topics_df["topic"].dropna().astype(str).tolist()
            if topics_df is not None and "topic" in topics_df.columns
            else None,
        )
    report_path = build_interactive_report(args.output)
    print("Done. Summary:")
    print(f"  Keywords: {result.metrics['n_keywords']}")
    print(f"  Clusters: {result.metrics['n_clusters']}")
    for label, path in saved.items():
        print(f"  {label}: {path}")
    if result.run_dir:
        print(f"  run_dir: {result.run_dir}")
    print(f"  interactive_report: {report_path}")


def run_compare(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    a = pd.read_csv(os.path.join(args.run_a, "clustered_keywords.csv"))
    b = pd.read_csv(os.path.join(args.run_b, "clustered_keywords.csv"))
    merged = a[["keyword", "cluster_id", "opportunity_score", "recommended_url"]].merge(
        b[["keyword", "cluster_id", "opportunity_score", "recommended_url"]],
        on="keyword",
        suffixes=("_a", "_b"),
        how="outer",
    )
    summary = {
        "clusters_a": int(a["cluster_id"].nunique()),
        "clusters_b": int(b["cluster_id"].nunique()),
        "keywords_moved": int((merged["cluster_id_a"] != merged["cluster_id_b"]).fillna(True).sum()),
        "mapping_changes": int((merged["recommended_url_a"] != merged["recommended_url_b"]).fillna(True).sum()),
        "avg_opportunity_delta": float(
            (merged["opportunity_score_b"].fillna(0) - merged["opportunity_score_a"].fillna(0)).mean()
        ),
    }
    merged.to_csv(os.path.join(args.output, "compare_keywords.csv"), index=False)
    with open(os.path.join(args.output, "compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def run_tune(args: argparse.Namespace) -> None:
    from .scoring import build_cluster_quality_report

    os.makedirs(args.output, exist_ok=True)
    df = _load_serp(pd.read_csv(args.keywords), args.serp_file)
    pages_df = pd.read_csv(args.pages) if args.pages else None
    topics_df = pd.read_csv(args.topics) if args.topics else None
    rows: list[dict[str, object]] = []
    best = None
    best_score = -10.0
    for method, clusters, emb_model, reduction, sem_w, tfidf_w, h_min_cluster, h_min_samples in product(
        args.methods,
        args.clusters_grid,
        args.embedding_models,
        args.reductions,
        args.semantic_weights,
        args.tfidf_weights,
        args.hdbscan_min_cluster_sizes,
        args.hdbscan_min_samples_grid,
    ):
        if args.similarity != "hybrid":
            sem_w, tfidf_w = args.semantic_weight, args.tfidf_weight
        # Copy args so the grid loop doesn't leak per-cell overrides back onto the caller's
        # Namespace. Earlier code mutated args in place, leaving its state at the last cell.
        cell_args = argparse.Namespace(**vars(args))
        cell_args.embedding_model = emb_model
        cell_args.reduction = reduction
        cell_args.min_cluster_size = h_min_cluster
        cell_args.min_samples = h_min_samples
        cfg = _config_from_args(
            cell_args, method_override=method, cluster_override=clusters, sem_w=sem_w, tfidf_w=tfidf_w
        )
        try:
            result = run_keyword_clustering(df, pages_df, topics_df, cfg)
            q = build_cluster_quality_report(result.df, result.keyword_vectors)
            sil = float(pd.to_numeric(q["silhouette_score"], errors="coerce").fillna(0).mean()) if not q.empty else -1
            intra = (
                float(pd.to_numeric(q["avg_intra_cluster_similarity"], errors="coerce").fillna(0).mean())
                if not q.empty
                else 0
            )
            weak = float(pd.to_numeric(q["weak_match_rate"], errors="coerce").fillna(1).mean()) if not q.empty else 1
            score = float(sil * 0.6 + intra * 0.4 - weak * 0.2)
            row = {
                "method": method,
                "clusters": clusters,
                "embedding_model": emb_model,
                "reduction": reduction,
                "hdbscan_min_cluster_size": h_min_cluster,
                "hdbscan_min_samples": h_min_samples,
                "semantic_weight": sem_w,
                "tfidf_weight": tfidf_w,
                "quality_score": score,
                "silhouette_score": sil,
                "avg_intra_cluster_similarity": intra,
                "avg_weak_match_rate": weak,
                "n_clusters": result.metrics["n_clusters"],
            }
        except Exception as exc:
            row = {
                "method": method,
                "clusters": clusters,
                "embedding_model": emb_model,
                "reduction": reduction,
                "hdbscan_min_cluster_size": h_min_cluster,
                "hdbscan_min_samples": h_min_samples,
                "semantic_weight": sem_w,
                "tfidf_weight": tfidf_w,
                "quality_score": -999.0,
                "error": str(exc),
            }
        rows.append(row)
        if float(row.get("quality_score", -999.0)) > best_score:
            best = row
            best_score = float(row["quality_score"])
    tuning_df = pd.DataFrame(rows).sort_values("quality_score", ascending=False)
    tuning_df.to_csv(os.path.join(args.output, "tuning_results.csv"), index=False)
    with open(os.path.join(args.output, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best or {}, f, indent=2)

    # Quality charts: a real Plotly bar+scatter view of the grid, with a fallback
    # to a plain HTML table if Plotly isn't available.
    quality_charts_path = os.path.join(args.output, "quality_charts.html")
    try:
        import plotly.express as px

        head_df = tuning_df.head(200).copy()
        head_df["config"] = head_df.apply(
            lambda r: f"{r['method']}/k={r['clusters']}/{r['embedding_model']}/{r['reduction']}",
            axis=1,
        )
        bar_fig = px.bar(
            head_df,
            x="config",
            y="quality_score",
            color="method",
            hover_data=["silhouette_score", "avg_intra_cluster_similarity", "avg_weak_match_rate", "n_clusters"],
            title="Tuning grid — quality score by configuration (top 200)",
        )
        bar_fig.update_layout(xaxis_tickangle=-60, height=600, margin=dict(b=200))
        scatter_fig = px.scatter(
            head_df,
            x="silhouette_score",
            y="avg_intra_cluster_similarity",
            color="method",
            size="quality_score",
            hover_data=["config", "n_clusters"],
            title="Quality landscape — silhouette vs. intra-cluster similarity",
        )
        scatter_fig.update_layout(height=500)
        with open(quality_charts_path, "w", encoding="utf-8") as f:
            f.write(
                "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
                "<title>Tuning Quality Charts</title>"
                "<style>body{font-family:sans-serif;padding:20px}h1{border-bottom:2px solid #333}</style>"
                "</head><body><h1>Tuning Quality Charts</h1>"
            )
            f.write(bar_fig.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write(scatter_fig.to_html(full_html=False, include_plotlyjs=False))
            f.write("</body></html>")
    except ImportError:
        tuning_df.head(200).to_html(quality_charts_path, index=False)
    print(f"Saved tuning report to {args.output}")


def run_import(source: str, args: argparse.Namespace) -> None:
    df = pd.read_csv(args.file)
    norm = normalize_connector_csv(source, df)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    norm.to_csv(args.output, index=False)
    print(f"Saved normalized {source} to {args.output}")


def run_crawl(args: argparse.Namespace) -> None:
    pages_df = pd.read_csv(args.pages)
    enriched = enrich_pages_dataframe(pages_df, timeout=args.timeout)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    enriched.to_csv(args.output, index=False)
    print(f"Saved enriched pages to {args.output} ({len(enriched)} rows)")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        cmd = args.command
        if cmd == "run":
            run_pipeline(args)
        elif cmd == "compare":
            run_compare(args)
        elif cmd == "tune":
            run_tune(args)
        elif cmd in {"crawl", "enrich-pages"}:
            run_crawl(args)
        elif cmd and cmd.startswith("import-"):
            run_import(cmd.replace("import-", ""), args)
        else:
            parser.print_help()
    except FileNotFoundError as exc:
        if getattr(args, "verbose", False):
            raise
        print(f"Error: file not found - {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        if getattr(args, "verbose", False):
            raise
        print(f"Error: invalid input - {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        if getattr(args, "verbose", False):
            raise
        print(f"Error: {exc} (re-run with --verbose for full traceback)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
