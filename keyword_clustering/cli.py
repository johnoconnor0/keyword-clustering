"""Command-line entry point for keyword-cluster."""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyword-cluster",
        description="SEO keyword clustering, page mapping, and content-gap analysis.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Cluster keywords and generate all outputs.")
    run.add_argument("--keywords", required=True, help="CSV with 'keyword' column (plus optional metrics).")
    run.add_argument("--pages", default=None, help="CSV with 'url' and 'page_name' columns.")
    run.add_argument("--topics", default=None, help="CSV with 'topic' column.")
    run.add_argument("--clusters", type=int, default=8, help="Number of clusters (default: 8).")
    run.add_argument(
        "--method", default="kmeans",
        choices=["kmeans", "agglomerative", "hdbscan"],
        help="Clustering method (default: kmeans).",
    )
    run.add_argument(
        "--embedding", default="tfidf",
        help="Embedding: 'tfidf' or a Sentence Transformers model name (default: tfidf).",
    )
    run.add_argument(
        "--reduction", default="pca",
        choices=["pca", "umap", "tsne"],
        help="Dimensionality reduction for plots (default: pca).",
    )
    run.add_argument("--output", default="outputs", help="Output directory (default: outputs/).")
    run.add_argument("--brand", nargs="*", default=[], help="Brand terms for branded/non-branded flagging.")
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    from .preprocessing import enrich_keywords, load_keywords, load_pages, load_topics
    from .vectorization import (
        build_tfidf_vectorizer,
        compute_similarity_matrix,
        vectorize_keywords_st,
        vectorize_keywords_tfidf,
    )
    from .clustering import cluster_keywords, reduce_dimensions
    from .labeling import apply_cluster_labels, generate_cluster_labels
    from .scoring import map_keywords_to_pages, compute_opportunity_score, add_notes
    from .visualization import save_all_charts
    from .export import build_interactive_report, save_all_outputs

    print(f"Loading keywords from {args.keywords}...")
    df = load_keywords(args.keywords)
    pages = load_pages(args.pages)
    topics = load_topics(args.topics)

    print(f"  {len(df)} keywords loaded.")
    df = enrich_keywords(df, brand_terms=args.brand)
    df = df.rename(columns={"intent": "primary_intent"})

    page_names = [p["page_name"] for p in pages]
    page_urls = [p["url"] for p in pages]
    keywords = df["keyword"].tolist()

    print(f"Vectorizing with method: {args.embedding}...")
    if args.embedding == "tfidf":
        corpus = keywords + page_names + topics
        vectorizer, _ = build_tfidf_vectorizer(corpus)
        kw_vectors = vectorize_keywords_tfidf(keywords, vectorizer)
        page_vectors = vectorize_keywords_tfidf(page_names, vectorizer) if page_names else None
        topic_vectors = vectorize_keywords_tfidf(topics, vectorizer) if topics else None
    else:
        all_texts = keywords + page_names + topics
        all_vecs = vectorize_keywords_st(all_texts, args.embedding)
        n_kw = len(keywords)
        n_page = len(page_names)
        kw_vectors = all_vecs[:n_kw]
        page_vectors = all_vecs[n_kw:n_kw + n_page] if page_names else None
        topic_vectors = all_vecs[n_kw + n_page:] if topics else None

    print(f"Clustering ({args.method}, n={args.clusters})...")
    n_clusters = min(args.clusters, len(df))
    labels = cluster_keywords(kw_vectors, method=args.method, n_clusters=n_clusters)
    df["cluster_id"] = labels

    print("Generating cluster labels...")
    cluster_label_map = generate_cluster_labels(df)
    df = apply_cluster_labels(df, cluster_label_map)

    if page_vectors is not None and page_names:
        print("Mapping keywords to pages...")
        df = map_keywords_to_pages(df, kw_vectors, page_vectors, page_names, page_urls)

    if topics and topic_vectors is not None:
        sim = compute_similarity_matrix(kw_vectors, topic_vectors)
        best_topic_idx = sim.argmax(axis=1)
        df["primary_topic"] = [topics[i] for i in best_topic_idx]
        df["topic_similarity_score"] = sim.max(axis=1).round(4).tolist()

    print("Computing opportunity scores...")
    df = compute_opportunity_score(df)
    df = add_notes(df)

    print(f"Reducing to 3D with {args.reduction}...")
    coords = reduce_dimensions(kw_vectors, n_components=3, method=args.reduction)

    print(f"Saving outputs to {args.output}/...")
    saved = save_all_outputs(df, args.output)
    for label, path in saved.items():
        print(f"  {label}: {path}")

    print("Generating charts...")
    save_all_charts(
        df, kw_vectors, coords, args.output,
        topic_vectors=topic_vectors,
        topic_labels=topics if topics else None,
    )
    report_path = build_interactive_report(args.output)
    print(f"  interactive_report: {report_path}")

    print("\nDone. Summary:")
    print(f"  Keywords: {len(df)}")
    print(f"  Clusters: {df['cluster_id'].nunique()}")
    if "content_gap" in df.columns:
        print(f"  Content gaps: {df['content_gap'].sum()}")
    if topic_vectors is not None:
        print(f"  heatmap.html: generated")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        try:
            run_pipeline(args)
        except FileNotFoundError as exc:
            print(f"Error: file not found — {exc}", file=sys.stderr)
            sys.exit(1)
        except ValueError as exc:
            print(f"Error: invalid input — {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
