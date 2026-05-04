# Output Files

All files are written to the `--output` directory.

## Primary reports

- `clustered_keywords.csv`: keyword-level master output.
- `keyword_page_map.csv`: keyword-to-page mapping with similarity and confidence bands.
- `content_gap_report.csv`: keywords below threshold (`fixed|percentile|adaptive` mode).
- `cannibalization_report.csv`: `ranking_cannibalization`, `mapping_conflict`, `intent_split`, `page_mismatch`, `consolidation_candidate`.
- `cluster_summary.csv`: per-cluster summary.
- `recommendations.md`: human-readable cluster recommendations.
- `cluster_quality_report.csv`: quality metrics per cluster.

## `match_confidence` bands

`keyword_page_map.csv` and `clustered_keywords.csv` carry a `match_confidence`
column derived from `page_similarity_score`. The four bands are:

| Band | Score range | Meaning |
|---|---|---|
| `poor_match` | `< 0.20` | The best-matching page is unrelated; treat as content gap candidate. |
| `weak_match` | `0.20 ≤ score < 0.40` | Loose topical match. Consider new page or significant rewrite. |
| `acceptable_match` | `0.40 ≤ score < 0.65` | Page covers the topic but isn't optimised for the keyword. |
| `strong_match` | `score ≥ 0.65` | The page is the right target; focus on on-page optimisation. |

The `weak_match_rate` and `weakly_matched_percentage` metrics in the cluster
quality report aggregate the proportion of keywords in a cluster sitting at the
`weak_match` band.

## Quality metrics included

- `silhouette_score`
- `avg_intra_cluster_similarity`
- `avg_nearest_cluster_similarity`
- `intent_purity`
- `page_purity`
- `serp_overlap_mean` (when `serp_urls` input exists)
- `weak_match_rate`
- `weakly_matched_percentage`
- `outlier_count`
- `global_outlier_count`
- `outlier_percentage`

## SEO workflow exports

- `page_briefs/*.md`
- `internal_linking_opportunities.csv`
- `keyword_to_heading_map.csv`
- `content_hub_map.html`
- `serp_feature_opportunities.csv`

## Visual outputs

- `cluster_map_3d.html`
- `cluster_map_2d.html`
- `treemap.html`
- `heatmap.html` (when topics are supplied)
- `sankey.html`
- `opportunity_matrix.html`
- `network_graph.html`
- `interactive_report.html`

## Run-history artifacts (`--run-history`)

Each run stores in `outputs/runs/<timestamp>_<id>/`:

- `clustered_keywords.csv`
- `cluster_quality_report.csv`
- `cluster_summary.csv`
- `config.json`
- `input_schema.json`
- `metrics.json`
- `charts/` (full chart bundle)
