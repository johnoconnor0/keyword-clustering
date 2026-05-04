# Output Files

All files are written to the directory specified by `--output` (default: `outputs/`).

## CSV reports

### `clustered_keywords.csv`

One row per keyword. Core output for strategy work.

| Column | Description |
|---|---|
| `keyword` | The keyword string |
| `cluster_id` | Integer cluster label |
| `cluster_label` | Human-readable cluster name (auto-generated) |
| `primary_intent` | informational / commercial / transactional / navigational |
| `recommended_page` | Best-matching page name |
| `recommended_url` | URL of the recommended page |
| `current_page` | Page currently ranking (derived from `current_url` in input) |
| `page_similarity_score` | Cosine similarity to recommended page (0–1) |
| `topic_similarity_score` | Cosine similarity to best-matching topic |
| `opportunity_score` | Composite score: volume × 0.4 − difficulty × 0.3 + rank_gap × 0.3 |
| `search_volume` | Monthly search volume |
| `keyword_difficulty` | 0–100 difficulty score |
| `cpc` | Cost per click |
| `rank` | Current ranking position |
| `branded` | True if keyword contains a brand term |
| `notes` | Auto-generated flags: page mismatch, content gap, high opportunity |

### `keyword_page_map.csv`

Keyword → page assignments with similarity scores and intent.

### `content_gap_report.csv`

Keywords where `page_similarity_score == 0` — no existing page matches the keyword.
These represent opportunities to create new content.

### `cannibalization_report.csv`

Clusters where two or more distinct pages are mapped as `recommended_page`.
Each row lists the competing pages and a sample of the keywords involved.

### `cluster_summary.csv`

Per-cluster aggregated metrics: keyword count, total search volume, average difficulty,
average opportunity score, and a sample of the top keywords.

### `recommendations.md`

Plain-English strategy notes per cluster including recommended page, primary intent,
total search volume, opportunity score, and suggested actions.

## Interactive HTML charts

All charts use Plotly with CDN-loaded JS (~20 KB per file).

| File | Description |
|---|---|
| `cluster_map_3d.html` | 3D scatter coloured by cluster, sized by search volume |
| `cluster_map_2d.html` | 2D topic map (PCA/UMAP/t-SNE components 1 & 2) |
| `treemap.html` | Cluster treemap — size: keyword count, colour: total volume |
| `heatmap.html` | Cosine similarity heatmap across topics |
| `sankey.html` | Page → cluster keyword flow |
| `opportunity_matrix.html` | SERP scatter: x=difficulty, y=opportunity score, size=volume |
| `network_graph.html` | Keyword similarity network (nodes=keywords, edges=high cosine similarity) |
| `interactive_report.html` | All charts bundled into a single portable HTML file |
