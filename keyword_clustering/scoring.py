"""Similarity scoring, mapping, opportunity scoring, cannibalization, and quality metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from .vectorization import compute_serp_overlap_matrix, compute_similarity_matrix

logger = logging.getLogger(__name__)

_OPPORTUNITY_SIGNAL_COLUMNS = (
    "search_volume",
    "keyword_difficulty",
    "rank",
    "business_value",
    "page_similarity_score",
)


@dataclass(frozen=True)
class SimilarityConfig:
    mode: str = "semantic"
    semantic_weight: float = 0.65
    tfidf_weight: float = 0.35
    serp_weight: float = 0.0

    def validate(self, has_serp: bool) -> None:
        if self.mode not in {"semantic", "tfidf", "hybrid"}:
            raise ValueError("similarity mode must be one of: semantic, tfidf, hybrid")
        if self.mode != "hybrid":
            return
        weights = [self.semantic_weight, self.tfidf_weight, self.serp_weight if has_serp else 0.0]
        if any(w < 0 for w in weights):
            raise ValueError("hybrid similarity weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("hybrid similarity weights must sum to > 0")


@dataclass(frozen=True)
class GapThresholdConfig:
    mode: str = "fixed"
    value: float = 0.25
    percentile: float = 25.0
    adaptive_floor: float = 0.20


@dataclass(frozen=True)
class OpportunityConfig:
    profile: str = "balanced"


PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "search_volume_score": 0.25,
        "rank_improvement_score": 0.20,
        "business_value_score": 0.20,
        "page_fit_score": 0.15,
        "intent_value_score": 0.10,
        "serp_feature_score": 0.05,
        "keyword_difficulty_score": -0.05,
    },
    "quick-wins": {
        "search_volume_score": 0.20,
        "rank_improvement_score": 0.10,
        "business_value_score": 0.10,
        "page_fit_score": 0.20,
        "intent_value_score": 0.10,
        "serp_feature_score": 0.05,
        "keyword_difficulty_score": -0.25,
    },
    "growth": {
        "search_volume_score": 0.35,
        "rank_improvement_score": 0.20,
        "business_value_score": 0.20,
        "page_fit_score": 0.10,
        "intent_value_score": 0.10,
        "serp_feature_score": 0.05,
        "keyword_difficulty_score": -0.00,
    },
    "commercial": {
        "search_volume_score": 0.20,
        "rank_improvement_score": 0.20,
        "business_value_score": 0.30,
        "page_fit_score": 0.10,
        "intent_value_score": 0.15,
        "serp_feature_score": 0.05,
        "keyword_difficulty_score": -0.00,
    },
}


def compose_hybrid_similarity_matrix(
    semantic_similarity: np.ndarray | None,
    tfidf_similarity: np.ndarray | None,
    serp_overlap: np.ndarray | None,
    cfg: SimilarityConfig,
) -> np.ndarray:
    has_serp = serp_overlap is not None
    cfg.validate(has_serp=has_serp)
    if cfg.mode == "semantic":
        if semantic_similarity is None:
            raise ValueError("semantic similarity matrix is required for similarity=semantic")
        return semantic_similarity
    if cfg.mode == "tfidf":
        if tfidf_similarity is None:
            raise ValueError("tfidf similarity matrix is required for similarity=tfidf")
        return tfidf_similarity

    mats: list[tuple[np.ndarray, float]] = []
    if semantic_similarity is not None and cfg.semantic_weight > 0:
        mats.append((semantic_similarity, cfg.semantic_weight))
    if tfidf_similarity is not None and cfg.tfidf_weight > 0:
        mats.append((tfidf_similarity, cfg.tfidf_weight))
    if serp_overlap is not None and cfg.serp_weight > 0:
        mats.append((serp_overlap, cfg.serp_weight))
    if not mats:
        raise ValueError("No similarity matrices available for hybrid scoring.")
    total = sum(w for _, w in mats)
    weighted = np.zeros_like(mats[0][0], dtype=float)
    for m, w in mats:
        weighted = weighted + m * w
    return weighted / total


def require_similarity_inputs(
    mode: str,
    semantic_vectors: object | None,
    tfidf_vectors: object | None,
) -> None:
    """Prevent silent fallback when requested similarity channels are unavailable."""
    if mode == "semantic" and semantic_vectors is None:
        raise ValueError(
            "similarity=semantic requires semantic embeddings. Set --embedding-model to a transformer model."
        )
    if mode == "tfidf" and tfidf_vectors is None:
        raise ValueError("similarity=tfidf requires TF-IDF vectors.")
    if mode == "hybrid" and semantic_vectors is None and tfidf_vectors is None:
        raise ValueError("similarity=hybrid requires at least one of semantic or TF-IDF vectors.")


def _normalise_series(
    series: pd.Series,
    *,
    bounds: tuple[float, float] | None = None,
    log_scale: bool = False,
) -> pd.Series:
    """Normalise a numeric Series to [0, 1].

    Args:
        bounds: Optional fixed (min, max) bounds. When supplied, normalisation is
            cross-run-comparable (e.g. difficulty=(0, 100)). When ``None``, the
            min/max of the series itself are used (legacy dataset-relative behaviour).
        log_scale: When ``True``, the input is log1p-transformed before normalisation —
            useful for long-tailed signals like search_volume where the top 1% otherwise
            squashes everything else into the bottom of the range.
    """
    s = series.astype(float)
    if log_scale:
        s = np.log1p(s.clip(lower=0))
    if bounds is not None:
        mn, mx = float(bounds[0]), float(bounds[1])
    else:
        mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((s - mn) / (mx - mn)).clip(0, 1)


def score_keywords_against_corpus(
    keyword_vectors: object,
    corpus_vectors: object,
    corpus_names: list[str],
) -> tuple[list[str], list[float]]:
    sim = compute_similarity_matrix(keyword_vectors, corpus_vectors)
    best_idx = sim.argmax(axis=1)
    best_scores = sim.max(axis=1).tolist()
    best_names = [corpus_names[i] for i in best_idx]
    return best_names, best_scores


def map_keywords_to_pages(
    df: pd.DataFrame,
    keyword_vectors: object,
    page_vectors: object,
    page_names: list[str],
    page_urls: list[str],
    gap_config: GapThresholdConfig | None = None,
) -> pd.DataFrame:
    sim = compute_similarity_matrix(keyword_vectors, page_vectors)
    best_idx = sim.argmax(axis=1)
    best_scores = sim.max(axis=1).tolist()
    out = df.copy()
    out["recommended_page"] = [page_names[i] for i in best_idx]
    out["recommended_url"] = [page_urls[i] for i in best_idx]
    out["page_similarity_score"] = [round(float(s), 4) for s in best_scores]
    cfg = gap_config or GapThresholdConfig()
    if cfg.mode == "fixed":
        threshold = float(cfg.value)
    elif cfg.mode == "percentile":
        threshold = float(np.percentile(out["page_similarity_score"], cfg.percentile))
    elif cfg.mode == "adaptive":
        threshold = max(cfg.adaptive_floor, float(np.percentile(out["page_similarity_score"], cfg.percentile)))
    else:
        raise ValueError("gap_threshold_mode must be fixed, percentile, or adaptive")
    out["gap_threshold_used"] = round(threshold, 4)
    out["content_gap"] = out["page_similarity_score"] <= threshold
    out["match_confidence"] = out["page_similarity_score"].apply(score_confidence_band)
    if "current_url" in out.columns:
        out["current_page"] = out["current_url"].map(dict(zip(page_urls, page_names))).fillna("")
    return out


def score_confidence_band(score: float) -> str:
    s = float(score)
    if s < 0.20:
        return "poor_match"
    if s < 0.40:
        return "weak_match"
    if s < 0.65:
        return "acceptable_match"
    return "strong_match"


def compute_opportunity_score(df: pd.DataFrame, cfg: OpportunityConfig | None = None) -> pd.DataFrame:
    out = df.copy()
    config = cfg or OpportunityConfig()
    if config.profile not in PROFILES:
        raise ValueError("opportunity profile must be one of balanced, quick-wins, growth, commercial")
    w = PROFILES[config.profile]

    insufficient_signal = not any(col in out.columns for col in _OPPORTUNITY_SIGNAL_COLUMNS)
    if insufficient_signal:
        logger.warning(
            "compute_opportunity_score: none of %s present; opportunity_score will collapse to a constant.",
            ", ".join(_OPPORTUNITY_SIGNAL_COLUMNS),
        )

    # Volume is long-tailed: log-scale before normalising so the top 1% don't
    # squash the rest of the corpus to ~0.
    out["search_volume_score"] = (
        _normalise_series(out["search_volume"].fillna(out["search_volume"].median()), log_scale=True)
        if "search_volume" in out.columns
        else 0.5
    )
    # Difficulty is bounded 0..100 by definition — use a fixed scale for cross-run comparability.
    out["keyword_difficulty_score"] = (
        _normalise_series(out["keyword_difficulty"].fillna(50), bounds=(0.0, 100.0))
        if "keyword_difficulty" in out.columns
        else 0.5
    )
    if "rank" in out.columns:
        # Rank gap = "opportunity room above current rank". Rank≤10 (top of page) has nothing
        # to gain → 0.0. Rank≥100 (deep in SERP) has the most to gain → 1.0. Single, documented
        # mapping; no second min-max normalisation on top of it.
        out["rank_improvement_score"] = (out["rank"].fillna(100).clip(lower=10, upper=100) - 10).div(90).clip(0, 1)
    else:
        out["rank_improvement_score"] = 0.5
    out["business_value_score"] = (
        _normalise_series(out["business_value"].fillna(out["business_value"].median()))
        if "business_value" in out.columns
        else 0.5
    )
    out["page_fit_score"] = out["page_similarity_score"].fillna(0.5) if "page_similarity_score" in out.columns else 0.5
    intent_val = {"transactional": 1.0, "commercial": 0.8, "local": 0.75, "informational": 0.5, "navigational": 0.3}
    out["intent_value_score"] = (
        out.get("primary_intent", pd.Series(["informational"] * len(out))).map(intent_val).fillna(0.5)
    )
    serp_cols = [
        c
        for c in ["featured_snippet", "local_pack", "people_also_ask", "image_pack", "video_result"]
        if c in out.columns
    ]
    out["serp_feature_score"] = _normalise_series(out[serp_cols].fillna(0).sum(axis=1)) if serp_cols else 0.5

    comp_cols = list(w.keys())
    for c in comp_cols:
        out[f"opportunity_components_{c}"] = (out[c] * w[c]).round(4)
    out["opportunity_score"] = out[[f"opportunity_components_{c}" for c in comp_cols]].sum(axis=1).round(4)
    out["opportunity_profile"] = config.profile
    if insufficient_signal:
        out["opportunity_reason"] = "insufficient signal: opportunity_score is a constant fallback"
    else:
        out["opportunity_reason"] = out.apply(
            lambda r: (
                f"Profile={config.profile}; volume={r['search_volume_score']:.2f}, rank_gap={r['rank_improvement_score']:.2f}, "
                f"difficulty={r['keyword_difficulty_score']:.2f}, page_fit={r['page_fit_score']:.2f}, intent={r['intent_value_score']:.2f}"
            ),
            axis=1,
        )
    return out


def detect_cannibalization(df: pd.DataFrame) -> pd.DataFrame:
    if "cluster_id" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for cluster_id, group in df.groupby("cluster_id"):
        label = group["cluster_label"].iloc[0] if "cluster_label" in group.columns else str(cluster_id)
        rec_pages = (
            group["recommended_page"].dropna().astype(str).unique().tolist()
            if "recommended_page" in group.columns
            else []
        )
        curr_urls = (
            group["current_url"].dropna().astype(str).unique().tolist() if "current_url" in group.columns else []
        )
        intents = (
            group["primary_intent"].dropna().astype(str).unique().tolist() if "primary_intent" in group.columns else []
        )
        serp_overlap_mean = None
        if "serp_urls" in group.columns and len(group) >= 2:
            local_overlap = compute_serp_overlap_matrix(group)
            tri = local_overlap[np.triu_indices_from(local_overlap, k=1)]
            serp_overlap_mean = float(tri.mean()) if tri.size else 0.0

        if len(curr_urls) > 1:
            detail = ", ".join(curr_urls)
            if serp_overlap_mean is not None:
                detail = f"{detail} | avg_serp_overlap={serp_overlap_mean:.3f}"
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": label,
                    "cannibalization_type": "ranking_cannibalization",
                    "detail": detail,
                    "keyword_count": len(group),
                }
            )
        if len(rec_pages) > 1:
            detail = ", ".join(rec_pages)
            if serp_overlap_mean is not None:
                detail = f"{detail} | avg_serp_overlap={serp_overlap_mean:.3f}"
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": label,
                    "cannibalization_type": "mapping_conflict",
                    "detail": detail,
                    "keyword_count": len(group),
                }
            )
        if len(intents) > 1:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": label,
                    "cannibalization_type": "intent_split",
                    "detail": ", ".join(intents),
                    "keyword_count": len(group),
                }
            )
        if "current_url" in group.columns and "recommended_url" in group.columns:
            mismatch = (
                (group["current_url"].fillna("").astype(str) != "")
                & (group["current_url"].fillna("").astype(str) != group["recommended_url"].fillna("").astype(str))
            ).sum()
            if mismatch > 0:
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "cluster_label": label,
                        "cannibalization_type": "page_mismatch",
                        "detail": f"{mismatch} keywords mismatch current vs recommended URL",
                        "keyword_count": len(group),
                    }
                )
        weak = (group["match_confidence"] == "weak_match").sum() if "match_confidence" in group.columns else 0
        if len(rec_pages) > 1 and weak >= max(1, len(group) // 3):
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": label,
                    "cannibalization_type": "consolidation_candidate",
                    "detail": f"{weak} weak matches across multiple pages",
                    "keyword_count": len(group),
                }
            )
    return pd.DataFrame(rows)


def add_notes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    notes = pd.Series([""] * len(out), index=out.index, dtype=str)
    if "current_page" in out.columns and "recommended_page" in out.columns:
        curr = out["current_page"].fillna("").astype(str)
        rec = out["recommended_page"].fillna("").astype(str)
        mismatch = (curr != "") & (curr != rec)
        notes[mismatch] = "Page mismatch - currently: " + curr[mismatch] + ", recommended: " + rec[mismatch]
    if "content_gap" in out.columns:
        gap = out["content_gap"].fillna(False).astype(bool)
        notes[gap] = notes[gap].apply(lambda n: (n + "; " if n else "") + "Content gap - no suitable page found")
    if "opportunity_score" in out.columns:
        hi = out["opportunity_score"].fillna(0) > out["opportunity_score"].quantile(0.75)
        notes[hi] = notes[hi].apply(lambda n: (n + "; " if n else "") + "High opportunity")
    out["notes"] = notes
    return out


# `compute_cluster_stability` was moved to keyword_clustering.quality but is
# re-exported here for backwards compatibility — older imports still work.
from .quality import compute_cluster_stability  # noqa: E402,F401  (re-export)


def detect_content_gaps(df: pd.DataFrame) -> pd.DataFrame:
    if "content_gap" not in df.columns:
        return pd.DataFrame()
    cols = [
        "keyword",
        "cluster_label",
        "primary_intent",
        "search_volume",
        "keyword_difficulty",
        "opportunity_score",
        "page_similarity_score",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[df["content_gap"]][cols].reset_index(drop=True)


def build_cluster_quality_report(
    df: pd.DataFrame,
    keyword_vectors: object | None = None,
    *,
    include_stability: bool = False,
    stability_method: str = "kmeans",
    stability_bootstrap: int = 10,
) -> pd.DataFrame:
    # Reset the index so positional slicing into `dense` aligns with df rows.
    df = df.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    dense = None
    if keyword_vectors is not None:
        dense = np.asarray(keyword_vectors.toarray() if hasattr(keyword_vectors, "toarray") else keyword_vectors)
    centroid_sim = None
    cluster_centroids: dict[int, np.ndarray] = {}
    if dense is not None and "cluster_id" in df.columns:
        for cid in sorted(df["cluster_id"].dropna().unique().tolist()):
            idx = df.index[df["cluster_id"] == cid].tolist()
            if idx:
                cluster_centroids[int(cid)] = dense[idx].mean(axis=0)
        if cluster_centroids:
            centroid_ids = list(cluster_centroids.keys())
            centroid_matrix = np.vstack([cluster_centroids[cid] for cid in centroid_ids])
            centroid_sim = pd.DataFrame(
                cosine_similarity(centroid_matrix),
                index=centroid_ids,
                columns=centroid_ids,
            )
    cluster_ids = sorted(df["cluster_id"].dropna().unique().tolist()) if "cluster_id" in df.columns else []
    global_silhouette = None
    if dense is not None and "cluster_id" in df.columns and len(set(df["cluster_id"])) > 1 and len(df) > 2:
        try:
            # sample_size keeps the metric tractable on large datasets; n_jobs uses all
            # available cores by default. Set both to None to recover the historical behaviour.
            n = dense.shape[0]
            sample_size = 2_000 if n > 5_000 else None
            global_silhouette = float(
                silhouette_score(
                    dense,
                    df["cluster_id"],
                    metric="cosine",
                    sample_size=sample_size,
                    random_state=42,
                    n_jobs=-1,
                )
            )
        except (ValueError, RuntimeError):
            global_silhouette = None
    serp_overlap_matrix = compute_serp_overlap_matrix(df) if "serp_urls" in df.columns else None
    global_outliers = int((df["cluster_id"] == -1).sum()) if "cluster_id" in df.columns else 0

    stability_scores: dict[int, float] = {}
    if include_stability and dense is not None and "cluster_id" in df.columns:
        try:
            stability_scores = compute_cluster_stability(
                dense,
                df["cluster_id"].to_numpy(),
                method=stability_method,
                n_bootstrap=stability_bootstrap,
            )
        except (ValueError, RuntimeError):
            stability_scores = {}

    for cid in cluster_ids:
        group = df[df["cluster_id"] == cid]
        idx = group.index.tolist()
        intra_sim: float | str = ""
        serp_overlap: float | str = ""
        if dense is not None and len(idx) >= 2:
            local_sim = cosine_similarity(dense[idx], dense[idx])
            tri = local_sim[np.triu_indices_from(local_sim, k=1)]
            intra_sim = round(float(tri.mean()), 4) if tri.size else ""
        if serp_overlap_matrix is not None and len(idx) >= 2:
            local_serp = serp_overlap_matrix[np.ix_(idx, idx)]
            tri = local_serp[np.triu_indices_from(local_serp, k=1)]
            serp_overlap = round(float(tri.mean()), 4) if tri.size else ""

        nearest_cluster_sim: float | str = ""
        if centroid_sim is not None and int(cid) in centroid_sim.index and len(centroid_sim.columns) > 1:
            others = centroid_sim.loc[int(cid)].drop(index=int(cid))
            nearest_cluster_sim = round(float(others.max()), 4) if not others.empty else ""

        row: dict[str, object] = {
            "cluster_id": cid,
            "cluster_label": group["cluster_label"].iloc[0] if "cluster_label" in group.columns else str(cid),
            "cluster_size": len(group),
            "top_terms": group["top_terms"].iloc[0] if "top_terms" in group.columns else "",
            "silhouette_score": global_silhouette if global_silhouette is not None else "",
            "avg_intra_cluster_similarity": intra_sim,
            "avg_nearest_cluster_similarity": nearest_cluster_sim,
            "intent_purity": round(group["primary_intent"].value_counts(normalize=True).iloc[0], 4)
            if "primary_intent" in group.columns and len(group)
            else "",
            "page_purity": round(group["recommended_page"].value_counts(normalize=True).iloc[0], 4)
            if "recommended_page" in group.columns and len(group)
            else "",
            "serp_overlap_mean": serp_overlap,
            "weak_match_rate": round(
                (group.get("match_confidence", pd.Series([], dtype=str)) == "weak_match").mean(), 4
            )
            if "match_confidence" in group.columns and len(group)
            else "",
            "weakly_matched_percentage": round(
                100 * (group.get("match_confidence", pd.Series([], dtype=str)) == "weak_match").mean(), 2
            )
            if "match_confidence" in group.columns and len(group)
            else "",
            "outlier_count": int((group["cluster_id"] == -1).sum()) if len(group) else 0,
            "global_outlier_count": global_outliers,
            "outlier_percentage": round(100 * global_outliers / max(1, len(df)), 2),
        }
        if include_stability:
            row["cluster_stability"] = stability_scores.get(int(cid), float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)
