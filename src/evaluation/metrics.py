import math
import pandas as pd
from typing import Sequence


def hit_rate_at_k(recs: Sequence[int], ground_truth: set[int], k: int) -> float:
    """
    Hit Rate at K.
    Returns 1.0 if at least one item in recs[:k] is relevant, else 0.0.
    Args:
        recs (Sequence[int]): Ordered list of recommended movie indices.
        ground_truth (set[int]): Set of relevant (liked) movie indices for the user.
        k (int): Cut-off rank.
    Returns:
        float: 1.0 if there is at least one hit in top-K, otherwise 0.0.
    """
    return float(bool(set(recs[:k]) & ground_truth))


def precision_at_k(recs: Sequence[int], ground_truth: set[int], k: int) -> float:
    """
    Precision at K.
    Fraction of the top-K list that is relevant.
    Args:
        recs (Sequence[int]): Ordered list of recommended movie indices.
        ground_truth (set[int]): Set of relevant movie indices for the user.
        k (int): Cut-off rank.
    Returns:
        float: Number of hits in top-K divided by K.
    """
    if k == 0:
        return 0.0
    hits = len(set(recs[:k]) & ground_truth)
    return hits / k


def recall_at_k(recs: Sequence[int], ground_truth: set[int], k: int) -> float:
    """
    Recall at K.
    Fraction of the user's relevant items that appear in the top-K list.
    Args:
        recs (Sequence[int]): Ordered list of recommended movie indices.
        ground_truth (set[int]): Set of relevant movie indices for the user.
        k (int): Cut-off rank.
    Returns:
        float: Number of hits in top-K divided by total number of relevant items.
    """
    if not ground_truth:
        return 0.0
    hits = len(set(recs[:k]) & ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(recs: Sequence[int], ground_truth: set[int], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    A hit at rank 1 contributes more than a hit at rank K. Normalised by the
    ideal ordering (all relevant items at the top).
    Args:
        recs (Sequence[int]): Ordered list of recommended movie indices.
        ground_truth (set[int]): Set of relevant movie indices for the user.
        k (int): Cut-off rank.
    Returns:
        float: NDCG@K in [0, 1].
    """
    dcg = 0.0
    for i, idx in enumerate(recs[:k]):
        if idx in ground_truth:
            dcg += 1.0 / math.log2(i + 2)  # rank i+1 -> log2(i+2)

    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def compute_user_metrics(
    recs: Sequence[int],
    ground_truth: set[int],
    ks: list[int],
) -> dict[str, float]:
    """
    Compute all four metrics at every cut-off K for a single user.
    Args:
        recs (Sequence[int]): Ordered list of recommended movie indices (length >= max(ks)).
        ground_truth (set[int]): Set of relevant movie indices for the user.
        ks (list[int]): List of cut-off values, e.g. [5, 10, 20].
    Returns:
        dict[str, float]: Dict with keys "hr@K", "p@K", "r@K", "ndcg@K" for each K in ks.
    """
    result = {}
    for k in ks:
        result[f"hr@{k}"] = hit_rate_at_k(recs, ground_truth, k)
        result[f"p@{k}"] = precision_at_k(recs, ground_truth, k)
        result[f"r@{k}"] = recall_at_k(recs, ground_truth, k)
        result[f"ndcg@{k}"] = ndcg_at_k(recs, ground_truth, k)
    return result


def aggregate_metrics(df, ks: list[int]) -> pd.Series:
    """
    Aggregate per-user results to mean metrics across all evaluated users.
    Args:
        df (pd.DataFrame): Per-user metric DataFrame produced by an evaluator.
        ks (list[int]): Cut-off values used during evaluation.
    Returns:
        pd.Series: Mean of each metric column, indexed by metric name.
    """
    metric_cols = [f"{m}@{k}" for k in ks for m in ("hr", "p", "r", "ndcg")]
    present = [c for c in metric_cols if c in df.columns]
    return df[present].mean()
