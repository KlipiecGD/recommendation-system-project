import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import compute_user_metrics, aggregate_metrics
from src.logging_utils.logger import logger
from src.config.config import config


# Paths
PROCESSED_DIR = Path(config.data_config.get("processed_dir", "data/processed/full"))
CF_ARTIFACTS_DIR = Path(config.data_config.get("cf_artifacts_dir", "artifacts/cf"))

# Evaluation parameters
KS = config.evaluation_config.get("ks", [5, 10, 20])
RELEVANCE_THRESHOLD = config.evaluation_config.get("relevance_threshold", 4.0)


def load_ratings_for_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and val ratings with raw userId/movieId (not encoded indices).
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_ratings, val_ratings)
    """
    with open(PROCESSED_DIR / "train_ratings.pkl", "rb") as f:
        train = pickle.load(f)
    with open(PROCESSED_DIR / "val_ratings.pkl", "rb") as f:
        val = pickle.load(f)

    logger.info(f"Loaded train: {len(train):,} | val: {len(val):,} ratings")
    return train, val


def load_movies() -> pd.DataFrame:
    """
    Load enriched movies DataFrame. Used to map movieId -> title in results.

    Returns:
        pd.DataFrame: movies_enriched with at least 'movieId' and 'title' columns.
    """
    df = pd.read_parquet(PROCESSED_DIR / "movies_enriched.parquet")
    return df.dropna(subset=["movieId"])[["movieId", "title", "movie_idx"]].copy()


def compute_rmse(model, val_df: pd.DataFrame) -> float:
    """
    Compute Root Mean Squared Error on the validation set.
    Args:
        model (CFModel): A fitted CFModel instance.
        val_df (pd.DataFrame): Validation ratings with 'userId', 'movieId', 'rating'.
    Returns:
        float: RMSE score. Lower is better.
    """
    errors = []
    for _, row in tqdm(
        val_df.iterrows(), total=len(val_df), desc=f"{model.model_name} RMSE"
    ):
        predicted = model.predict(int(row["userId"]), int(row["movieId"]))
        errors.append((predicted - row["rating"]) ** 2)

    rmse = float(np.sqrt(np.mean(errors)))
    logger.info(f"{model.model_name} RMSE: {rmse:.4f}")
    return rmse


def evaluate_ranking(
    model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    ks: list[int] = KS,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    max_users: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate ranking quality using HR@K, Precision@K, Recall@K, NDCG@K.
    Note: uses movie_idx (not movieId) for ground truth set because
    compute_user_metrics expects integer indices consistent with CB evaluation.
    Args:
        model (CFModel): A fitted CFModel instance.
        train_df (pd.DataFrame): Train ratings with 'userId', 'movieId', 'movie_idx', 'rating'.
        val_df (pd.DataFrame): Val ratings with same columns.
        movies_df (pd.DataFrame): Movies with 'movieId', 'title', 'movie_idx'.
        ks (list[int]): Cut-off values.
        relevance_threshold (float): Min rating to count as relevant.
        max_users (int | None): Cap evaluated users for speed during development.
    Returns:
        pd.DataFrame: One row per eligible user with HR@K, P@K, R@K, NDCG@K columns.
    """
    # movieId -> movie_idx mapping for ground truth alignment with CB metrics
    movieid_to_idx = dict(zip(movies_df["movieId"], movies_df["movie_idx"]))

    # Group by user
    train_by_user = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    val_by_user = (
        val_df.groupby("userId")
        .apply(lambda df: list(zip(df["movieId"], df["movie_idx"], df["rating"])))
        .to_dict()
    )

    max_k = max(ks)
    eval_users = list(val_by_user.keys())
    if max_users is not None:
        eval_users = eval_users[:max_users]

    logger.info(
        f"Evaluating {model.model_name} ranking | "
        f"users={len(eval_users):,} | K={ks} | threshold={relevance_threshold}"
    )

    results = []

    for user_id in tqdm(eval_users, desc=f"{model.model_name} ranking"):
        # Skip users with no training data
        if user_id not in train_by_user or not train_by_user[user_id]:
            continue

        # Ground truth: val items rated >= threshold, expressed as movie_idx
        gt_indices = {
            int(movie_idx)
            for _, movie_idx, rating in val_by_user[user_id]
            if rating >= relevance_threshold and not pd.isna(movie_idx)
        }
        if not gt_indices:
            continue

        # Get recommendations — returns list of {movieId, title, predicted_rating}
        recs = model.recommend(user_id, movies_df, n=max_k, filter_rated=True)
        if not recs:
            continue

        # Convert recommended movieIds to movie_idx for metric computation
        rec_indices = [
            int(movieid_to_idx[r["movieId"]])
            for r in recs
            if r["movieId"] in movieid_to_idx
        ]

        row = {
            "user_id": user_id,
            "n_train": len(train_by_user[user_id]),
            "n_gt": len(gt_indices),
        }
        row.update(compute_user_metrics(rec_indices, gt_indices, ks=ks))
        results.append(row)

    result_df = pd.DataFrame(results)
    n_skipped = len(eval_users) - len(result_df)
    logger.info(
        f"{model.model_name}: evaluated {len(result_df):,} users "
        f"(skipped {n_skipped:,} ineligible)"
    )
    return result_df


def evaluate_model(
    model,
    ks: list[int] = KS,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    max_users: int | None = None,
) -> dict:
    """
    Run full evaluation (RMSE + ranking) for a single fitted CFModel.
    Args:
        model (CFModel): A fitted CFModel instance.
        ks (list[int]): Cut-off values for ranking metrics.
        relevance_threshold (float): Min rating to count as relevant.
        max_users (int | None): Cap on evaluated users for ranking metrics.
    Returns:
        dict: {
            'model': model_name,
            'n_users': int,
            'rmse': float,
            'hr@5': float, 'p@5': float, ... (all ranking metrics)
        }
    """
    train_df, val_df = load_ratings_for_eval()
    movies_df = load_movies()

    # RMSE
    rmse = compute_rmse(model, val_df)

    # Ranking
    per_user_df = evaluate_ranking(
        model,
        train_df,
        val_df,
        movies_df,
        ks=ks,
        relevance_threshold=relevance_threshold,
        max_users=max_users,
    )

    summary = aggregate_metrics(per_user_df, ks=ks).to_dict()
    return {
        "model": model.model_name,
        "n_users": len(per_user_df),
        "rmse": round(rmse, 4),
        **{k: round(v, 6) for k, v in summary.items()},
    }
