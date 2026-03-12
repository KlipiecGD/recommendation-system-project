import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config.config import config
from src.models.content_based.model_registry import (
    MODEL_REGISTRY_STRINGS as MODEL_REGISTRY,
)
from src.evaluation.metrics import compute_user_metrics
from src.logging_utils.logger import logger

# Paths
_data_cfg = config.data_config
PROCESSED_DIR = Path(_data_cfg.get("processed_dir", "data/processed/full"))
# Evaluation always uses artifacts trained on training-split data only.
ARTIFACTS_DIR = Path(_data_cfg.get("eval_artifacts_dir", "artifacts/eval"))

# Evaluation parameters
KS = config.evaluation_config.get("ks", [5, 10, 20])
RELEVANCE_THRESHOLD = config.evaluation_config.get("relevance_threshold", 4.0)
SPLIT = config.evaluation_config.get("split", "val")


# Model loading
def load_model(model_key: str) -> object:
    """
    Load a fitted CB model from its saved artifact directory.
    Args:
        model_key (str): Short model key, e.g. 'cb1'. Must be in MODEL_REGISTRY.
    Returns:
        object: The deserialised model object (BaseCBModel subclass).
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    subdir, model_name = MODEL_REGISTRY[model_key]
    path = ARTIFACTS_DIR / subdir / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No artifact found for '{model_key}' at {path}. "
            "Run: python -m src.fitting.fit_cb_train"
        )
    logger.info(f"Loading model '{model_key}' from {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# Split loading
def _load_split_ratings(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and eval ratings from the preprocessed artifacts produced by
    build_dataset.py.
    Args:
        split (str): 'val' or 'test'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_ratings, eval_ratings) as DataFrames with columns:
            userId (int), movieId (int), rating (float).
    """
    assert split in ("val", "test"), f"split must be 'val' or 'test', got '{split}'"

    train_path = PROCESSED_DIR / "train_ratings.pkl"
    eval_path = PROCESSED_DIR / f"{split}_ratings.pkl"

    for p in (train_path, eval_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Ratings artifact not found at {p}. "
                "Run python -m src.data_source.build_dataset first to generate split artifacts."
            )

    with open(train_path, "rb") as f:
        train_ratings = pickle.load(f)
    with open(eval_path, "rb") as f:
        eval_ratings = pickle.load(f)

    logger.info(
        f"Loaded split artifacts — "
        f"train: {len(train_ratings):,} | {split}: {len(eval_ratings):,}"
    )
    return train_ratings, eval_ratings


# Core evaluation loop
def evaluate_model(
    model,
    split: str = SPLIT,
    ks: list[int] = KS,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    max_users: int | None = None,
    profile_strategy: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate a fitted CB model using the offline held-out protocol.
    For each eligible user:
        1. Build a title->rating profile from their TRAIN ratings.
        2. Collect ground-truth = items rated >= relevance_threshold in eval split.
        3. Call model.recommend_from_ratings(profile, n=max(ks)).
        4. Compute HR@K, P@K, R@K, NDCG@K for each K.
    A user is skipped if they have no training ratings, no title matches in the
    feature matrix, or no relevant items in the eval split.
    Args:
        model (BaseCBModel): A fitted BaseCBModel instance.
        split (str): 'val' or 'test'.
        ks (list[int]): Cut-off values, e.g. [5, 10, 20].
        relevance_threshold (float): Minimum rating to count as a positive (liked) item.
        max_users (int | None): If set, cap the number of evaluated users
        profile_strategy (str | None): If provided, temporarily override the model's
            profile_strategy for this evaluation run ('weighted' or 'mean_centering').
            The model's original strategy is restored after evaluation.
    Returns:
        pd.DataFrame: DataFrame with one row per eligible user and columns:
            user_idx, n_train, n_gt, hr@K, p@K, r@K, ndcg@K for each K in ks.
    """
    train_ratings, eval_ratings = _load_split_ratings(split)

    # Temporarily override profile_strategy if requested
    original_strategy = getattr(model, "profile_strategy", None)
    if profile_strategy is not None and original_strategy is not None:
        model.profile_strategy = profile_strategy

    active_strategy = getattr(model, "profile_strategy", "n/a")

    # Group by user
    train_by_user = (
        train_ratings.groupby("userId")
        .apply(
            lambda df: list(
                zip(df["movie_idx"].astype(int), df["rating"].astype(float))
            )
        )
        .to_dict()
    )
    eval_by_user = (
        eval_ratings.groupby("userId")
        .apply(
            lambda df: list(
                zip(df["movie_idx"].astype(int), df["rating"].astype(float))
            )
        )
        .to_dict()
    )

    max_k = max(ks)
    eval_users = list(eval_by_user.keys())
    if max_users is not None:
        eval_users = eval_users[:max_users]

    logger.info(
        f"Evaluating {model.model_name} | split={split} | "
        f"strategy={active_strategy} | "
        f"users={len(eval_users):,} | K={ks} | threshold={relevance_threshold}"
    )

    results = []

    for user_idx in tqdm(eval_users, desc=model.model_name):
        # Build training profile
        train_items = train_by_user.get(user_idx, [])
        if not train_items:
            continue

        profile = {}
        for movie_idx, rating in train_items:
            title = model.idx_to_title.get(movie_idx)
            if title is not None:
                profile[title] = rating

        if not profile:
            continue  # None of the user's movies are in the feature matrix

        # Build ground truth
        gt_indices = {
            movie_idx
            for movie_idx, rating in eval_by_user[user_idx]
            if rating >= relevance_threshold
        }
        if not gt_indices:
            continue  # Nothing to hit

        # Get recommendations
        recs = model.recommend_from_ratings(profile, n=max_k)
        rec_indices = [r["movie_idx"] for r in recs]

        # Compute metrics
        row = {
            "user_idx": user_idx,
            "n_train": len(train_items),
            "n_gt": len(gt_indices),
        }
        row.update(compute_user_metrics(rec_indices, gt_indices, ks=ks))
        results.append(row)

    result_df = pd.DataFrame(results)
    n_skipped = len(eval_users) - len(result_df)
    logger.info(
        f"{model.model_name} [{active_strategy}]: evaluated {len(result_df):,} users "
        f"(skipped {n_skipped:,} ineligible)"
    )

    # Restore original strategy
    if profile_strategy is not None and original_strategy is not None:
        model.profile_strategy = original_strategy

    return result_df
