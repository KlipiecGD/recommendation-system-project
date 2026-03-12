import argparse
import pickle
import time
from pathlib import Path

import pandas as pd

from src.models.collaborative_filtering.algorithm_registry import ALGORITHM_REGISTRY
from src.models.collaborative_filtering.cf_model import CFModel
from src.logging_utils.logger import logger
from src.config.config import config

PROCESSED_DIR = Path(config.data_config.get("processed_dir", "data/processed/full"))
CF_ARTIFACTS_DIR = Path(config.data_config.get("cf_artifacts_dir", "artifacts/cf"))


def load_train_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train ratings and movies_enriched for fitting.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_ratings, movies_enriched)
            train_ratings has raw userId, movieId, rating columns.
            movies_enriched has movieId, title for recommend() calls.
    """
    logger.info(f"Loading data from {PROCESSED_DIR} ...")

    with open(PROCESSED_DIR / "train_ratings.pkl", "rb") as f:
        train_ratings: pd.DataFrame = pickle.load(f)

    movies_enriched = pd.read_parquet(PROCESSED_DIR / "movies_enriched.parquet")
    movies_enriched = movies_enriched.dropna(subset=["movieId"])[
        ["movieId", "title", "movie_idx"]
    ].copy()

    logger.info(
        f"train_ratings: {len(train_ratings):,} | "
        f"users: {train_ratings['userId'].nunique():,} | "
        f"movies: {train_ratings['movieId'].nunique():,}"
    )
    logger.info(f"movies_enriched: {len(movies_enriched):,} movies")

    return train_ratings, movies_enriched


def fit_model(
    algo_key: str,
    train_df: pd.DataFrame,
    params: dict | None = None,
) -> CFModel:
    """
    Instantiate, fit, and save a single CF model.
    Args:
        algo_key (str): One of 'svd', 'svdpp', 'nmf', 'knn'.
        train_df (pd.DataFrame): Training ratings.
        params (dict | None): Optional hyperparameter overrides.
    Returns:
        CFModel: The fitted model instance.
    """
    logger.info(f"Training {algo_key.upper()} ...")

    t0 = time.perf_counter()

    model = CFModel(algo_key=algo_key, params=params)
    model.fit(train_df)

    elapsed = time.perf_counter() - t0
    logger.info(f"{algo_key.upper()} trained in {elapsed:.1f}s")

    model.save(CF_ARTIFACTS_DIR)
    return model


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training CF models.
    Returns:
        argparse.Namespace: Parsed arguments with 'models' or 'skip' attributes.
    """
    all_keys = list(ALGORITHM_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Train collaborative filtering models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--models",
        nargs="+",
        choices=all_keys,
        metavar="MODEL",
        help=f"Train only these models. Choices: {all_keys}",
    )
    group.add_argument(
        "--skip",
        nargs="+",
        choices=all_keys,
        metavar="MODEL",
        help="Train all models except these.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for training CF models.
    Parses arguments, loads data, trains specified models, and logs results.
    """
    args = _parse_args()
    all_keys = list(ALGORITHM_REGISTRY.keys())

    if args.models:
        keys_to_train = args.models
    elif args.skip:
        keys_to_train = [k for k in all_keys if k not in args.skip]
    else:
        keys_to_train = all_keys

    logger.info(f"Models to train: {keys_to_train}")
    logger.info(f"Artifacts will be saved to: {CF_ARTIFACTS_DIR}")

    train_df, _ = load_train_data()

    results = []

    for key in keys_to_train:
        try:
            fit_model(key, train_df)
            results.append((key, "OK"))
        except Exception as exc:
            logger.error(f"{key.upper()} FAILED: {exc}", exc_info=True)
            results.append((key, f"FAILED: {exc}"))

    # Summary
    logger.info("Training summary:")
    for key, status in results:
        logger.info(f"  {key.upper():8s}  {status}")


if __name__ == "__main__":
    main()
