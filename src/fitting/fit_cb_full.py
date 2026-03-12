import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config import config
from src.logging_utils.logger import logger
from src.models.content_based.model_registry import MODEL_REGISTRY

# Paths
_data_cfg = config.data_config
PROCESSED_DIR = Path(_data_cfg.get("processed_dir", "data/processed/full"))
ARTIFACTS_DIR = Path(_data_cfg.get("artifacts_dir", "artifacts/full"))


# Data loading (cached across models)
def load_data() -> tuple[pd.DataFrame, dict, np.ndarray]:
    """
    Load and return (movies_enriched, movie_enc, genome_matrix).
    Returns:
        tuple[pd.DataFrame, dict, np.ndarray]: Loaded data components.
    """
    logger.info(f"Loading data from {PROCESSED_DIR} ...")

    movies_enriched = pd.read_parquet(PROCESSED_DIR / "movies_enriched.parquet")
    logger.info(f"movies_enriched: {movies_enriched.shape}")

    with open(PROCESSED_DIR / "movie_enc.pkl", "rb") as f:
        movie_enc = pickle.load(f)
    logger.info(f"movie_enc: {len(movie_enc['to_idx']):,} movies")

    genome_matrix = np.load(PROCESSED_DIR / "genome_matrix.npy")
    logger.info(f"genome_matrix: {genome_matrix.shape}")

    return movies_enriched, movie_enc, genome_matrix


# Fit one model
def fit_model(
    key: str,
    model_cls: type,
    needs_genome: bool,
    movies_enriched: pd.DataFrame,
    movie_enc: dict,
    genome_matrix: np.ndarray,
) -> None:
    """
    Instantiate, fit, and save one model.
    Args:
        key (str): Short key for the model (e.g., 'cb1').
        model_cls (type): The model class to instantiate.
        needs_genome (bool): Whether this model requires the genome_matrix.
        movies_enriched (pd.DataFrame): Enriched movies DataFrame.
        movie_enc (dict): Encoder dict with 'to_idx' mapping.
        genome_matrix (np.ndarray): (n_movies x 1128) genome scores.
    """
    artifact_dir = ARTIFACTS_DIR / key

    logger.info(f"Fitting {key.upper()} ...")

    t0 = time.perf_counter()
    model = model_cls()

    if needs_genome:
        model.fit(movies_enriched, movie_enc, genome_matrix)
    else:
        model.fit(movies_enriched, movie_enc)

    elapsed = time.perf_counter() - t0
    model.save(artifact_dir)
    logger.info(f"{key.upper()} done in {elapsed:.1f}s - saved to {artifact_dir}")


# Entry point
def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    Usage examples:
        - `--models cb1 cb3` to train only CB-1 and CB-3.
        - `--skip cb5 cb6` to train all except CB-5 and CB-6.
        - No args to train all models.
    Returns:
        argparse.Namespace: Parsed arguments with 'models' and 'skip' attributes.
    """
    all_keys = list(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Train content-based recommender models."
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
    Main function to orchestrate fitting of content-based models.
    Usage:
        - `python -m src.fitting.fit_cb_full` to fit all models.
        - `python -m src.fitting.fit_cb_full --models cb1 cb3` to fit only CB-1 and CB-3.
        - `python -m src.fitting.fit_cb_full --skip cb5 cb6` to fit all except CB-5 and CB-6.
    """
    args = _parse_args()

    if args.models:
        keys_to_train = args.models
    elif args.skip:
        keys_to_train = [k for k in MODEL_REGISTRY if k not in args.skip]
    else:
        keys_to_train = list(MODEL_REGISTRY.keys())

    logger.info(f"Models to fit: {keys_to_train}")

    movies_enriched, movie_enc, genome_matrix = load_data()

    results = []

    for key in keys_to_train:
        model_cls, needs_genome = MODEL_REGISTRY[key]
        try:
            fit_model(
                key, model_cls, needs_genome, movies_enriched, movie_enc, genome_matrix
            )
            results.append((key, "OK"))
        except Exception as exc:
            logger.error(f"{key.upper()} FAILED: {exc}", exc_info=True)
            results.append((key, f"FAILED: {exc}"))

    # Summary
    logger.info("Fitting summary:")
    for key, status in results:
        logger.info(f"{key.upper():6s}  {status}")


if __name__ == "__main__":
    main()
