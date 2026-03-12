import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config import config
from src.logging_utils.logger import logger
from src.models.content_based.model_registry import MODEL_REGISTRY
from src.preprocessing.loaders import TAGS_PATH

# Paths
_data_cfg = config.data_config
PROCESSED_DIR = Path(_data_cfg.get("processed_dir", "data/processed/full"))
# Evaluation artifacts are stored separately from production artifacts
EVAL_ARTIFACTS_DIR = Path(_data_cfg.get("eval_artifacts_dir", "artifacts/eval"))


# Build train-only movies_enriched
def build_train_only_movies_enriched() -> tuple[pd.DataFrame, dict, np.ndarray]:
    """
    Load base artifacts and rebuild movies_enriched with user_tags_text derived
    exclusively from training-split interactions.
    Steps:
      1. Load train_ratings.pkl (encoded indices).
      2. Decode userId / movieId back to raw IDs using user_enc / movie_enc.
      3. Load raw tags.csv and inner-join on (userId, movieId) train pairs.
      4. Re-aggregate per-movie tag text and patch movies_enriched.
      5. Return (movies_enriched_train_only, movie_enc, genome_matrix).
    Returns:
        tuple[pd.DataFrame, dict, np.ndarray]: (movies_enriched, movie_enc, genome_matrix)
    """
    logger.info("Loading split artifacts ...")

    # Load base artifacts
    with open(PROCESSED_DIR / "train_ratings.pkl", "rb") as f:
        train_ratings: pd.DataFrame = pickle.load(f)

    with open(PROCESSED_DIR / "movie_enc.pkl", "rb") as f:
        movie_enc: dict = pickle.load(f)

    movies_enriched = pd.read_parquet(PROCESSED_DIR / "movies_enriched.parquet")
    genome_matrix = np.load(PROCESSED_DIR / "genome_matrix.npy")

    logger.info(
        f"train_ratings: {len(train_ratings):,} | "
        f"movies_enriched: {movies_enriched.shape}"
    )

    # train_ratings contains both raw IDs (userId, movieId) and encoded indices
    # (user_idx, movie_idx). Use raw IDs directly to match against tags.csv.
    train_pairs = set(
        zip(train_ratings["userId"].astype(int), train_ratings["movieId"].astype(int))
    )
    logger.info(f"Training (userId, movieId) pairs: {len(train_pairs):,}")

    # Load raw tags and restrict to training pairs
    if not TAGS_PATH.exists():
        logger.warning(
            f"Tags file not found at {TAGS_PATH}. user_tags_text will be empty."
        )
        tags_train = pd.DataFrame(columns=["userId", "movieId", "tag"])
    else:
        logger.info(f"Loading raw tags from {TAGS_PATH} ...")
        tags_raw = pd.read_csv(TAGS_PATH)
        logger.info(f"Raw tags: {len(tags_raw):,}")

        # Keep only tags whose (userId, movieId) pair is in the training split
        catalogue_movie_ids = set(movies_enriched["movieId"].astype(int))
        # Keep only tags for movies in the catalogue (after filtering - dropping some of movies from the
        # original dataset - we don't want tags for movies that won't be in the model's feature matrix at all)
        tags_relevant = tags_raw[tags_raw["movieId"].isin(catalogue_movie_ids)]
        logger.info(f"Tags for catalogue movies: {len(tags_relevant):,}")

        mask = [
            (int(u), int(m)) in train_pairs
            for u, m in zip(tags_relevant["userId"], tags_relevant["movieId"])
        ]
        tags_train = tags_relevant[mask].copy()
        logger.info(
            f"Tags after train-only filter: {len(tags_train):,} "
            f"({100 * len(tags_train) / max(len(tags_relevant), 1):.1f}% of catalogue tags)"
        )

    # Rebuild user_tags_text from train-only tags
    if len(tags_train) > 0:
        tag_agg = (
            tags_train.groupby("movieId")["tag"]
            .apply(lambda t: " ".join(t.dropna().astype(str)))
            .reset_index()
            .rename(columns={"tag": "user_tags_text"})
        )
        # Drop existing user_tags_text and replace with train-only version
        movies_enriched = movies_enriched.drop(
            columns=["user_tags_text"], errors="ignore"
        )
        movies_enriched = movies_enriched.merge(tag_agg, on="movieId", how="left")
        logger.info(
            f"Rebuilt user_tags_text for "
            f"{tag_agg['movieId'].nunique():,} movies (train-only tags)"
        )
    else:
        logger.warning("No train tags available — user_tags_text will be NaN.")

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
    Instantiate, fit, and save one model to the eval artifacts directory.
    Args:
        key (str): Short model key, e.g. 'cb1'.
        model_cls (type): Model class to instantiate.
        needs_genome (bool): Whether the model requires genome_matrix.
        movies_enriched (pd.DataFrame): Train-only enriched movies DataFrame.
        movie_enc (dict): Encoder dict with 'to_idx' mapping.
        genome_matrix (np.ndarray): (n_movies × 1128) genome scores.
    """
    artifact_dir = EVAL_ARTIFACTS_DIR / key
    logger.info(f"Fitting {key.upper()} (eval mode) ...")

    t0 = time.perf_counter()
    model = model_cls()

    if needs_genome:
        model.fit(movies_enriched, movie_enc, genome_matrix)
    else:
        model.fit(movies_enriched, movie_enc)

    elapsed = time.perf_counter() - t0
    model.save(artifact_dir)
    logger.info(f"{key.upper()} done in {elapsed:.1f}s - saved to {artifact_dir}")


# Command-line interface
def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments to select which models to fit for evaluation.
    Returns:
        argparse.Namespace: Parsed arguments with attributes 'models' and 'skip'.
    """
    all_keys = list(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description=(
            "Fit CB models on training-split data only, "
            "for use with the offline evaluation pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--models",
        nargs="+",
        choices=all_keys,
        metavar="MODEL",
        help=f"Fit only these models. Choices: {all_keys}",
    )
    group.add_argument(
        "--skip",
        nargs="+",
        choices=all_keys,
        metavar="MODEL",
        help="Fit all models except these.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to fit selected content-based models for evaluation.
    Models are fitted on the training split only, with user_tags_text derived
    exclusively from training interactions. Fitted models are saved to the eval artifacts directory.
    """
    args = _parse_args()

    if args.models:
        keys_to_train = args.models
    elif args.skip:
        keys_to_train = [k for k in MODEL_REGISTRY if k not in args.skip]
    else:
        keys_to_train = list(MODEL_REGISTRY.keys())

    logger.info(f"Models to fit (eval mode): {keys_to_train}")
    logger.info(f"Artifacts will be saved to: {EVAL_ARTIFACTS_DIR}")

    movies_enriched, movie_enc, genome_matrix = build_train_only_movies_enriched()

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

    logger.info("Training summary (eval mode):")
    for key, status in results:
        logger.info(f"{key.upper():6s}  {status}")


if __name__ == "__main__":
    main()
