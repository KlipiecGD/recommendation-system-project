import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.logging_utils.logger import logger
from src.preprocessing.loaders import (
    filter_ratings,
    load_ratings,
    load_movies,
    preprocess_movies,
    load_tags,
    load_links,
    load_genome,
    PROCESSED_DIR,
    LINKS_PATH,
    GENOME_SCORES_PATH,
)
from src.preprocessing.encoding import encode_ids
from src.preprocessing.matrices import (
    build_sparse_matrix,
    build_genome_matrix,
)
from src.data_management.artifacts_management import save_artifacts
from src.fitting.temporal_split import temporal_train_val_test_split

from src.config.config import config

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

VAL_RATIO = config.data_config.get("val_ratio", 0.1)
TEST_RATIO = config.data_config.get("test_ratio", 0.1)
MIN_RATINGS_PER_USER = config.data_config.get("min_ratings_per_user_split", 5)


def build(fetch_tmdb: bool = True, tmdb_key: str = TMDB_API_KEY) -> None:
    """
    Run the full data preparation pipeline.
    Steps:
        1. Load raw data
        2. Encode user/movie IDs to contiguous 0-based indices
        3. Temporal train / val / test split
        4. Build sparse rating matrices (train, train+val, full)
        5. Build genome matrix (full mode only)
        6. Optionally fetch TMDB enrichment data
        7. Build enriched movies table and save as parquet
        8. Save all artefacts to PROCESSED_DIR
    Args:
        fetch_tmdb (bool): Whether to fetch TMDB metadata via the API.
            Requires a valid "tmdb_key".
        tmdb_key (str): TMDB API key. Required when "fetch_tmdb=True".
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    ratings = load_ratings()
    ratings = filter_ratings(ratings)
    movies = load_movies()
    movies = preprocess_movies(movies)
    tags = load_tags()
    links = load_links()
    genome_scores, genome_tags = load_genome()

    # Surviving movie/user IDs after filtering - used to align all other tables
    surviving_movie_ids = set(ratings["movieId"].unique())
    surviving_user_ids = set(ratings["userId"].unique())

    # Restrict all tables to surviving movies
    movies = movies[movies["movieId"].isin(surviving_movie_ids)]
    tags = tags[tags["movieId"].isin(surviving_movie_ids)] if len(tags) > 0 else tags
    links = (
        links[links["movieId"].isin(surviving_movie_ids)] if len(links) > 0 else links
    )

    logger.info(
        f"Aligned tables to filtered movies - "
        f"movies: {len(movies):,} | "
        f"tags: {len(tags):,} | "
        f"links: {len(links):,}"
    )

    # 2. Encode IDs
    ratings_enc, user_enc, movie_enc = encode_ids(ratings)

    n_users = len(user_enc["to_idx"])
    n_movies = len(movie_enc["to_idx"])

    # 3. Train / Val / Test split
    train_ratings, val_ratings, test_ratings = temporal_train_val_test_split(
        ratings_enc,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        min_ratings_per_user=MIN_RATINGS_PER_USER,
    )

    # 4. Sparse matrices
    # Train only - used during model fitting
    train_matrix = build_sparse_matrix(train_ratings, n_users, n_movies)

    # Train + Val combined - used for final retraining after hyperparameter tuning
    train_val_ratings = pd.concat([train_ratings, val_ratings]).reset_index(drop=True)
    train_val_matrix = build_sparse_matrix(train_val_ratings, n_users, n_movies)

    # Full matrix (all ratings) - used for production inference in the app
    full_matrix = build_sparse_matrix(ratings_enc, n_users, n_movies)

    # 5. Genome matrix
    genome_matrix = None
    if GENOME_SCORES_PATH is not None and len(genome_scores) > 0:
        # Filter genome scores to surviving movies only
        genome_scores = genome_scores[
            genome_scores["movieId"].isin(surviving_movie_ids)
        ]
        genome_matrix = build_genome_matrix(genome_scores, movie_enc)
        np.save(PROCESSED_DIR / "genome_matrix.npy", genome_matrix)
        genome_tags.to_parquet(PROCESSED_DIR / "genome_tags.parquet", index=False)
        logger.info("Genome matrix saved")

    # 6. TMDB enrichment
    tmdb_df = pd.DataFrame()
    if fetch_tmdb and LINKS_PATH is not None and len(links) > 0:
        from src.data_source.tmdb_fetcher import TMDBFetcher

        fetcher = TMDBFetcher(
            api_key=tmdb_key,
            cache_path=PROCESSED_DIR / "tmdb_cache.json",
        )
        fetcher.fetch_all(links)
        tmdb_df = fetcher.transform()
        logger.info(f"Fetched TMDB data for {len(tmdb_df)} movies")

    # 7. Build enriched movies table
    movies_enriched = movies.copy()
    movies_enriched["movie_idx"] = movies_enriched["movieId"].map(movie_enc["to_idx"])

    if len(links) > 0:
        movies_enriched = movies_enriched.merge(links, on="movieId", how="left")

    if len(tmdb_df) > 0:
        tmdb_df["tmdbId"] = pd.to_numeric(tmdb_df["tmdbId"], errors="coerce").astype(
            "Int64"
        )
        movies_enriched = movies_enriched.merge(tmdb_df, on="tmdbId", how="left")

    if len(tags) > 0:
        tag_agg = (
            tags.groupby("movieId")["tag"]
            .apply(lambda t: " ".join(t.dropna().astype(str)))
            .reset_index()
            .rename(columns={"tag": "user_tags_text"})
        )
        movies_enriched = movies_enriched.merge(tag_agg, on="movieId", how="left")

    movies_enriched.to_parquet(PROCESSED_DIR / "movies_enriched.parquet", index=False)
    logger.info(f"Saved enriched movies -> {len(movies_enriched)} rows")

    # 8. Save all artifacts
    save_artifacts(
        PROCESSED_DIR,
        user_enc=user_enc,
        movie_enc=movie_enc,
        train_ratings=train_ratings,
        val_ratings=val_ratings,
        test_ratings=test_ratings,
        train_val_ratings=train_val_ratings,
        train_matrix=train_matrix,
        train_val_matrix=train_val_matrix,
        full_matrix=full_matrix,
    )

    # 9. Summary
    logger.info("Pipeline complete")
    logger.info(f"Users: {n_users:,}")
    logger.info(f"Movies: {n_movies:,}")
    logger.info(
        f"Train ratings: {len(train_ratings):,}  ({len(train_ratings) / len(ratings_enc):.0%})"
    )
    logger.info(
        f"Val ratings: {len(val_ratings):,}  ({len(val_ratings) / len(ratings_enc):.0%})"
    )
    logger.info(
        f"Test ratings: {len(test_ratings):,}  ({len(test_ratings) / len(ratings_enc):.0%})"
    )
    logger.info(f"Genome matrix: {'yes' if genome_matrix is not None else 'no'}")
    logger.info(f"TMDB enriched: {'yes' if len(tmdb_df) > 0 else 'no'}")


if __name__ == "__main__":
    build(True)
