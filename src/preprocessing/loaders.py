import pandas as pd
from pathlib import Path

from src.config.config import config
from src.logging_utils.logger import logger


_data_cfg: dict = config.data_config

RAW_DIR: Path = config.project_root / _data_cfg["raw_dir"]
PROCESSED_DIR: Path = config.project_root / _data_cfg["processed_dir"]
ARTIFACTS_DIR: Path = config.project_root / _data_cfg["artifacts_dir"]

RATINGS_PATH: Path = RAW_DIR / _data_cfg["ratings_file"]
MOVIES_PATH: Path = RAW_DIR / _data_cfg["movies_file"]
USERS_PATH: Path = RAW_DIR / _data_cfg["users_file"]
TAGS_PATH: Path = RAW_DIR / _data_cfg["tags_file"]
LINKS_PATH: Path = RAW_DIR / _data_cfg["links_file"]
GENOME_SCORES_PATH: Path = RAW_DIR / _data_cfg["genome_scores_file"]
GENOME_TAGS_PATH: Path = RAW_DIR / _data_cfg["genome_tags_file"]

ENCODING: str = _data_cfg["encoding"]


def load_ratings() -> pd.DataFrame:
    """
    Load ratings from ml-25m (ratings.csv).
    Returns:
        pd.DataFrame: DataFrame with columns "userId", "movieId",
            "rating", "timestamp".
    """
    logger.info(f"Loading ratings from {RATINGS_PATH}")
    df = pd.read_csv(RATINGS_PATH, encoding=ENCODING)
    logger.info(
        f"Loaded {len(df):,} ratings | {df['userId'].nunique():,} users | {df['movieId'].nunique():,} movies"
    )
    return df


def filter_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out inactive users and unpopular movies.
    Removes users with fewer than "min_user_ratings" ratings and movies
    with fewer than "min_movie_ratings" ratings. Thresholds are read from
    config.
    Args:
        ratings (pd.DataFrame): Raw ratings DataFrame with columns "userId", "movieId".
    Returns:
        pd.DataFrame: Filtered ratings DataFrame.
    """
    min_user_ratings = config.data_config.get("min_user_ratings", 20)
    min_movie_ratings = config.data_config.get("min_movie_ratings", 50)

    logger.info(
        f"Filtering ratings - "
        f"min_user_ratings={min_user_ratings} | "
        f"min_movie_ratings={min_movie_ratings}"
    )
    logger.info(
        f"Before filtering - "
        f"{ratings['userId'].nunique():,} users | "
        f"{ratings['movieId'].nunique():,} movies | "
        f"{len(ratings):,} ratings"
    )

    active_users = ratings.groupby("userId").size()
    ratings = ratings[
        ratings["userId"].isin(active_users[active_users >= min_user_ratings].index)
    ]

    popular_movies = ratings.groupby("movieId").size()
    ratings = ratings[
        ratings["movieId"].isin(
            popular_movies[popular_movies >= min_movie_ratings].index
        )
    ]

    logger.info(
        f"After filtering - "
        f"{ratings['userId'].nunique():,} users | "
        f"{ratings['movieId'].nunique():,} movies | "
        f"{len(ratings):,} ratings"
    )
    return ratings


def load_movies() -> pd.DataFrame:
    """
    Load raw movies from ml-25m (movies.csv).
    Returns:
        pd.DataFrame: DataFrame with columns "movieId", "title", "genres".
    """
    logger.info(f"Loading movies from {MOVIES_PATH}")
    df = pd.read_csv(MOVIES_PATH, encoding=ENCODING)
    logger.info(f"Loaded {len(df):,} movies")
    return df


def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a raw movies DataFrame.
    - Extracts the release year from the title field (e.g. "Toy Story (1995)").
    - Produces a cleaned title without the year suffix.
    - Splits the pipe-separated "genres" column into a Python list.
    Args:
        df (pd.DataFrame): Raw movies DataFrame with columns "movieId", "title", "genres".
    Returns:
        pd.DataFrame: DataFrame with additional columns "year" (nullable Int64),
            "title_clean", and "genres_list".
    """
    df = df.copy()

    # Extract year from title e.g. "Toy Story (1995)"
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int64")
    df["title_clean"] = (
        df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    )

    # Genres as list — treat "(no genres listed)" as empty
    df["genres_list"] = df["genres"].str.split("|")
    df["genres_list"] = df["genres_list"].apply(
        lambda g: [] if g == ["(no genres listed)"] else g
    )

    return df


def load_tags() -> pd.DataFrame:
    """
    Load user-applied tags from ml-25m (tags.csv).
    Tag text is lower-cased and stripped of surrounding whitespace.
    Returns:
        pd.DataFrame: DataFrame with columns "userId", "movieId", "tag",
            "timestamp".
    """
    logger.info(f"Loading tags from {TAGS_PATH}")
    df = pd.read_csv(TAGS_PATH, encoding=ENCODING)
    df["tag"] = df["tag"].fillna("").str.lower().str.strip()
    return df


def load_links() -> pd.DataFrame:
    """
    Load TMDB / IMDB links from ml-25m (links.csv).
    Returns:
        pd.DataFrame: DataFrame with columns "movieId", "imdbId" (str),
            "tmdbId" (nullable Int64).
    """
    logger.info(f"Loading links from {LINKS_PATH}")
    df = pd.read_csv(LINKS_PATH, encoding=ENCODING, dtype={"imdbId": str})
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce").astype("Int64")
    return df


def load_genome() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load genome relevance scores and genome tag labels from ml-25m.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A "(genome_scores, genome_tags)"
            pair where "genome_scores" has columns "movieId", "tagId",
            "relevance" and "genome_tags" has columns "tagId", "tag".
    """

    logger.info(f"Loading genome tags from {GENOME_TAGS_PATH}")
    genome_tags = pd.read_csv(GENOME_TAGS_PATH)

    logger.info(f"Loading genome scores from {GENOME_SCORES_PATH}")
    genome_scores = pd.read_csv(GENOME_SCORES_PATH)

    logger.info(
        f"Genome: {len(genome_tags):,} tags | {genome_scores['movieId'].nunique():,} movies scored"
    )
    return genome_scores, genome_tags


if __name__ == "__main__":
    # Quick test to verify loading works without errors and prints expected info
    ratings_df = load_ratings()
    ratings_df = filter_ratings(ratings_df)
    movies_df = preprocess_movies(load_movies())
    tags_df = load_tags()
    links_df = load_links()
    genome_scores_df, genome_tags_df = load_genome()

    print("Ratings:")
    print(ratings_df.head())
    print("Movies:")
    print(movies_df.head())
    print("Tags:")
    print(tags_df.head())
    print("Links:")
    print(links_df.head())
    print("Genome Scores:")
    print(genome_scores_df.head())
    print("Genome Tags:")
    print(genome_tags_df.head())
