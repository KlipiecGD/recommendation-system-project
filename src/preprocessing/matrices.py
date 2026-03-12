import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from src.logging_utils.logger import logger


def build_sparse_matrix(
    ratings: pd.DataFrame, n_users: int, n_movies: int
) -> csr_matrix:
    """
    Build user-movie sparse matrix from encoded ratings DataFrame.
    Args:
        ratings (pd.DataFrame): DataFrame with columns "user_idx", "movie_idx", and "rating".
        n_users (int): Total number of unique users.
        n_movies (int): Total number of unique movies.
    Returns:
        csr_matrix: Sparse matrix of shape (n_users, n_movies) with ratings as values.
    """
    matrix = csr_matrix(
        (
            ratings["rating"].astype(np.float32),
            (ratings["user_idx"], ratings["movie_idx"]),
        ),
        shape=(n_users, n_movies),
    )
    sparsity = 1.0 - (matrix.count_nonzero() / (n_users * n_movies))
    logger.info(
        f"Built sparse matrix with shape {matrix.shape} and sparsity {sparsity:.4f}"
    )
    return matrix


def build_genome_matrix(genome_scores: pd.DataFrame, movies_enc: dict) -> np.ndarray:
    """
    Build a (n_movies x 1128 (number of genome tags)) dense genome tag matrix.
    Rows are aligned to movie_enc indices. Movies without genome data get zeros.
    Args:
        genome_scores (pd.DataFrame): DataFrame with columns "movieId", "tagId", and "relevance".
        movies_enc (dict): Movie encoder dict with 'to_idx' mapping original movieId to new index.
    Returns:
        np.ndarray: Dense matrix of shape (n_movies, 1128) with relevance scores.
    """
    # Get number of movies and tags
    n_movies = len(movies_enc["to_idx"])
    n_tags = genome_scores["tagId"].nunique()

    logger.info(f"Building genome matrix for {n_movies} movies and {n_tags} tags")

    # Initialize the genome matrix with zeros
    genome_matrix = np.zeros((n_movies, n_tags), dtype=np.float32)

    # Pivot: movieId -> tagId -> relevance
    pivot = genome_scores.pivot(index="movieId", columns="tagId", values="relevance")

    # Fill the genome matrix using the pivot table and movie encoder
    for movie_id, idx in movies_enc["to_idx"].items():
        if movie_id in pivot.index:
            genome_matrix[idx] = pivot.loc[movie_id].values

    covered_movies = np.sum(genome_matrix.sum(axis=1) > 0)
    logger.info(
        f"Filled genome matrix. {covered_movies}/{n_movies} movies have genome data."
    )
    return genome_matrix


if __name__ == "__main__":
    # Example usage of building the sparse matrix
    from src.preprocessing.loaders import load_ratings
    from src.preprocessing.encoding import encode_ids

    ratings_df = load_ratings()
    encoded_df, user_enc, movie_enc = encode_ids(ratings_df)
    n_users = len(user_enc["to_idx"])
    n_movies = len(movie_enc["to_idx"])
    sparse_matrix = build_sparse_matrix(encoded_df, n_users, n_movies)

    # Example usage of building the genome matrix
    from src.preprocessing.loaders import load_genome

    genome_scores_df, genome_tags_df = load_genome()
    genome_matrix = build_genome_matrix(genome_scores_df, movie_enc)
