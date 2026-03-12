import pandas as pd

from src.logging_utils.logger import logger


def encode_ids(ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Re-index userId and movieId to contiguous 0-based integers.
    Args:
        ratings (pd.DataFrame): The ratings DataFrame containing 'userId' and 'movieId'
    Returns:
        tuple[pd.DataFrame, dict, dict]: (encoded_df, user_encoder, movie_encoder)
    """
    # Get unique users and movies
    unique_users = sorted(ratings["userId"].unique())
    unique_movies = sorted(ratings["movieId"].unique())

    # Create mapping dicts for users and movies
    user2idx = {uid: i for i, uid in enumerate(unique_users)}
    movie2idx = {mid: i for i, mid in enumerate(unique_movies)}

    # Apply mappings to create new columns in the DataFrame
    encoded_df = ratings.copy()
    encoded_df["user_idx"] = encoded_df["userId"].map(user2idx)
    encoded_df["movie_idx"] = encoded_df["movieId"].map(movie2idx)

    # Create encoder dicts with reverse mappings
    user_enc = {"to_idx": user2idx, "to_id": {i: uid for uid, i in user2idx.items()}}
    movie_enc = {"to_idx": movie2idx, "to_id": {i: mid for mid, i in movie2idx.items()}}

    logger.info(
        f"Encoded {len(unique_users):,} users and {len(unique_movies):,} movies"
    )

    return encoded_df, user_enc, movie_enc


if __name__ == "__main__":
    # Example usage
    from src.preprocessing.loaders import load_ratings

    ratings_df = load_ratings()
    encoded_df, user_enc, movie_enc = encode_ids(ratings_df)
    print(encoded_df.head())
    print("User encoder example:", list(user_enc["to_idx"].items())[:5])
    print("Movie encoder example:", list(movie_enc["to_idx"].items())[:5])
