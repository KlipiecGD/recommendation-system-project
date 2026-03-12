import pandas as pd

from src.logging_utils.logger import logger


def temporal_train_val_test_split(
    ratings: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_ratings_per_user: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per-user temporal split into train / val / test.
    Strategy:
    - Sort each user's ratings by timestamp
    - Last `test_ratio` fraction goes to test
    - Next `val_ratio` fraction goes to validation
    - Remainder goes to training
    - Users with < min_ratings_per_user go entirely to train (not enough to split)
    Args:
        ratings (pd.DataFrame): DataFrame with columns "userId", "movieId",
            "rating", and "timestamp".
        val_ratio (float): Fraction of each user's ratings to allocate to validation set.
        test_ratio (float): Fraction of each user's ratings to allocate to test set.
        min_ratings_per_user (int): Minimum number of ratings a user must have to be split;
            otherwise all their ratings go to the training set.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test DataFrames.
    """
    # Sort ratings by user and timestamp to ensure temporal splitting
    ratings_sorted = ratings.sort_values(["userId", "timestamp"])

    # Initialize lists to collect train/val/test splits
    train_list, val_list, test_list = [], [], []
    skipped = 0

    for _, group in ratings_sorted.groupby("userId"):
        n = len(group)

        # If user has too few ratings, put all in train
        if n < min_ratings_per_user:
            train_list.append(group)
            skipped += 1
            continue

        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        # Append splits for this user
        train_list.append(group.iloc[: -(n_val + n_test)])
        val_list.append(group.iloc[-(n_val + n_test) : -n_test])
        test_list.append(group.iloc[-n_test:])

    # Concatenate all user splits into final DataFrames
    train = pd.concat(train_list).reset_index(drop=True)
    val = pd.concat(val_list).reset_index(drop=True) if val_list else pd.DataFrame()
    test = pd.concat(test_list).reset_index(drop=True) if test_list else pd.DataFrame()

    logger.info(
        f"Split - Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,} | "
        f"Users skipped (sparse): {skipped:,}"
    )
    return train, val, test


if __name__ == "__main__":
    from src.preprocessing.loaders import load_ratings

    ratings_df = load_ratings()
    train_df, val_df, test_df = temporal_train_val_test_split(ratings_df)
    logger.info("Sample train ratings:\n" + train_df.head().to_string(index=False))
    logger.info("Sample val ratings:\n" + val_df.head().to_string(index=False))
    logger.info("Sample test ratings:\n" + test_df.head().to_string(index=False))
