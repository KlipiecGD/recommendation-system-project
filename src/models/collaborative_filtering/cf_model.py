from pathlib import Path

import pandas as pd
from surprise import SVD, SVDpp, NMF, Dataset, Reader

from src.models.collaborative_filtering.algorithm_registry import (
    ALGORITHM_REGISTRY,
    DEFAULT_PARAMS,
)
from src.models.base_model import BaseModel
from src.logging_utils.logger import logger


class CFModel(BaseModel):
    """
    Collaborative filtering wrapper around Surprise algorithms.
    Inherits save(), load(), _check_fitted(), __repr__() from BaseModel.

    Primary interface differs from CB models — recommendations require
    a userId present in training data, not a title->rating dict.
    Use recommend(user_id, movies_df, n) as the main entry point.

    recommend_from_ratings() is implemented as required by BaseModel
    but raises NotImplementedError since CF has a different contract.
    """

    def __init__(self, algo_key: str, params: dict | None = None) -> None:
        """
        Initialize the CF model with a specific algorithm and parameters.
        Args:
            algo_key (str): Key of the algorithm to use, e.g. 'svd', 'nmf', etc.
            params (dict, optional): Algorithm-specific parameters to override defaults.
        """
        if algo_key not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{algo_key}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
        super().__init__(model_name=f"CF_{algo_key.upper()}")

        self.algo_key = algo_key

        algo_params = {**DEFAULT_PARAMS.get(algo_key, {}), **(params or {})}
        self._algo = ALGORITHM_REGISTRY[algo_key](**algo_params)

        self._trainset = None
        self._rated_by_user = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit the model on a ratings DataFrame.
        Args:
            train_df (pd.DataFrame): Must contain 'userId', 'movieId', 'rating'.
                Raw IDs expected (not encoded indices).
        """
        logger.info(
            f"Fitting {self.model_name} on "
            f"{len(train_df):,} ratings | "
            f"{train_df['userId'].nunique():,} users | "
            f"{train_df['movieId'].nunique():,} movies"
        )

        reader = Reader(rating_scale=(0.5, 5.0))
        dataset = Dataset.load_from_df(
            train_df[["userId", "movieId", "rating"]], reader
        )
        self._trainset = dataset.build_full_trainset()

        self._rated_by_user = train_df.groupby("userId")["movieId"].apply(set).to_dict()

        self._algo.fit(self._trainset)
        self.is_fitted = True
        logger.info(f"{self.model_name} fitted successfully")

    def recommend_from_ratings(
        self,
        user_ratings: dict[str, float],
        n: int = 10,
    ) -> list[dict]:
        """
        Required by BaseModel but not applicable to CF models.
        CF requires a userId, not a title->rating dict.
        Use recommend(user_id, movies_df, n) instead.
        """
        raise NotImplementedError(
            f"{self.model_name} is a collaborative filtering model and requires "
            "a userId present in training data. "
            "Use recommend(user_id, movies_df, n) instead."
        )

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating a user would give a specific movie.
        Args:
            user_id (int): Raw userId.
            movie_id (int): Raw movieId.
        Returns:
            float: Predicted rating in [0.5, 5.0].
        """
        self._check_fitted()
        prediction = self._algo.predict(user_id, movie_id)
        return prediction.est

    def recommend(
        self,
        user_id: int,
        movies_df: pd.DataFrame,
        n: int = 10,
        filter_rated: bool = True,
    ) -> list[dict]:
        """
        Recommend top-N movies for a given userId.
        Scores all movies in movies_df, optionally masks already-rated
        ones, and returns top-N sorted by predicted rating descending.
        Args:
            user_id (int): Raw userId present in training data.
            movies_df (pd.DataFrame): Must contain 'movieId' and 'title'.
            n (int): Number of recommendations to return.
            filter_rated (bool): If True, exclude already-rated movies.
        Returns:
            list[dict]: Each dict has 'movieId', 'title', 'predicted_rating'. Sorted by predicted_rating descending.
        """
        self._check_fitted()

        already_rated = self._rated_by_user.get(user_id, set())

        candidates = movies_df.copy()
        if filter_rated:
            candidates = candidates[~candidates["movieId"].isin(already_rated)]

        if candidates.empty:
            logger.warning(
                f"No candidate movies left for user {user_id} after filtering."
            )
            return []

        predictions = [
            {
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "predicted_rating": self._algo.predict(
                    user_id, int(row["movieId"])
                ).est,
            }
            for _, row in candidates.iterrows()
        ]

        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return predictions[:n]

    @classmethod
    def load_cf(cls, artifact_dir: Path, algo_key: str) -> "CFModel":
        """
        Convenience wrapper around BaseModel.load() for CF models.
        Args:
            artifact_dir (Path): Directory where the artifact was saved.
            algo_key (str): e.g. 'svd' — reconstructs 'CF_SVD.pkl'.
        Returns:
            CFModel: The loaded model instance.
        """
        return cls.load(artifact_dir, f"CF_{algo_key.upper()}")
