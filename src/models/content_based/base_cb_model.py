import numpy as np
import pandas as pd
from abc import abstractmethod

from src.models.base_model import BaseModel
from src.logging_utils.logger import logger
from src.config.config import config

STRATEGIES = config.model_config.get("strategies", ["weighted", "mean_centering"])
METRICS = config.model_config.get("metrics", ["cosine", "euclidean"])


class BaseCBModel(BaseModel):
    """
    Shared infrastructure for all content-based models.
    Subclasses only need to implement fit() to populate self.feature_matrix
    and call self._build_lookups() / self._precompute_topk() at the end.
    Handles:
        - Top-K precomputation (cosine or euclidean, batched)
        - recommend_from_ratings() with two profile-building strategies:
            * "weighted" — proportional to raw ratings (sum-normalised).
              Every rated movie pulls the profile toward its content; higher
              rating = stronger pull. Equivalent to the original behaviour.
            * "mean_centering" — weights are (rating - user_mean_rating).
              Movies rated above average attract the profile; movies rated
              below average actively repel it.
        - similar_movies()
    """

    def __init__(
        self,
        model_name: str,
        top_k: int = 100,
        metric: str = "cosine",
        profile_strategy: str = "mean_centering",
    ) -> None:
        """
        Args:
            model_name (str): Unique model identifier used for logging and serialisation.
            top_k (int): Number of pre-computed nearest neighbours per movie.
            metric (str): 'cosine' or 'euclidean'.
            profile_strategy (str): How to aggregate rated-movie vectors into a user
                profile. 'weighted' uses proportional raw ratings; 'mean_centering'
                uses mean-centered ratings so low-rated films repel the profile.
        """
        if metric not in METRICS:
            raise ValueError(f"metric must be in {METRICS}, got '{metric}'")
        if profile_strategy not in STRATEGIES:
            raise ValueError(
                f"profile_strategy must be in {STRATEGIES}, got '{profile_strategy}'"
            )
        super().__init__(model_name=model_name)
        self.top_k = top_k
        self.metric = metric
        self.profile_strategy = profile_strategy
        self.feature_matrix: np.ndarray | None = None
        self._norm_matrix: np.ndarray | None = None  # cosine only
        self.topk_indices: np.ndarray | None = None
        self.topk_scores: np.ndarray | None = None
        self.title_to_idx: dict = {}
        self.idx_to_title: dict = {}

    # Abstract method to be implemented by subclasses
    @abstractmethod
    def fit(self, *args, **kwargs) -> None: ...

    # Helper methods for fitting and precomputation
    def _build_lookups(self, movies_enriched: pd.DataFrame) -> None:
        """
        Build title <-> encoded-index lookup dicts from movies_enriched.
        Args:
            movies_enriched (pd.DataFrame): Enriched movies DataFrame with 'movie_idx' and 'title' columns.
        """
        valid = movies_enriched.dropna(subset=["movie_idx"]).copy()
        valid["movie_idx"] = valid["movie_idx"].astype(int)
        self.title_to_idx = dict(zip(valid["title"], valid["movie_idx"]))
        self.idx_to_title = dict(zip(valid["movie_idx"], valid["title"]))

    def _precompute_topk(self, batch_size: int = 512) -> None:
        """
        Precompute top-K similar movies for every movie.
        Cosine: row-normalised dot product -> score in [−1, 1]
        Euclidean: negative squared L2 distance -> score less than 0
        Args:
            batch_size (int): Batch size for similarity computation to manage memory usage.
        """
        n_movies = self.feature_matrix.shape[0]
        self.topk_indices = np.zeros((n_movies, self.top_k), dtype=np.int32)
        self.topk_scores = np.zeros((n_movies, self.top_k), dtype=np.float32)

        if self.metric == "cosine":
            # Precompute row-normalised feature matrix for cosine similarity
            norms = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
            # Avoid division by zero for zero vectors by setting their norm to 1 (they will have zero similarity with all movies)
            norms = np.where(norms == 0, 1.0, norms)
            self._norm_matrix = (self.feature_matrix / norms).astype(np.float32)
            sq_norms = None
        else:
            self._norm_matrix = None
            sq_norms = np.sum(self.feature_matrix**2, axis=1)

        logger.info(
            f"Precomputing top-{self.top_k} neighbours "
            f"({self.metric}) for {n_movies:,} movies..."
        )

        for start in range(0, n_movies, batch_size):
            end = min(start + batch_size, n_movies)

            if self.metric == "cosine":
                sims = self._norm_matrix[start:end] @ self._norm_matrix.T
            else:
                batch = self.feature_matrix[start:end]
                sims = -(
                    sq_norms[start:end, None]
                    + sq_norms[None, :]
                    - 2.0 * (batch @ self.feature_matrix.T)
                )

            for i in range(end - start):
                sims[i, start + i] = -np.inf

            self.top_k = min(self.top_k, n_movies - 1)
            top_k_idx = np.argpartition(sims, -self.top_k, axis=1)[:, -self.top_k :]
            for i, idx in enumerate(top_k_idx):
                sorted_idx = idx[np.argsort(sims[i, idx])[::-1]]
                self.topk_indices[start + i] = sorted_idx
                self.topk_scores[start + i] = sims[i, sorted_idx]

            if (start // batch_size) % 5 == 0:
                logger.info(f"Precompute progress: {end}/{n_movies}")

        logger.info("Precompute complete")

    # Profile-building strategies
    def _profile_weighted(
        self,
        rated_indices: list[int],
        ratings: list[float],
    ) -> np.ndarray:
        """
        Build a user profile as a rating-proportional weighted average of
        rated movie feature vectors. Every movie pulls the profile toward its
        content; higher rating = stronger pull.
        Args:
            rated_indices (list[int]): Encoded movie indices the user has rated.
            ratings (list[float]): Corresponding raw ratings (e.g. 0.5–5.0).
        Returns:
            np.ndarray: (dims,) profile vector, not yet L2-normalised.
        """
        w = np.array(ratings, dtype=np.float32)
        w /= w.sum()

        profile = np.zeros(self.feature_matrix.shape[1], dtype=np.float32)
        for idx, wi in zip(rated_indices, w):
            profile += self.feature_matrix[idx] * wi

        return profile

    def _profile_mean_centering(
        self,
        rated_indices: list[int],
        ratings: list[float],
    ) -> np.ndarray:
        """
        Build a user profile using mean-centered ratings as weights.
        Movies rated above the user's average attract the profile;
        movies rated below actively repel it.
        If all ratings are identical, all weights are 0 and we fall
        back to uniform weighting (equivalent to plain average).
        Args:
            rated_indices (list[int]): Encoded movie indices the user has rated.
            ratings (list[float]): Corresponding raw ratings (e.g. 0.5–5.0).
        Returns:
            np.ndarray: (dims,) profile vector, not yet L2-normalised.
        """
        r = np.array(ratings, dtype=np.float32)
        mu = r.mean()
        w = r - mu

        # Fallback: all ratings identical -> treat as uniform average
        if np.allclose(w, 0.0):
            logger.debug(
                "Mean-centering: all ratings identical, falling back to uniform average."
            )
            w = np.ones_like(r) / len(r)

        profile = np.zeros(self.feature_matrix.shape[1], dtype=np.float32)
        for idx, wi in zip(rated_indices, w):
            profile += self.feature_matrix[idx] * wi

        return profile

    # Recommendation methods
    def recommend_from_ratings(
        self,
        user_ratings: dict[str, float],
        n: int = 10,
    ) -> list[dict]:
        """
        Recommend movies based on a user-provided title -> rating dict.
        Args:
            user_ratings (dict[str, float]): Movie title -> rating (e.g. 1–5).
            n (int): Number of recommendations to return.
        Returns:
            list[dict]: List of dicts with keys 'movie_idx', 'title', 'score'.
        """
        self._check_fitted()
        if not user_ratings:
            logger.warning("No ratings provided - returning empty list")
            return []

        rated_indices, weights = [], []
        for title, rating in user_ratings.items():
            idx = self.title_to_idx.get(title)
            if idx is None:
                logger.warning(f"Title not found: '{title}' - skipping")
                continue
            rated_indices.append(idx)
            weights.append(rating)

        if not rated_indices:
            logger.warning(
                "None of the provided titles were found - returning empty list"
            )
            return []

        # Build user profile using the configured strategy
        if self.profile_strategy == "mean_centering":
            profile = self._profile_mean_centering(rated_indices, weights)
        else:
            profile = self._profile_weighted(rated_indices, weights)

        if self.metric == "cosine":
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm
            scores = self._norm_matrix @ profile
        else:
            sq_norms = np.sum(self.feature_matrix**2, axis=1)
            scores = -(
                sq_norms + np.sum(profile**2) - 2.0 * (self.feature_matrix @ profile)
            )

        scores[rated_indices] = -np.inf

        top_n_idx = np.argpartition(scores, -n)[-n:]
        top_n_idx = top_n_idx[np.argsort(scores[top_n_idx])[::-1]]

        return [
            {
                "movie_idx": int(idx),
                "title": self.idx_to_title.get(int(idx), "Unknown"),
                "score": float(scores[idx]),
            }
            for idx in top_n_idx
        ]

    def similar_movies(self, title: str, n: int = 10) -> list[dict]:
        """
        Return top-n most similar movies to a given title.
        Args:
            title (str): Movie title to find neighbours for.
            n (int): Number of similar movies to return.
        Returns:
            list[dict]: List of dicts with keys 'movie_idx', 'title', 'score'.
        """
        self._check_fitted()
        movie_idx = self.title_to_idx.get(title)
        if movie_idx is None:
            logger.warning(f"Title not found: '{title}'")
            return []

        indices = self.topk_indices[movie_idx, :n]
        scores = self.topk_scores[movie_idx, :n]

        return [
            {
                "movie_idx": int(idx),
                "title": self.idx_to_title.get(int(idx), "Unknown"),
                "score": float(score),
            }
            for idx, score in zip(indices, scores)
        ]
