import numpy as np
import pandas as pd
from pathlib import Path

from src.models.content_based.base_cb_model import BaseCBModel
from src.models.content_based.feature_builder import FeatureBuilder
from src.logging_utils.logger import logger
from src.config.config import config

_cfg = config.get_specific_model_config("cb3_genome_genres")
MODEL_NAME: str = _cfg.get("model_name", "CB3_GenomeGenres")
TOP_K: int = _cfg.get("top_k", 100)
METRIC: str = _cfg.get("metric", "cosine")
BATCH_SIZE: int = _cfg.get("batch_size", 512)
PROFILE_STRATEGY: str = _cfg.get("profile_strategy", "mean_centering")

_weights = _cfg.get("weights", {})
W_GENOME: float = _weights.get("genome", 1.0)
W_GENRES: float = _weights.get("genres", 0.5)
W_YEAR: float = _weights.get("year", 0.2)


class CB3GenomeGenres(BaseCBModel):
    """
    Content-based recommender using genome + genres + year (CB-3).
    Features: G1 (genome, 1128-dim) + G2 (genres, 19-dim) + G3 (year, 1-dim).
    Embedding: weighted concatenation - genome weight scaled down so genres
                and year are not drowned by the high-dimensional genome.
    Similarity: configurable (cosine default).
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        metric: str = METRIC,
        profile_strategy: str = PROFILE_STRATEGY,
    ) -> None:
        """
        Initialise the CB3GenomeGenres model.
        Args:
            top_k (int): Number of top similar movies to precompute for each movie.
            metric (str): Similarity metric to use ('cosine' or 'euclidean').
            profile_strategy (str): User profile aggregation strategy ('weighted' or 'mean_centering').
        """
        super().__init__(
            model_name=MODEL_NAME,
            top_k=top_k,
            metric=metric,
            profile_strategy=profile_strategy,
        )

    def fit(
        self,
        movies_enriched: pd.DataFrame,
        movie_enc: dict,
        genome_matrix: np.ndarray,
    ) -> None:
        """
        Args:
            movies_enriched (pd.DataFrame): Enriched movies DataFrame.
            movie_enc (dict): Encoder dict with 'to_idx' mapping.
            genome_matrix (np.ndarray): (n_movies x 1128) genome scores.
        """
        logger.info(f"Fitting {self.model_name} (metric={self.metric})...")

        builder = FeatureBuilder(movies_enriched, movie_enc, genome_matrix)
        self.feature_matrix = builder.build_combined(
            [
                (builder.build_genome(), W_GENOME),
                (builder.build_genres(), W_GENRES),
                (builder.build_year(), W_YEAR),
            ]
        )

        self._build_lookups(movies_enriched)
        self._precompute_topk(batch_size=BATCH_SIZE)
        self.is_fitted = True
        logger.info(
            f"{self.model_name} fitted - {self.feature_matrix.shape[0]:,} movies"
        )


if __name__ == "__main__":
    import pickle

    PROCESSED_DIR = Path("data/processed/full")

    movies_enriched = pd.read_parquet(PROCESSED_DIR / "movies_enriched.parquet")
    movie_enc = pickle.load(open(PROCESSED_DIR / "movie_enc.pkl", "rb"))
    genome_matrix = np.load(PROCESSED_DIR / "genome_matrix.npy")

    model = CB3GenomeGenres()
    model.fit(movies_enriched, movie_enc, genome_matrix)
    model.save(Path("artifacts/cb3_genome_genres"))

    print("\nSimilar to: Toy Story (1995)")
    print("-" * 50)
    for r in model.similar_movies("Toy Story (1995)", n=5):
        print(f"  {r['title']:45s} score: {r['score']:.4f}")
