import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse, csr_matrix

from src.config.config import config
from src.logging_utils.logger import logger


ALL_GENRES = config.data_config.get("all_genres", [])

RANDOM_SEED = config.feature_builder_config.get("random_seed", 42)
NUMERICAL_FEATURES = config.feature_builder_config.get("numerical_features", [])

_cfg_tags = config.feature_builder_config_user_tags
USER_TAGS_N_COMPONENTS: int = _cfg_tags.get("n_components", 100)
USER_TAGS_MAX_FEATURES: int = _cfg_tags.get("max_features", 5000)
USER_TAGS_MIN_DF: int = _cfg_tags.get("min_df", 2)
USER_TAGS_MAX_DF: float = _cfg_tags.get("max_df", 0.95)
USER_TAGS_SUBLINEAR_TF: bool = _cfg_tags.get("sublinear_tf", True)

_cfg_overview_tfidf = config.feature_builder_config_overview_tfidf
OVERVIEW_TFIDF_N_COMPONENTS: int = _cfg_overview_tfidf.get("n_components", 100)
OVERVIEW_TFIDF_MAX_FEATURES: int = _cfg_overview_tfidf.get("max_features", 5000)
OVERVIEW_TFIDF_MIN_DF: int = _cfg_overview_tfidf.get("min_df", 2)
OVERVIEW_TFIDF_MAX_DF: float = _cfg_overview_tfidf.get("max_df", 0.95)
OVERVIEW_TFIDF_STOP_WORDS: str = _cfg_overview_tfidf.get("stop_words", "english")
OVERVIEW_TFIDF_SUBLINEAR_TF: bool = _cfg_overview_tfidf.get("sublinear_tf", True)

_cfg_sbert = config.feature_builder_config_overview_sbert
OVERVIEW_SBERT_MODEL_NAME: str = _cfg_sbert.get("model_name", "all-MiniLM-L6-v2")
OVERVIEW_SBERT_BATCH_SIZE: int = _cfg_sbert.get("batch_size", 64)

_cfg_director = config.feature_builder_config_director
DIRECTOR_N_COMPONENTS: int = _cfg_director.get("n_components", 50)

_cfg_cast = config.feature_builder_config_cast
CAST_N_COMPONENTS: int = _cfg_cast.get("n_components", 50)
CAST_TOP_N: int = _cfg_cast.get("top_n", 3)


class FeatureBuilder:
    """
    Builds and combines feature vectors for content-based similarity models.

    Each build_* method produces a (n_movies x dims) float32 numpy
    array aligned to movie_enc indices. Features can then be combined via
    build_combined() with optional per-group weights.

    Usage:
        builder = FeatureBuilder(movies_enriched, movie_enc, genome_matrix)

        # Build individual groups
        genome = builder.build_genome()
        genres = builder.build_genres()
        year = builder.build_year()
        tags = builder.build_user_tags(n_components=100)
        overview = builder.build_overview_tfidf(n_components=100)
        sbert = builder.build_overview_sbert()
        struct = builder.build_tmdb_structural(n_components=50)

        # Combine with weights
        vector = builder.build_combined([
            (genome, 1.0),
            (genres, 0.5),
            (year, 0.2),
        ])
    """

    def __init__(
        self,
        movies_enriched: pd.DataFrame,
        movie_enc: dict,
        genome_matrix: np.ndarray | None = None,
    ) -> None:
        """
        Initialise the FeatureBuilder with enriched movies, movie encoder, and optional genome matrix.
        Args:
            movies_enriched (pd.DataFrame): Enriched movies DataFrame with features and 'movie_idx'.
            movie_enc (dict): Movie encoder with 'to_idx' mapping movieId to index.
            genome_matrix (np.ndarray, optional): Pre-aligned genome scores (n_movies x 1128). If None, genome features will be zeros.
        """
        self.movie_enc = movie_enc
        self.genome_matrix = genome_matrix
        self.n_movies = len(movie_enc["to_idx"])

        # Align movies_enriched to encoded indices - one row per encoded movie
        self.movies = (
            movies_enriched.dropna(subset=["movie_idx"])
            .sort_values("movie_idx")
            .reset_index(drop=True)
        )

        # Ensure list-like columns are properly parsed (cast, genres_list, tmdb_genres)
        def parse_list_column(val: object) -> list:
            """
            Parse a column that may contain list-like data in various formats:
            Args:
                val (object): The value to parse, which can be a list, a string representation of a list, or NaN.
            Returns:
                list: A list of items if parsing is successful, or an empty list if the value is NaN or cannot be parsed.
            """
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return []
            return []

        # Parse list-like columns if they exist
        for col in ["cast", "genres_list", "tmdb_genres"]:
            if col in self.movies.columns:
                self.movies[col] = self.movies[col].apply(parse_list_column)

        logger.info(
            f"FeatureBuilder initialised - "
            f"{self.n_movies:,} movies | "
            f"genome: {'yes' if genome_matrix is not None else 'no'}"
        )

    # Helper methods for building and aligning feature matrices

    def _empty(self) -> np.ndarray:
        """
        Return a zero matrix when a feature group is unavailable.
        Returns:
            np.ndarray: (n_movies x 1) zero matrix.
        """
        return np.zeros((self.n_movies, 1), dtype=np.float32)

    def _aligned(self, values: np.ndarray, movie_idxs: np.ndarray) -> np.ndarray:
        """
        Place feature rows into a full (n_movies x dims) matrix using
        encoded movie indices. Movies missing from the source get zeros.
        Args:
            values (np.ndarray): Feature matrix for available movies.
            movie_idxs (np.ndarray): Encoded integer indices for each row in values.
        Returns:
            np.ndarray: (n_movies x dims) feature matrix aligned to all encoded movies.
        """
        n_dims = values.shape[1]
        out = np.zeros((self.n_movies, n_dims), dtype=np.float32)
        out[movie_idxs] = values.astype(np.float32)
        return out

    def _to_dense(self, matrix: np.ndarray | csr_matrix) -> np.ndarray:
        """
        Convert sparse matrix to dense if needed.
        Args:
            matrix (np.ndarray | csr_matrix): Sparse or dense matrix to convert.
        Returns:
            np.ndarray: Dense array representation.
        """
        return matrix.toarray() if issparse(matrix) else np.array(matrix)

    # Genome

    def build_genome(self) -> np.ndarray:
        """
        Return the genome matrix as-is (already 0-1 normalised).
        Movies without genome data have zero vectors.
        Returns:
            np.ndarray: (n_movies x 1128) genome tag relevance scores.
        """
        if self.genome_matrix is None:
            logger.warning("Genome matrix not provided - returning zeros")
            return self._empty()

        logger.info(f"Genome features: {self.genome_matrix.shape}")
        return self.genome_matrix.astype(np.float32)

    # Genres
    def build_genres(self) -> np.ndarray:
        """
        Multi-hot encode genres into a binary vector.
        Returns:
            np.ndarray: (n_movies x 19) binary genre indicators.
        """
        # Multi-hot encode genres using ALL_GENRES to ensure consistent ordering and dimensions
        mlb = MultiLabelBinarizer(classes=ALL_GENRES)

        # Ensure genres_list is a list of genres, even if originally missing or malformed
        genres = (
            self.movies["genres_list"]
            .fillna("")
            .apply(lambda g: g if isinstance(g, list) else [])
        )

        # Fit and transform genres
        encoded = mlb.fit_transform(genres).astype(np.float32)
        movie_idxs = self.movies["movie_idx"].values.astype(int)
        out = self._aligned(encoded, movie_idxs)

        logger.info(f"Genres features: {out.shape}")
        return out

    # Year
    def build_year(self) -> np.ndarray:
        """
        Normalise release year to [0, 1] using MinMaxScaler.
        Missing years are filled with the median.
        Returns:
            np.ndarray: (n_movies x 1) normalised year values.
        """
        # Extract years, fill missing with median, and reshape for scaler
        years = self.movies["year"].astype(float)
        median_year = years.median()
        years = years.fillna(median_year).values.reshape(-1, 1)

        year_scaler = MinMaxScaler()
        scaled = year_scaler.fit_transform(years).astype(np.float32)

        movie_idxs = self.movies["movie_idx"].values.astype(int)
        out = self._aligned(scaled, movie_idxs)

        logger.info(f"Year features: {out.shape}")
        return out

    # User tags TF-IDF
    def build_user_tags(self, n_components: int = USER_TAGS_N_COMPONENTS) -> np.ndarray:
        """
        TF-IDF on aggregated user tag text, reduced with TruncatedSVD.
        Movies with no tags get zero vectors.
        Args:
            n_components (int): Number of SVD components.
        Returns:
            np.ndarray: (n_movies x n_components) tag embedding matrix.
        """
        if "user_tags_text" not in self.movies.columns:
            logger.warning("user_tags_text not found - returning zeros")
            return self._empty()

        texts = self.movies["user_tags_text"].fillna("").tolist()

        # Fit TF-IDF on user tags
        tfidf_tags = TfidfVectorizer(
            max_features=USER_TAGS_MAX_FEATURES,
            min_df=USER_TAGS_MIN_DF,
            max_df=USER_TAGS_MAX_DF,
            sublinear_tf=USER_TAGS_SUBLINEAR_TF,
        )
        tfidf_matrix = tfidf_tags.fit_transform(texts)

        # Reduce dimensionality with SVD
        svd_tags = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        reduced = svd_tags.fit_transform(tfidf_matrix).astype(np.float32)

        movie_idxs = self.movies["movie_idx"].values.astype(int)
        out = self._aligned(reduced, movie_idxs)

        logger.info(
            f"User tags features: {out.shape} | "
            f"SVD explained variance: {svd_tags.explained_variance_ratio_.sum():.2%}"
        )
        return out

    # TMDB overview TF-IDF + SVD
    def build_overview_tfidf(
        self, n_components: int = OVERVIEW_TFIDF_N_COMPONENTS
    ) -> np.ndarray:
        """
        TF-IDF on TMDB movie overview, reduced with TruncatedSVD.
        Movies with no overview get zero vectors.
        Args:
            n_components (int): Number of SVD components.
        Returns:
            np.ndarray: (n_movies x n_components) overview embedding matrix.
        """
        if "overview" not in self.movies.columns:
            logger.warning("overview not found - returning zeros")
            return self._empty()

        texts = self.movies["overview"].fillna("").tolist()

        # Fit TF-IDF on overviews
        tfidf_overview = TfidfVectorizer(
            max_features=OVERVIEW_TFIDF_MAX_FEATURES,
            min_df=OVERVIEW_TFIDF_MIN_DF,
            max_df=OVERVIEW_TFIDF_MAX_DF,
            stop_words=OVERVIEW_TFIDF_STOP_WORDS,
            sublinear_tf=OVERVIEW_TFIDF_SUBLINEAR_TF,
        )
        tfidf_matrix = tfidf_overview.fit_transform(texts)

        # Reduce dimensionality with SVD
        svd_overview = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        reduced = svd_overview.fit_transform(tfidf_matrix).astype(np.float32)

        movie_idxs = self.movies["movie_idx"].values.astype(int)
        out = self._aligned(reduced, movie_idxs)

        logger.info(
            f"Overview TF-IDF features: {out.shape} | "
            f"SVD explained variance: {svd_overview.explained_variance_ratio_.sum():.2%}"
        )
        return out

    # TMDB overview SBERT embeddings
    def build_overview_sbert(
        self, model_name: str = OVERVIEW_SBERT_MODEL_NAME
    ) -> np.ndarray:
        """
        Dense semantic embeddings of TMDB overviews using a sentence transformer.
        Movies with no overview get zero vectors.
        Args:
            model_name (str): Sentence transformer model name.
        Returns:
            np.ndarray: (n_movies x embedding_dim) semantic embedding matrix.
        """
        from sentence_transformers import (
            SentenceTransformer,
        )  # local import to avoid unnecessary dependency if not used

        if "overview" not in self.movies.columns:
            logger.warning("overview not found - returning zeros")
            return self._empty()

        texts = self.movies["overview"].fillna("").tolist()

        logger.info(f"Encoding {len(texts):,} overviews with {model_name}...")
        sbert = SentenceTransformer(model_name)
        embeddings = sbert.encode(
            texts,
            batch_size=OVERVIEW_SBERT_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        movie_idxs = self.movies["movie_idx"].values.astype(int)
        out = self._aligned(embeddings, movie_idxs)

        logger.info(f"SBERT overview features shape: {out.shape}")
        return out

    # TMDB director SVD
    def build_director(self, n_components: int = DIRECTOR_N_COMPONENTS) -> np.ndarray:
        """
        Multi-hot encode directors, reduced with TruncatedSVD.
        Args:
            n_components (int): Number of SVD components to keep.
        Returns:
            np.ndarray: (n_movies x n_components) director embedding matrix.
        """
        if "director" not in self.movies.columns:
            logger.warning("director column not found - returning zeros")
            return self._empty()

        # Multi-hot encode directors with a "dir_" prefix to distinguish from cast members
        director_lists = (
            self.movies["director"]
            .apply(
                lambda d: (
                    [f"dir_{d.lower().replace(' ', '_')}"] if isinstance(d, str) else []
                )
            )
            .tolist()
        )

        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(director_lists).astype(np.float32)

        n_components = min(n_components, encoded.shape[1] - 1)

        # Reduce dimensionality with SVD
        svd_director = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        reduced = svd_director.fit_transform(encoded).astype(np.float32)

        movie_idxs = self.movies["movie_idx"].values.astype(int)
        out = self._aligned(reduced, movie_idxs)

        logger.info(
            f"Director features shape: {out.shape} | "
            f"SVD explained variance: {svd_director.explained_variance_ratio_.sum():.2%}"
        )
        return out

    # TMDB cast SVD
    def build_cast(
        self, n_components: int = CAST_N_COMPONENTS, top_n: int = CAST_TOP_N
    ) -> np.ndarray:
        """
        Multi-hot encode top-N cast members, reduced with TruncatedSVD.
        Shape: (n_movies x n_components)
        Movies with no cast data get zero vectors.
        Args:
            n_components (int): Number of SVD components to keep.
            top_n (int): Only keep top N billed cast members per movie.
        """
        if "cast" not in self.movies.columns:
            logger.warning("cast column not found - returning zeros")
            return self._empty()

        # Multi-hot encode top-N cast members with a "cast_" prefix to distinguish from directors
        cast_lists = (
            self.movies["cast"]
            .apply(
                lambda c: [
                    f"cast_{name.lower().replace(' ', '_')}"
                    for name in (c[:top_n] if isinstance(c, list) else [])
                ]
            )
            .tolist()
        )

        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(cast_lists).astype(np.float32)

        n_components = min(n_components, encoded.shape[1] - 1)

        # Reduce dimensionality with SVD
        svd_cast = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        reduced = svd_cast.fit_transform(encoded).astype(np.float32)

        movie_idxs = self.movies["movie_idx"].values.astype(int)
        out = self._aligned(reduced, movie_idxs)

        logger.info(
            f"Cast features shape: {out.shape} | "
            f"SVD explained variance: {svd_cast.explained_variance_ratio_.sum():.2%}"
        )
        return out

    # TMDB numerical features (runtime, vote_average, popularity)

    def build_tmdb_numerical(self) -> np.ndarray:
        """
        Normalise TMDB numerical features: runtime, vote_average, popularity.
        Missing values filled with column medians.
        Returns:
            np.ndarray: (n_movies x 3) normalised numerical features.
        """
        # Columns to use if available
        cols = NUMERICAL_FEATURES
        available = [c for c in cols if c in self.movies.columns]

        if not available:
            logger.warning("No TMDB numerical columns found - returning zeros")
            return self._empty()

        df = self.movies[available].copy()
        for col in available:
            df[col] = df[col].fillna(df[col].median())

        num_scaler = MinMaxScaler()
        scaled = num_scaler.fit_transform(df.values).astype(np.float32)

        movie_idxs = self.movies["movie_idx"].values.astype(int)
        out = self._aligned(scaled, movie_idxs)

        logger.info(f"TMDB numerical features shape: {out.shape} | cols: {available}")
        return out

    # Combined features
    def build_combined(self, groups: list[tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Concatenate multiple feature groups with per-group weights.
        Each group is L2-normalised before weighting so no single
        high-dimensional group dominates by scale alone.
        Args:
            groups (list[tuple[np.ndarray, float]]): List of (feature_matrix, weight) pairs to combine.
        Returns:
            np.ndarray: (n_movies x total_dims) combined float32 matrix.
        Example:
            vector = builder.build_combined([
                (genome,  1.0),
                (genres,  0.5),
                (year,    0.2),
            ])
        """
        weighted = []
        # L2 normalise each group and apply weight
        for matrix, weight in groups:
            # L2 normalise each group independently
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
            normalised = (matrix / norms) * weight
            weighted.append(normalised)

        combined = np.concatenate(weighted, axis=1).astype(np.float32)
        logger.info(f"Combined feature vector shape: {combined.shape}")
        return combined


if __name__ == "__main__":
    import pickle
    import numpy as np
    import pandas as pd
    from pathlib import Path

    PROCESSED_DIR = Path("data/processed/full")

    movies_enriched = pd.read_parquet(PROCESSED_DIR / "movies_enriched.parquet")
    movie_enc = pickle.load(open(PROCESSED_DIR / "movie_enc.pkl", "rb"))
    genome_matrix = np.load(PROCESSED_DIR / "genome_matrix.npy")

    builder = FeatureBuilder(movies_enriched, movie_enc, genome_matrix)

    genome = builder.build_genome()
    genres = builder.build_genres()
    year = builder.build_year()
    tags = builder.build_user_tags()
    overview = builder.build_overview_tfidf()
    sbert = builder.build_overview_sbert()
    director = builder.build_director()
    cast = builder.build_cast()
    numerical = builder.build_tmdb_numerical()

    combined = builder.build_combined(
        [
            (genome, 1.0),
            (genres, 0.5),
            (year, 0.2),
            (tags, 0.5),
            (sbert, 0.8),
            (director, 0.8),
            (cast, 0.3),
        ]
    )

    print(f"genome: {genome.shape}")
    print(f"genres: {genres.shape}")
    print(f"year: {year.shape}")
    print(f"tags: {tags.shape}")
    print(f"overview: {overview.shape}")
    print(f"sbert: {sbert.shape}")
    print(f"director: {director.shape}")
    print(f"cast: {cast.shape}")
    print(f"numerical: {numerical.shape}")
    print(f"combined: {combined.shape}")
