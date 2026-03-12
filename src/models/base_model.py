from abc import ABC, abstractmethod
from pathlib import Path
import pickle

from src.logging_utils.logger import logger


class BaseModel(ABC):
    """
    Abstract base class for all recommender models.
    Subclasses must implement:
        - fit()
        - recommend_from_ratings()
    Subclasses may override:
        - save() / load() if they need custom serialisation
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialize the model.
        Args:
            model_name (str): A unique name for the model, used in logging and artifact naming.
        """
        self.model_name = model_name
        self.is_fitted = False

    # Required abstract methods
    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """
        Train or prepare the model.
        Must set self.is_fitted = True on completion.
        """
        ...

    @abstractmethod
    def recommend_from_ratings(
        self,
        user_ratings: dict[str, float],
        n: int = 10,
    ) -> list[dict]:
        """
        Recommend movies based on a user-provided title -> rating dict.
        Args:
            user_ratings (dict[str, float]): A dict mapping movie titles to user ratings.
            n (int): The number of recommendations to return.
        Returns:
            list[dict]: A list of recommended movies, each represented as a dict with keys like "title", "predicted_rating", etc.
        """
        ...

    # Default save/load implementations using pickle

    def save(self, artifact_dir: Path) -> None:
        """
        Persist model to disk.
        Default implementation pickles the entire object.
        Override for models with non-picklable state.
        Args:
            artifact_dir (Path): Directory to save model artifact into.
        """
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{self.model_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved {self.model_name} -> {path}")

    @classmethod
    def load(cls, artifact_dir: Path, model_name: str) -> "BaseModel":
        """
        Load a previously saved model from disk.
        Default implementation unpickles the object.
        Override for models with non-picklable state.
        Args:
            artifact_dir (Path): Directory where the artifact was saved.
            model_name (str): Must match the name used when saving.
        Returns:
            BaseModel: The loaded model instance.
        """
        path = artifact_dir / f"{model_name}.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded {model_name} from {path}")
        return model

    # Helper methods
    def _check_fitted(self) -> None:
        """Raise if recommend() is called before fit()."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.model_name} is not fitted yet. Call fit() first."
            )

    def __repr__(self) -> str:
        """String representation showing model name and fitted status."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.model_name}', status={status})"
