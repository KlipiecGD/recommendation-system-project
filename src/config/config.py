import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """Configuration class that loads and provides access to config values."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize configuration from YAML file.

        Args:
            config_path (str): Path to the config YAML file
        """
        # Determine the location of this file (src/config/config.py)
        current_file = Path(__file__)

        # Calculate Project Root: src/config/ -> src/ -> Project Root
        self.project_root = current_file.parent.parent.parent

        if config_path:
            config_file = Path(config_path)
        else:
            # Dynamically get the path relative to this python file
            config_file = current_file.parent / "config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get a configuration value using dot notation.

        Args:
            key_path (str): Dot-separated path to config value (e.g., "loader.document_path")
            default: Default value if key not found

        Returns:
                Configuration value or default

        """
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def data_config(self) -> dict:
        """
        Get the data configuration section (ml-25m settings).
        Returns:
            dict: The data configuration section from the configuration.
        """
        return self._config.get("data_config", {})

    @property
    def tmdb_config(self) -> dict:
        """
        Get the TMDB API configuration section.
        Returns:
            dict: The TMDB API configuration section from the configuration.
        """
        return self._config.get("tmdb_api", {})

    @property
    def ui_config(self) -> dict:
        """
        Get the UI configuration section.
        Returns:
            dict: The UI configuration section from the configuration.
        """
        return self._config.get("ui", {})

    @property
    def model_config(self) -> dict:
        """
        Get the model configuration section.
        Returns:
            dict: The model configuration section from the configuration.
        """
        return self._config.get("models", {})

    def get_specific_model_config(self, model_key: str) -> dict:
        """
        Get the configuration for any model by its config key.
        Args:
            model_key (str): Key under models in config.yaml (e.g. 'cb2_genres_year').
        Returns:
            dict: The model configuration section.
        """
        return self.model_config.get(model_key, {})

    @property
    def feature_builder_config(self) -> dict:
        """
        Get the feature builder configuration section.
        Returns:
            dict: The feature builder configuration section from the configuration.
        """
        return self._config.get("feature_builder", {})

    @property
    def feature_builder_config_user_tags(self) -> dict:
        """
        Get the configuration for the user tags feature builder.
        Returns:
            dict: The configuration for the user tags feature builder.
        """
        feature_builder_config = self.feature_builder_config
        return feature_builder_config.get("user_tags", {})

    @property
    def feature_builder_config_director(self) -> dict:
        """
        Get the configuration for the director feature builder.
        Returns:
            dict: The configuration for the director feature builder.
        """
        feature_builder_config = self.feature_builder_config
        return feature_builder_config.get("director", {})

    @property
    def feature_builder_config_cast(self) -> dict:
        """
        Get the configuration for the cast feature builder.
        Returns:
            dict: The configuration for the cast feature builder.
        """
        feature_builder_config = self.feature_builder_config
        return feature_builder_config.get("cast", {})

    @property
    def feature_builder_config_overview_tfidf(self) -> dict:
        """
        Get the configuration for the overview TF-IDF feature builder.
        Returns:
            dict: The configuration for the overview TF-IDF feature builder.
        """
        feature_builder_config = self.feature_builder_config
        return feature_builder_config.get("overview_tfidf", {})

    @property
    def feature_builder_config_overview_sbert(self) -> dict:
        """
        Get the configuration for the overview SBERT feature builder.
        Returns:
            dict: The configuration for the overview SBERT feature builder.
        """
        feature_builder_config = self.feature_builder_config
        return feature_builder_config.get("overview_sbert", {})

    @property
    def random_seed(self) -> int:
        """
        Get the random seed from the feature builder configuration.
        Returns:
            int: The random seed value.
        """
        feature_builder_config = self.feature_builder_config
        return feature_builder_config.get("random_seed", 42)

    @property
    def evaluation_config(self) -> dict:
        """
        Get the evaluation configuration section.
        Returns:
            dict: The evaluation configuration section from the configuration.
        """
        return self._config.get("evaluation", {})


config = Config()
