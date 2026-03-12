import pickle

from pathlib import Path

from src.logging_utils.logger import logger


def save_artifacts(processed_dir: Path, **kwargs) -> None:
    """
    Save any dict of named object to processed_dir as pickle
    Args:
        processed_dir (Path): Directory to save artifacts
        **kwargs: Named objects to save as pickle files
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in kwargs.items():
        path = processed_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved artifact '{name}' to {path}")


def load_artifact(processed_dir: Path, name: str) -> object:
    """
    Load a named artifact from processed_dir as pickle
    Args:
        processed_dir (Path): Directory to load artifact from
        name (str): Name of the artifact to load (without .pkl extension)
    Returns:
        object: The loaded artifact object
    """
    path = processed_dir / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Artifact '{name}' not found at {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded artifact '{name}' from {path}")
    return obj
