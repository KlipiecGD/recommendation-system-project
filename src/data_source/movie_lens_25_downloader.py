import zipfile
from pathlib import Path

import requests

from src.config.config import config
from src.logging_utils.logger import logger

ML25M_URL = config.data_config.get(
    "data_source_link", "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
)


def download_ml25m(url: str = ML25M_URL) -> None:
    """
    Download the MovieLens 25M dataset, extract it to data/raw/, and remove the zip.
    Equivalent to:
        wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
        unzip ml-25m.zip && rm ml-25m.zip
    Args:
        url (str): URL to download the dataset from.
    """
    # Set up paths
    raw_dir = Path(config.project_root) / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "ml-25m.zip"

    # Download with progress logging
    logger.info(f"Downloading MovieLens 25M from {url} ...")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        logger.info(
                            f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.1f}%)"
                        )

    logger.info(f"Download complete: {zip_path}")

    # Extract zip
    logger.info(f"Extracting {zip_path} to {raw_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    logger.info("Extraction complete.")

    # Remove zip
    zip_path.unlink()
    logger.info(f"Removed {zip_path}")


if __name__ == "__main__":
    download_ml25m()
