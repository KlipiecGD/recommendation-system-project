import requests
import pandas as pd
import time
import json
import threading
import concurrent.futures
from pathlib import Path
from typing import Optional

from src.config.config import config
from src.logging_utils.logger import logger

TMDB_BASE = config.tmdb_config.get("tmdb_base_url", "https://api.themoviedb.org/3")
TMDB_IMG_BASE = config.tmdb_config.get(
    "tmdb_img_base_url", "https://image.tmdb.org/t/p/"
)


class TMDBFetcher:
    """
    Fetches and transforms movie metadata from the TMDB API.
    Caches raw API responses in a JSON file to avoid redundant calls and speed up development.
        - fetch_all(links_df): Fetch raw JSON for all movies in a links DataFrame
        - transform(): Transform the in-memory cache of raw JSON into a clean DataFrame with selected features
        - fetch_and_transform(links_df): Convenience method to do both in one call (main entry point for build_dataset.py)
    Usage:
        fetcher = TMDBFetcher(api_key="...", cache_path="data/processed/full/tmdb_cache.json")
        df = fetcher.fetch_and_transform(links_df)

        # Or step by step:
        fetcher.fetch_all(links_df)
        df = fetcher.transform()
    """

    def __init__(
        self,
        api_key: str,
        cache_path: str,
        top_cast: int = 5,
        image_size: str = "w500",
    ) -> None:
        """
        Initialize the TMDBFetcher.
        Args:
            api_key (str): TMDB API key.
            cache_path (str): Path to JSON file for caching raw API responses.
            top_cast (int): Number of top cast members to include in the transformed data.
            image_size (str): Size of poster images to fetch (w92/w185/w342/w500/w780/original).
        """
        self.api_key = api_key
        self.cache_path = Path(cache_path)
        self.top_cast = top_cast
        self.img_base = TMDB_IMG_BASE + image_size

        self.session = requests.Session()  # Use a session for connection pooling

        self._cache: dict = self._load_cache()

    # Caching methods
    def _load_cache(self) -> dict:
        """
        Load the cache from disk if it exists, otherwise return an empty dict.
        Returns:
            dict: Mapping of tmdbId (as string) to raw API response (dict) or None if fetch failed.
        """
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached entries from {self.cache_path}")
            return cache
        return {}

    def _save_cache(self) -> None:
        """
        Save the in-memory cache to disk as JSON.
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f)

    @property
    def cache_size(self) -> int:
        """
        Number of entries in the cache (both successful and failed).
        Returns:
            int: Total number of cached entries.
        """
        return len(self._cache)

    @property
    def failed_ids(self) -> list[int]:
        """
        TMDBIds where fetch was attempted but returned None.
        Returns:
            list[int]: List of tmdbIds that failed to fetch.
        """
        return [int(k) for k, v in self._cache.items() if v is None]

    @property
    def successful_ids(self) -> list[int]:
        """
        TMDBIds where fetch was successful (raw response is not None).
        Returns:
            list[int]: List of tmdbIds that were successfully fetched.
        """
        return [int(k) for k, v in self._cache.items() if v is not None]

    # Fetching methods
    def _get_raw(self, tmdb_id: int) -> dict | None:
        """
        Single raw TMDB API call. Returns raw JSON or None on failure.
        Args:
            tmdb_id (int): TMDB movie ID to fetch.
        Returns:
            dict | None: Raw API response as dict, or None if fetch failed.
        """
        url = f"{TMDB_BASE}/movie/{tmdb_id}"
        params = {"api_key": self.api_key, "append_to_response": "credits"}

        for attempt in range(3):  # Retry up to 3 times on failure
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 1))
                    logger.warning(
                        f"Rate limited on tmdbId={tmdb_id}. "
                        f"Attempt {attempt + 1}/3. Retrying after {retry_after}s..."
                    )
                    time.sleep(retry_after)
                elif resp.status_code == 404:
                    logger.debug(f"tmdbId={tmdb_id} not found (404)")
                else:
                    logger.warning(
                        f"tmdbId={tmdb_id} returned status {resp.status_code}"
                    )
            except requests.exceptions.Timeout:
                logger.warning(f"tmdbId={tmdb_id} timed out")
            except requests.exceptions.RequestException as e:
                logger.warning(f"tmdbId={tmdb_id} request error: {e}")
            return None

    def fetch_all(
        self,
        links: pd.DataFrame,
        max_movies: Optional[int] = None,
        retry_failed: bool = False,
    ) -> None:
        """
        Fetch raw TMDB JSON for all movies in a links DataFrame concurrently.
        Skips already-cached entries - safe to call multiple times to resume.
        Args:
            links (pd.DataFrame): DataFrame with a 'tmdbId' column to fetch.
            max_movies (int, optional): If set, limits the number of movies to fetch (for testing purposes).
            retry_failed (bool): If True, will retry fetching movies that previously failed (where cache value is None).
        """
        links_valid = links.dropna(subset=["tmdbId"]).copy()
        links_valid["tmdbId"] = links_valid["tmdbId"].astype(int)
        if max_movies:
            links_valid = links_valid.head(max_movies)

        total = len(links_valid)
        fetched = 0

        cache_lock = threading.Lock()  # To synchronize cache access across threads

        logger.info(
            f"Starting concurrent TMDB fetch for {total} movies | retry_failed={retry_failed}"
        )

        def _fetch_worker(tmbd_id_str: str) -> bool:
            with cache_lock:
                already_cached = tmbd_id_str in self._cache
                is_failed = self._cache.get(tmbd_id_str) is None

            if already_cached and not (retry_failed and is_failed):
                return False  # Skip if already cached and not retrying failed

            raw_data = self._get_raw(int(tmbd_id_str))
            with cache_lock:
                self._cache[tmbd_id_str] = raw_data
            return True

        # Spin up 40 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            future_to_id = {
                executor.submit(_fetch_worker, str(int(row["tmdbId"]))): str(
                    int(row["tmdbId"])
                )
                for _, row in links_valid.iterrows()
            }

            for future in concurrent.futures.as_completed(future_to_id):
                try:
                    was_fetched = future.result()
                    if was_fetched:
                        with cache_lock:
                            fetched += 1
                            if fetched % 5000 == 0:
                                logger.info(
                                    f"Fetched {fetched} new movies | cache size: {self.cache_size}"
                                )
                                self._save_cache()

                except Exception as exc:
                    logger.error(f"Movie fetch generated an exception: {exc}")

        # Final save when all threads are complete
        with cache_lock:
            self._save_cache()

        logger.info(
            f"Fetch complete - {self.cache_size} total | "
            f"{len(self.successful_ids)} ok | {len(self.failed_ids)} failed"
        )

    def fetch_one(self, tmdb_id: int, force: bool = False) -> dict | None:
        """
        Fetch and cache a single movie. Useful for on-demand lookups in the app.
        Args:
            tmdb_id (int): TMDB movie ID to fetch.
            force (bool): If True, will re-fetch from API even if already cached.
        Returns:
            dict | None: Raw API response as dict, or None if fetch failed.
        """
        key = str(tmdb_id)
        if key not in self._cache or force:
            self._cache[key] = self._get_raw(tmdb_id)
            self._save_cache()
        return self._cache.get(key)

    # Transformation methods
    def _extract_cast(self, credits: dict) -> list[str]:
        """
        Extract the top N cast members' names from the credits dict.
        Args:
            credits (dict): The 'credits' section of the TMDB API response.
        Returns:
            list[str]: List of top cast member names, up to self.top_cast.
        """
        return [c["name"] for c in credits.get("cast", [])[: self.top_cast]]

    def _extract_director(self, credits: dict) -> str | None:
        """
        Extract the director's name from the credits dict.
        Args:
            credits (dict): The 'credits' section of the TMDB API response.
        Returns:
            str | None: Director's name, or None if not found.
        """
        return next(
            (c["name"] for c in credits.get("crew", []) if c["job"] == "Director"), None
        )

    def _extract_genres(self, raw: dict) -> list[str]:
        """
        Extract genre names from the raw API response.
        Args:
            raw (dict): The raw API response.
        Returns:
            list[str]: List of genre names.
        """
        return [g["name"] for g in raw.get("genres", [])]

    def _transform_one(self, tmdb_id: int, raw: dict) -> dict:
        """
        Transform a single raw API response into a clean flat record.
        Args:
            tmdb_id (int): The TMDB ID of the movie.
            raw (dict): The raw API response for that movie.
        Returns:
            dict: A flat dictionary with selected and transformed fields.
        """
        credits = raw.get("credits", {})
        poster = raw.get("poster_path")
        return {
            "tmdbId": tmdb_id,
            "overview": raw.get("overview"),
            "poster_url": f"{self.img_base}{poster}" if poster else None,
            "release_date": raw.get("release_date"),
            "runtime_min": raw.get("runtime"),
            "vote_average": raw.get("vote_average"),
            "vote_count": raw.get("vote_count"),
            "popularity": raw.get("popularity"),
            "original_language": raw.get("original_language"),
            "tmdb_genres": self._extract_genres(raw),
            "cast": self._extract_cast(credits),
            "director": self._extract_director(credits),
        }

    def transform(self) -> pd.DataFrame:
        """
        Transform the full in-memory cache into a clean DataFrame.
        Skips entries where fetch failed (None).
        Returns:
            pd.DataFrame: DataFrame where each row is a movie with transformed metadata.
        """
        records = []
        skipped = 0
        for tmdb_id_str, raw in self._cache.items():
            if raw is None:
                skipped += 1
                continue
            records.append(self._transform_one(int(tmdb_id_str), raw))

        logger.info(f"Transformed {len(records)} records | skipped {skipped} failed")
        return pd.DataFrame(records)

    # Convenience method to fetch and transform in one call
    def fetch_and_transform(
        self,
        links: pd.DataFrame,
        max_movies: Optional[int] = None,
        retry_failed: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch then transform in one call. Main entry point for build_dataset.py.
        Args:
            links (pd.DataFrame): DataFrame with a 'tmdbId' column to fetch.
            max_movies (int, optional): If set, limits the number of movies to fetch (for testing purposes).
            retry_failed (bool): If True, will retry fetching movies that previously failed (where cache value is None).
        Returns:
            pd.DataFrame: Transformed DataFrame with one row per movie.
        """
        self.fetch_all(links, max_movies=max_movies, retry_failed=retry_failed)
        return self.transform()


if __name__ == "__main__":
    # Example usage for testing
    import os
    from dotenv import load_dotenv

    load_dotenv()

    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
    fetcher = TMDBFetcher(
        api_key=TMDB_API_KEY,
        cache_path="data/processed/dev/tmdb_cache.json",
        top_cast=5,
        image_size="w500",
    )
    # Example: fetch and transform using a sample links DataFrame
    sample_links = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "tmdbId": [862, 8844, 15602],  # Toy Story, Jumanji, Grumpier Old Men
        }
    )
    df = fetcher.fetch_and_transform(sample_links)
    df.to_csv("data/processed/dev/tmdb_sample.csv", index=False)
    print(df.head())
