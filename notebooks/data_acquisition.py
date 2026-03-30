"""Data Acquisition module for the Indian Movies IMDB Research project.

Downloads and loads datasets from IMDB, Mendeley, GitHub Bollywood, and Kaggle
into pandas DataFrames with local caching in the data/ directory.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 13.1
"""

from __future__ import annotations

import gzip
import io
import logging
import shutil
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMDB_BASE_URL = "https://datasets.imdbws.com/"
IMDB_FILES = {
    "title.basics.tsv.gz": "title.basics.tsv.gz",
    "title.principals.tsv.gz": "title.principals.tsv.gz",
    "name.basics.tsv.gz": "name.basics.tsv.gz",
}

MENDELEY_DATASET_URL = (
    "https://data.mendeley.com/datasets/wcb4bxbyxx/2"
)

GITHUB_BOLLYWOOD_RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "devensinghbhagtani/Bollywood-Movie-Dataset/main/"
)

KAGGLE_DATASET_HANDLE = "nareshbhat/indian-moviesimdb"

DOWNLOAD_TIMEOUT = 120  # seconds
CHUNK_SIZE = 8192


class DataAcquisition:
    """Downloads and loads all four research datasets with local caching.

    Parameters
    ----------
    data_dir : str
        Path to the local cache directory. Created if it does not exist.
    """

    def __init__(self, data_dir: str = "data/") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataAcquisition initialised – cache dir: %s", self.data_dir)


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_file(self, url: str, dest: Path) -> bool:
        """Download *url* to *dest* with a tqdm progress bar.

        Returns ``True`` on success, ``False`` on any network error.
        """
        try:
            logger.info("Downloading %s …", url)
            resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                disable=total == 0,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)
                    bar.update(len(chunk))
            logger.info("Saved %s (%d bytes)", dest, dest.stat().st_size)
            return True
        except (requests.RequestException, OSError) as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            return False

    def _cached_or_download(self, filename: str, url: str) -> Path | None:
        """Return the local path if cached, otherwise download.

        Returns ``None`` when the download fails.
        """
        local = self.data_dir / filename
        if local.exists() and local.stat().st_size > 0:
            logger.info("Using cached file: %s", local)
            return local
        if self._download_file(url, local):
            return local
        return None

    def _read_imdb_tsv(self, gz_path: Path) -> pd.DataFrame:
        """Read a gzip-compressed IMDB TSV into a DataFrame."""
        logger.info("Reading %s …", gz_path)
        return pd.read_csv(
            gz_path,
            sep="\t",
            compression="gzip",
            low_memory=False,
            na_values=["\\N"],
        )

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_imdb_basics(self) -> pd.DataFrame:
        """Load IMDB ``title.basics.tsv.gz``.

        Columns: tconst, titleType, primaryTitle, originalTitle, isAdult,
        startYear, endYear, runtimeMinutes, genres.
        """
        filename = "title.basics.tsv.gz"
        url = IMDB_BASE_URL + filename
        path = self._cached_or_download(filename, url)
        if path is None:
            logger.warning("IMDB basics unavailable – returning empty DataFrame")
            return pd.DataFrame()
        return self._read_imdb_tsv(path)

    def load_imdb_principals(self) -> pd.DataFrame:
        """Load IMDB ``title.principals.tsv.gz``.

        Columns: tconst, ordering, nconst, category, job, characters.
        """
        filename = "title.principals.tsv.gz"
        url = IMDB_BASE_URL + filename
        path = self._cached_or_download(filename, url)
        if path is None:
            logger.warning("IMDB principals unavailable – returning empty DataFrame")
            return pd.DataFrame()
        return self._read_imdb_tsv(path)

    def load_imdb_names(self) -> pd.DataFrame:
        """Load IMDB ``name.basics.tsv.gz``.

        Columns: nconst, primaryName, birthYear, deathYear,
        primaryProfession, knownForTitles.
        """
        filename = "name.basics.tsv.gz"
        url = IMDB_BASE_URL + filename
        path = self._cached_or_download(filename, url)
        if path is None:
            logger.warning("IMDB names unavailable – returning empty DataFrame")
            return pd.DataFrame()
        return self._read_imdb_tsv(path)

    def load_mendeley_dataset(self) -> pd.DataFrame:
        """Load the Mendeley Indian movies dataset.

        Attempts to download the CSV from the Mendeley data repository.
        Falls back to any ``*mendeley*.csv`` already present in the cache dir.
        """
        filename = "mendeley_indian_movies.csv"
        local = self.data_dir / filename

        if local.exists() and local.stat().st_size > 0:
            logger.info("Using cached Mendeley file: %s", local)
            return pd.read_csv(local)

        # Mendeley data API – try the direct download endpoint
        download_url = (
            "https://data.mendeley.com/public-files/datasets/"
            "wcb4bxbyxx/files/"
            "IMDb-Indian-Movies.csv"
        )
        try:
            logger.info("Attempting Mendeley download …")
            resp = requests.get(
                download_url, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True
            )
            resp.raise_for_status()
            local.write_bytes(resp.content)
            logger.info("Saved Mendeley dataset to %s", local)
            return pd.read_csv(local)
        except (requests.RequestException, OSError) as exc:
            logger.warning("Mendeley download failed: %s", exc)

        # Fallback: look for any CSV with 'mendeley' in the name
        for p in self.data_dir.glob("*mendeley*.csv"):
            logger.info("Falling back to local file: %s", p)
            return pd.read_csv(p)

        logger.warning("Mendeley dataset unavailable – returning empty DataFrame")
        return pd.DataFrame()

    def load_github_bollywood(self) -> pd.DataFrame:
        """Load the Bollywood Movie Dataset from GitHub.

        Tries the raw CSV URL from the repository's main branch.
        """
        filename = "bollywood_movies.csv"
        # The repo stores the data in a CSV at the root
        csv_url = GITHUB_BOLLYWOOD_RAW_BASE + "Bollywood_Movie_Dataset.csv"
        local = self.data_dir / filename

        if local.exists() and local.stat().st_size > 0:
            logger.info("Using cached GitHub Bollywood file: %s", local)
            return pd.read_csv(local)

        try:
            logger.info("Downloading GitHub Bollywood dataset …")
            resp = requests.get(csv_url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            local.write_bytes(resp.content)
            logger.info("Saved GitHub Bollywood dataset to %s", local)
            return pd.read_csv(local)
        except (requests.RequestException, OSError) as exc:
            logger.warning("GitHub Bollywood download failed: %s", exc)

        # Fallback: look for any CSV with 'bollywood' in the name
        for p in self.data_dir.glob("*bollywood*.csv"):
            logger.info("Falling back to local file: %s", p)
            return pd.read_csv(p)

        logger.warning(
            "GitHub Bollywood dataset unavailable – returning empty DataFrame"
        )
        return pd.DataFrame()

    def load_kaggle_indian_movies(self) -> pd.DataFrame:
        """Load the Kaggle Indian Movies IMDB dataset via ``kagglehub``.

        Falls back to any ``*kaggle*.csv`` in the cache directory if the
        download fails (e.g. missing API credentials).
        """
        filename = "kaggle_indian_movies.csv"
        local = self.data_dir / filename

        if local.exists() and local.stat().st_size > 0:
            logger.info("Using cached Kaggle file: %s", local)
            return pd.read_csv(local)

        try:
            import kagglehub  # noqa: F811 – optional dependency

            logger.info("Downloading Kaggle dataset via kagglehub …")
            dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
            dataset_dir = Path(dataset_path)

            # Find the first CSV in the downloaded directory
            csv_files = list(dataset_dir.rglob("*.csv"))
            if not csv_files:
                logger.warning("No CSV found in Kaggle download at %s", dataset_dir)
                return pd.DataFrame()

            # Copy to cache and load
            shutil.copy2(csv_files[0], local)
            logger.info("Saved Kaggle dataset to %s", local)
            return pd.read_csv(local)
        except Exception as exc:  # noqa: BLE001 – broad catch intentional
            logger.warning("Kaggle download failed: %s", exc)

        # Fallback: look for any CSV with 'kaggle' in the name
        for p in self.data_dir.glob("*kaggle*.csv"):
            logger.info("Falling back to local file: %s", p)
            return pd.read_csv(p)

        logger.warning("Kaggle dataset unavailable – returning empty DataFrame")
        return pd.DataFrame()

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all datasets and return a dict of non-empty DataFrames.

        Keys: ``"imdb_basics"``, ``"imdb_principals"``, ``"imdb_names"``,
        ``"mendeley"``, ``"github_bollywood"``, ``"kaggle_indian_movies"``.

        Sources that fail to load or return empty DataFrames are omitted
        from the result and a warning is logged.
        """
        loaders: dict[str, callable] = {
            "imdb_basics": self.load_imdb_basics,
            "imdb_principals": self.load_imdb_principals,
            "imdb_names": self.load_imdb_names,
            "mendeley": self.load_mendeley_dataset,
            "github_bollywood": self.load_github_bollywood,
            "kaggle_indian_movies": self.load_kaggle_indian_movies,
        }

        results: dict[str, pd.DataFrame] = {}
        for name, loader in loaders.items():
            try:
                logger.info("Loading source: %s", name)
                df = loader()
                if df is not None and not df.empty:
                    results[name] = df
                    logger.info(
                        "  %s: %d rows, %d columns", name, len(df), len(df.columns)
                    )
                else:
                    logger.warning("  %s returned empty – skipped", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  %s failed: %s – skipped", name, exc)

        logger.info(
            "load_all complete – %d / %d sources loaded", len(results), len(loaders)
        )
        return results
