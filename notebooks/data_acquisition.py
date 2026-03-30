"""Data Acquisition module for the Indian Movies IMDB Research project.

Downloads and loads datasets from IMDB, TIMDB (Indian movies), GitHub
Bollywood, and Kaggle into pandas DataFrames with local caching.

Key features:
- Resumable: cached files are reused on re-run; only failed/missing sources retry
- System gunzip: uses OS gunzip command for reliable .gz decompression
- Verbose logging: chunk-level progress for large IMDB files
- Chunked reading with early row filtering to stay within 12 GB RAM

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 13.1
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMDB_BASE_URL = "https://datasets.imdbws.com/"

MENDELEY_DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "pncnmnp/TIMDB/master/1950-2019/bollywood_full.csv"
)

GITHUB_BOLLYWOOD_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "devensinghbhagtani/Bollywood-Movie-Dataset/main/"
    "IMDB-Movie-Dataset(2023-1951).csv"
)

KAGGLE_DATASET_HANDLE = "nareshbhat/indian-moviesimdb"

DOWNLOAD_TIMEOUT = 300  # seconds (large IMDB files need more time)
CHUNK_SIZE = 8192


class DataAcquisition:
    """Downloads and loads all research datasets with local caching.

    Designed to be **resumable**: every downloaded or processed file is
    cached locally.  Re-running skips anything already on disk, so only
    failed or missing sources are retried.

    Parameters
    ----------
    data_dir : str
        Path to the local cache directory.  Created if it does not exist.
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
            downloaded = 0
            with open(dest, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                disable=total == 0,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    bar.update(len(chunk))
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info("Saved %s (%.1f MB)", dest, size_mb)
            return True
        except (requests.RequestException, OSError) as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            # Remove partial file
            if dest.exists():
                dest.unlink()
            return False

    def _cached_or_download(self, filename: str, url: str) -> Path | None:
        """Return the local path if cached, otherwise download.

        Returns ``None`` when the download fails.
        """
        local = self.data_dir / filename
        if local.exists() and local.stat().st_size > 0:
            size_mb = local.stat().st_size / (1024 * 1024)
            logger.info("Using cached file: %s (%.1f MB)", local, size_mb)
            return local
        if self._download_file(url, local):
            return local
        return None

    def _decompress_gz(self, gz_path: Path) -> Path:
        """Decompress a .gz file using system gunzip, returning the TSV path.

        If the decompressed file already exists, skip decompression.
        Uses system ``gunzip`` command for reliability over Python gzip.
        """
        tsv_path = gz_path.with_suffix("")  # e.g. title.basics.tsv
        if tsv_path.exists() and tsv_path.stat().st_size > 0:
            size_mb = tsv_path.stat().st_size / (1024 * 1024)
            logger.info(
                "Decompressed file already exists: %s (%.1f MB)", tsv_path, size_mb
            )
            return tsv_path

        logger.info("Decompressing %s using system gunzip …", gz_path)
        try:
            # gunzip -k keeps the original .gz file
            result = subprocess.run(
                ["gunzip", "-k", "-f", str(gz_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.warning("gunzip failed: %s", result.stderr)
                raise RuntimeError(f"gunzip failed: {result.stderr}")
        except FileNotFoundError:
            # gunzip not available, fall back to Python gzip
            logger.info("gunzip not found, falling back to Python gzip …")
            import gzip

            with gzip.open(gz_path, "rb") as f_in, open(tsv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        size_mb = tsv_path.stat().st_size / (1024 * 1024)
        logger.info("Decompressed to %s (%.1f MB)", tsv_path, size_mb)
        return tsv_path

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

    def _read_imdb_tsv_chunked(
        self,
        gz_path: Path,
        usecols: list[str] | None = None,
        filter_fn=None,
        chunksize: int = 100_000,
    ) -> pd.DataFrame:
        """Read an IMDB TSV in chunks with early filtering.

        Decompresses .gz files using system gunzip first, then reads the
        plain TSV in chunks.  This avoids Python gzip memory issues and
        provides better progress logging.
        """
        # Decompress to plain TSV first
        tsv_path = self._decompress_gz(gz_path)

        # Count total lines for progress reporting
        logger.info("Counting lines in %s …", tsv_path.name)
        total_lines = 0
        with open(tsv_path, "r") as f:
            for _ in f:
                total_lines += 1
        total_lines -= 1  # subtract header
        logger.info("  %s has %s data rows", tsv_path.name, f"{total_lines:,}")

        logger.info(
            "Reading %s in chunks of %s rows …", tsv_path.name, f"{chunksize:,}"
        )
        chunks: list[pd.DataFrame] = []
        rows_read = 0
        rows_kept = 0
        chunk_num = 0

        reader = pd.read_csv(
            tsv_path,
            sep="\t",
            low_memory=False,
            na_values=["\\N"],
            usecols=usecols,
            chunksize=chunksize,
        )
        for chunk in reader:
            chunk_num += 1
            rows_read += len(chunk)

            if filter_fn is not None:
                chunk = filter_fn(chunk)

            if not chunk.empty:
                rows_kept += len(chunk)
                chunks.append(chunk)

            # Log progress every 10 chunks
            if chunk_num % 10 == 0:
                pct = (rows_read / total_lines * 100) if total_lines > 0 else 0
                logger.info(
                    "  chunk %d: %s / %s rows read (%.0f%%), %s kept so far",
                    chunk_num,
                    f"{rows_read:,}",
                    f"{total_lines:,}",
                    pct,
                    f"{rows_kept:,}",
                )

        if not chunks:
            logger.info("  → 0 rows after filtering")
            return pd.DataFrame(columns=usecols or [])

        result = pd.concat(chunks, ignore_index=True)
        logger.info(
            "  → %s rows kept out of %s read (%d chunks)",
            f"{len(result):,}",
            f"{rows_read:,}",
            chunk_num,
        )
        return result


    # ------------------------------------------------------------------
    # Public loaders — each checks for a PROCESSED parquet cache first,
    # then falls back to raw download + chunked read.
    # ------------------------------------------------------------------

    def load_imdb_basics(self) -> pd.DataFrame:
        """Load IMDB ``title.basics.tsv.gz``, filtered to movies only.

        Cache chain: parquet → raw .tsv.gz (download if needed) → chunked read.
        """
        parquet = self.data_dir / "imdb_basics_movies.parquet"
        if parquet.exists() and parquet.stat().st_size > 0:
            logger.info("Loading cached IMDB basics from %s", parquet)
            df = pd.read_parquet(parquet)
            logger.info("  → %s rows from parquet cache", f"{len(df):,}")
            return df

        filename = "title.basics.tsv.gz"
        url = IMDB_BASE_URL + filename
        path = self._cached_or_download(filename, url)
        if path is None:
            logger.warning("IMDB basics unavailable – returning empty DataFrame")
            return pd.DataFrame()

        def _filter_movies(chunk: pd.DataFrame) -> pd.DataFrame:
            return chunk[chunk["titleType"] == "movie"]

        df = self._read_imdb_tsv_chunked(path, filter_fn=_filter_movies)

        # Coerce mixed-type numeric columns before parquet save
        if not df.empty:
            for col in ("startYear", "endYear", "runtimeMinutes", "isAdult"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.to_parquet(parquet, index=False)
            logger.info("Cached IMDB basics (movies) to %s", parquet)

        return df

    def load_imdb_principals(
        self, valid_tconsts: set[str] | None = None
    ) -> pd.DataFrame:
        """Load IMDB ``title.principals.tsv.gz``.

        If *valid_tconsts* is provided, filters to those tconsts and
        actor/actress/director categories during read.

        Cache chain: parquet → raw .tsv.gz → chunked read.
        """
        parquet = self.data_dir / "imdb_principals_filtered.parquet"
        if parquet.exists() and parquet.stat().st_size > 0:
            logger.info("Loading cached IMDB principals from %s", parquet)
            df = pd.read_parquet(parquet)
            logger.info("  → %s rows from parquet cache", f"{len(df):,}")
            return df

        filename = "title.principals.tsv.gz"
        url = IMDB_BASE_URL + filename
        path = self._cached_or_download(filename, url)
        if path is None:
            logger.warning("IMDB principals unavailable – returning empty DataFrame")
            return pd.DataFrame()

        if valid_tconsts:
            tconst_set = valid_tconsts  # closure reference

            def _filter_tconsts(chunk: pd.DataFrame) -> pd.DataFrame:
                filtered = chunk[chunk["tconst"].isin(tconst_set)]
                if "category" in filtered.columns:
                    filtered = filtered[
                        filtered["category"].isin({"actor", "actress", "director"})
                    ]
                return filtered

            df = self._read_imdb_tsv_chunked(path, filter_fn=_filter_tconsts)
        else:
            df = self._read_imdb_tsv_chunked(path)

        # Coerce mixed-type columns before parquet save
        if not df.empty:
            if "ordering" in df.columns:
                df["ordering"] = pd.to_numeric(df["ordering"], errors="coerce")
            df.to_parquet(parquet, index=False)
            logger.info("Cached IMDB principals to %s", parquet)

        return df

    def load_imdb_names(
        self, valid_nconsts: set[str] | None = None
    ) -> pd.DataFrame:
        """Load IMDB ``name.basics.tsv.gz``.

        If *valid_nconsts* is provided, filters to those nconsts during read.

        Cache chain: parquet → raw .tsv.gz → chunked read.
        """
        parquet = self.data_dir / "imdb_names_filtered.parquet"
        if parquet.exists() and parquet.stat().st_size > 0:
            logger.info("Loading cached IMDB names from %s", parquet)
            df = pd.read_parquet(parquet)
            logger.info("  → %s rows from parquet cache", f"{len(df):,}")
            return df

        filename = "name.basics.tsv.gz"
        url = IMDB_BASE_URL + filename
        path = self._cached_or_download(filename, url)
        if path is None:
            logger.warning("IMDB names unavailable – returning empty DataFrame")
            return pd.DataFrame()

        if valid_nconsts:
            nconst_set = valid_nconsts

            def _filter_nconsts(chunk: pd.DataFrame) -> pd.DataFrame:
                return chunk[chunk["nconst"].isin(nconst_set)]

            df = self._read_imdb_tsv_chunked(path, filter_fn=_filter_nconsts)
        else:
            df = self._read_imdb_tsv_chunked(path)

        # Coerce mixed-type columns before parquet save
        if not df.empty:
            for col in ("birthYear", "deathYear"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.to_parquet(parquet, index=False)
            logger.info("Cached IMDB names to %s", parquet)

        return df

    def load_mendeley_dataset(self) -> pd.DataFrame:
        """Load the TIMDB Indian movies dataset (pncnmnp/TIMDB).

        Contains ~5000 Bollywood movies 1950-2019 with cast, crew, and
        metadata.  Falls back to any cached CSV on disk.
        """
        filename = "mendeley_indian_movies.csv"
        local = self.data_dir / filename

        if local.exists() and local.stat().st_size > 0:
            size_mb = local.stat().st_size / (1024 * 1024)
            logger.info("Using cached TIMDB file: %s (%.1f MB)", local, size_mb)
            return pd.read_csv(local)

        try:
            logger.info("Downloading TIMDB (Indian movies) dataset …")
            resp = requests.get(
                MENDELEY_DATASET_URL, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True
            )
            resp.raise_for_status()
            local.write_bytes(resp.content)
            size_mb = local.stat().st_size / (1024 * 1024)
            logger.info("Saved TIMDB dataset to %s (%.1f MB)", local, size_mb)
            return pd.read_csv(local)
        except (requests.RequestException, OSError) as exc:
            logger.warning("TIMDB download failed: %s", exc)

        # Fallback: look for any matching CSV in the cache
        for pattern in ("*mendeley*.csv", "*indian_movies*.csv", "*timdb*.csv"):
            for p in self.data_dir.glob(pattern):
                logger.info("Falling back to local file: %s", p)
                return pd.read_csv(p)

        logger.warning("TIMDB dataset unavailable – returning empty DataFrame")
        return pd.DataFrame()

    def load_github_bollywood(self) -> pd.DataFrame:
        """Load the Bollywood Movie Dataset from GitHub.

        The CSV is ``IMDB-Movie-Dataset(2023-1951).csv`` in the repo root.
        """
        filename = "bollywood_movies.csv"
        local = self.data_dir / filename

        if local.exists() and local.stat().st_size > 0:
            size_mb = local.stat().st_size / (1024 * 1024)
            logger.info("Using cached GitHub Bollywood file: %s (%.1f MB)", local, size_mb)
            return pd.read_csv(local)

        try:
            logger.info("Downloading GitHub Bollywood dataset …")
            resp = requests.get(GITHUB_BOLLYWOOD_CSV_URL, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            local.write_bytes(resp.content)
            size_mb = local.stat().st_size / (1024 * 1024)
            logger.info("Saved GitHub Bollywood dataset to %s (%.1f MB)", local, size_mb)
            return pd.read_csv(local)
        except (requests.RequestException, OSError) as exc:
            logger.warning("GitHub Bollywood download failed: %s", exc)

        for p in self.data_dir.glob("*bollywood*.csv"):
            logger.info("Falling back to local file: %s", p)
            return pd.read_csv(p)

        logger.warning("GitHub Bollywood dataset unavailable – returning empty DataFrame")
        return pd.DataFrame()

    def load_kaggle_indian_movies(self) -> pd.DataFrame:
        """Load the Kaggle Indian Movies IMDB dataset via ``kagglehub``.

        Falls back to any ``*kaggle*.csv`` in the cache directory.
        """
        filename = "kaggle_indian_movies.csv"
        local = self.data_dir / filename

        if local.exists() and local.stat().st_size > 0:
            size_mb = local.stat().st_size / (1024 * 1024)
            logger.info("Using cached Kaggle file: %s (%.1f MB)", local, size_mb)
            return pd.read_csv(local)

        try:
            import kagglehub

            logger.info("Downloading Kaggle dataset via kagglehub …")
            dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
            dataset_dir = Path(dataset_path)

            csv_files = list(dataset_dir.rglob("*.csv"))
            if not csv_files:
                logger.warning("No CSV found in Kaggle download at %s", dataset_dir)
                return pd.DataFrame()

            shutil.copy2(csv_files[0], local)
            size_mb = local.stat().st_size / (1024 * 1024)
            logger.info("Saved Kaggle dataset to %s (%.1f MB)", local, size_mb)
            return pd.read_csv(local)
        except Exception as exc:
            logger.warning("Kaggle download failed: %s", exc)

        for p in self.data_dir.glob("*kaggle*.csv"):
            logger.info("Falling back to local file: %s", p)
            return pd.read_csv(p)

        logger.warning("Kaggle dataset unavailable – returning empty DataFrame")
        return pd.DataFrame()

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all datasets and return a dict of non-empty DataFrames.

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
                        "  ✓ %s: %s rows, %d columns",
                        name,
                        f"{len(df):,}",
                        len(df.columns),
                    )
                else:
                    logger.warning("  ✗ %s returned empty – skipped", name)
            except Exception as exc:
                logger.warning("  ✗ %s failed: %s – skipped", name, exc)

        logger.info(
            "load_all complete – %d / %d sources loaded", len(results), len(loaders)
        )
        return results
