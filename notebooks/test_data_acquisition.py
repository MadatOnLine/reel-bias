"""Unit tests for the DataAcquisition class.

Validates core behaviour: init, caching, error handling, and load_all contract.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_acquisition import DataAcquisition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary data directory."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def acq(tmp_data_dir: Path) -> DataAcquisition:
    """Return a DataAcquisition instance pointed at a temp directory."""
    return DataAcquisition(data_dir=str(tmp_data_dir))


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "new_dir"
        assert not target.exists()
        da = DataAcquisition(data_dir=str(target))
        assert target.is_dir()
        assert da.data_dir == target

    def test_uses_existing_directory(self, tmp_data_dir: Path) -> None:
        da = DataAcquisition(data_dir=str(tmp_data_dir))
        assert da.data_dir == tmp_data_dir

    def test_default_data_dir(self) -> None:
        da = DataAcquisition()
        assert da.data_dir == Path("data/")


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cached_file_is_reused(self, acq: DataAcquisition, tmp_data_dir: Path) -> None:
        """If a file already exists in data/, no download should happen."""
        # Create a fake cached gzip TSV
        tsv_content = "tconst\ttitleType\nprimaryTitle\n"
        gz_path = tmp_data_dir / "title.basics.tsv.gz"
        with gzip.open(gz_path, "wt") as f:
            f.write("tconst\ttitleType\nprimaryTitle\tmovie\n")

        with patch.object(acq, "_download_file") as mock_dl:
            df = acq.load_imdb_basics()
            mock_dl.assert_not_called()
            assert isinstance(df, pd.DataFrame)
            assert not df.empty


# ---------------------------------------------------------------------------
# Network error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_load_imdb_basics_returns_empty_on_failure(
        self, acq: DataAcquisition
    ) -> None:
        with patch.object(acq, "_download_file", return_value=False):
            df = acq.load_imdb_basics()
            assert isinstance(df, pd.DataFrame)
            assert df.empty

    def test_load_imdb_principals_returns_empty_on_failure(
        self, acq: DataAcquisition
    ) -> None:
        with patch.object(acq, "_download_file", return_value=False):
            df = acq.load_imdb_principals()
            assert isinstance(df, pd.DataFrame)
            assert df.empty

    def test_load_imdb_names_returns_empty_on_failure(
        self, acq: DataAcquisition
    ) -> None:
        with patch.object(acq, "_download_file", return_value=False):
            df = acq.load_imdb_names()
            assert isinstance(df, pd.DataFrame)
            assert df.empty

    def test_load_mendeley_returns_empty_on_failure(
        self, acq: DataAcquisition
    ) -> None:
        import requests as _requests

        with patch("data_acquisition.requests.get") as mock_get:
            mock_get.side_effect = _requests.RequestException("network error")
            df = acq.load_mendeley_dataset()
            assert isinstance(df, pd.DataFrame)
            assert df.empty

    def test_load_github_returns_empty_on_failure(
        self, acq: DataAcquisition
    ) -> None:
        import requests as _requests

        with patch("data_acquisition.requests.get") as mock_get:
            mock_get.side_effect = _requests.RequestException("network error")
            df = acq.load_github_bollywood()
            assert isinstance(df, pd.DataFrame)
            assert df.empty

    def test_load_kaggle_returns_empty_on_failure(
        self, acq: DataAcquisition
    ) -> None:
        with patch("kagglehub.dataset_download", side_effect=Exception("no creds")):
            df = acq.load_kaggle_indian_movies()
            assert isinstance(df, pd.DataFrame)
            assert df.empty


# ---------------------------------------------------------------------------
# load_all tests
# ---------------------------------------------------------------------------


class TestLoadAll:
    def test_returns_dict_of_dataframes(self, acq: DataAcquisition) -> None:
        """load_all should return a dict; empty sources are excluded."""
        fake_df = pd.DataFrame({"col": [1, 2, 3]})
        empty_df = pd.DataFrame()

        with (
            patch.object(acq, "load_imdb_basics", return_value=fake_df),
            patch.object(acq, "load_imdb_principals", return_value=empty_df),
            patch.object(acq, "load_imdb_names", return_value=fake_df),
            patch.object(acq, "load_mendeley_dataset", return_value=empty_df),
            patch.object(acq, "load_github_bollywood", return_value=fake_df),
            patch.object(acq, "load_kaggle_indian_movies", return_value=empty_df),
        ):
            result = acq.load_all()

        assert isinstance(result, dict)
        # Only non-empty sources should be present
        assert "imdb_basics" in result
        assert "imdb_names" in result
        assert "github_bollywood" in result
        assert "imdb_principals" not in result
        assert "mendeley" not in result
        assert "kaggle_indian_movies" not in result

        for name, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert not df.empty

    def test_load_all_skips_exceptions(self, acq: DataAcquisition) -> None:
        """If a loader raises, load_all logs a warning and continues."""
        fake_df = pd.DataFrame({"col": [1]})

        with (
            patch.object(acq, "load_imdb_basics", side_effect=RuntimeError("boom")),
            patch.object(acq, "load_imdb_principals", return_value=fake_df),
            patch.object(acq, "load_imdb_names", return_value=fake_df),
            patch.object(acq, "load_mendeley_dataset", return_value=fake_df),
            patch.object(acq, "load_github_bollywood", return_value=fake_df),
            patch.object(acq, "load_kaggle_indian_movies", return_value=fake_df),
        ):
            result = acq.load_all()

        assert "imdb_basics" not in result
        assert len(result) == 5

    def test_load_all_keys_are_expected_source_names(
        self, acq: DataAcquisition
    ) -> None:
        """All keys in the result must be from the known source set."""
        expected_keys = {
            "imdb_basics",
            "imdb_principals",
            "imdb_names",
            "mendeley",
            "github_bollywood",
            "kaggle_indian_movies",
        }
        fake_df = pd.DataFrame({"col": [1]})
        with (
            patch.object(acq, "load_imdb_basics", return_value=fake_df),
            patch.object(acq, "load_imdb_principals", return_value=fake_df),
            patch.object(acq, "load_imdb_names", return_value=fake_df),
            patch.object(acq, "load_mendeley_dataset", return_value=fake_df),
            patch.object(acq, "load_github_bollywood", return_value=fake_df),
            patch.object(acq, "load_kaggle_indian_movies", return_value=fake_df),
        ):
            result = acq.load_all()

        assert set(result.keys()).issubset(expected_keys)


# ---------------------------------------------------------------------------
# Gzip handling test
# ---------------------------------------------------------------------------


class TestGzipHandling:
    def test_reads_gzip_tsv_correctly(
        self, acq: DataAcquisition, tmp_data_dir: Path
    ) -> None:
        """Verify _read_imdb_tsv correctly decompresses and parses a .tsv.gz."""
        tsv_text = "tconst\ttitleType\tstartYear\ntt0000001\tmovie\t1999\n"
        gz_path = tmp_data_dir / "test.tsv.gz"
        with gzip.open(gz_path, "wt") as f:
            f.write(tsv_text)

        df = acq._read_imdb_tsv(gz_path)
        assert list(df.columns) == ["tconst", "titleType", "startYear"]
        assert len(df) == 1
        assert df.iloc[0]["tconst"] == "tt0000001"
        assert df.iloc[0]["startYear"] == 1999
