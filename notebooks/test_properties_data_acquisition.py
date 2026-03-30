"""Property-based tests for the DataAcquisition.load_all method.

**Validates: Requirements 1.5**
Property 15 (partial): Unified Schema Compliance — verify returned dict keys
match expected source names and all values are non-empty DataFrames.

Uses Hypothesis to generate random non-empty DataFrames for each source,
mocking individual loaders.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from data_acquisition import DataAcquisition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_SOURCE_NAMES = frozenset(
    {
        "imdb_basics",
        "imdb_principals",
        "imdb_names",
        "mendeley",
        "github_bollywood",
        "kaggle_indian_movies",
    }
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: a non-empty DataFrame with 1-5 rows and a single text column
# Use st.text with alphabet restricted to BMP characters to avoid surrogates
_safe_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=20,
)
_nonempty_df = data_frames(
    columns=[column("val", dtype=str, elements=_safe_text)],
    index=range_indexes(min_size=1, max_size=5),
)

# Strategy: a dict mapping every expected source name to a non-empty DataFrame
_all_source_dfs = st.fixed_dictionaries(
    {name: _nonempty_df for name in sorted(EXPECTED_SOURCE_NAMES)}
)

# Strategy: a non-empty subset of source names (at least 1)
_source_subset = st.frozensets(
    st.sampled_from(sorted(EXPECTED_SOURCE_NAMES)), min_size=1
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestLoadAllProperties:
    """Property-based tests for DataAcquisition.load_all."""

    @given(source_dfs=_all_source_dfs)
    @settings(max_examples=30, deadline=None)
    def test_all_loaders_nonempty_keys_match_expected(
        self, source_dfs: dict[str, pd.DataFrame]
    ) -> None:
        """**Validates: Requirements 1.5**

        When every loader returns a non-empty DataFrame, load_all must return
        a dict whose keys are exactly the expected source names and whose
        values are all non-empty DataFrames.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            acq = DataAcquisition(data_dir=tmp_dir)

            with (
                patch.object(acq, "load_imdb_basics", return_value=source_dfs["imdb_basics"]),
                patch.object(acq, "load_imdb_principals", return_value=source_dfs["imdb_principals"]),
                patch.object(acq, "load_imdb_names", return_value=source_dfs["imdb_names"]),
                patch.object(acq, "load_mendeley_dataset", return_value=source_dfs["mendeley"]),
                patch.object(acq, "load_github_bollywood", return_value=source_dfs["github_bollywood"]),
                patch.object(acq, "load_kaggle_indian_movies", return_value=source_dfs["kaggle_indian_movies"]),
            ):
                result = acq.load_all()

            # Keys must be exactly the expected set
            assert set(result.keys()) == EXPECTED_SOURCE_NAMES

            # Every value must be a non-empty DataFrame
            for key, df in result.items():
                assert isinstance(df, pd.DataFrame), f"{key} is not a DataFrame"
                assert not df.empty, f"{key} DataFrame is empty"

    @given(nonempty_sources=_source_subset, source_dfs=_all_source_dfs)
    @settings(max_examples=30, deadline=None)
    def test_empty_loaders_excluded_from_result(
        self,
        nonempty_sources: frozenset[str],
        source_dfs: dict[str, pd.DataFrame],
    ) -> None:
        """**Validates: Requirements 1.5**

        When some loaders return empty DataFrames, those keys must be
        excluded from the result. Only non-empty sources appear.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            acq = DataAcquisition(data_dir=tmp_dir)

            def _mock_return(name: str) -> pd.DataFrame:
                if name in nonempty_sources:
                    return source_dfs[name]
                return pd.DataFrame()

            with (
                patch.object(acq, "load_imdb_basics", return_value=_mock_return("imdb_basics")),
                patch.object(acq, "load_imdb_principals", return_value=_mock_return("imdb_principals")),
                patch.object(acq, "load_imdb_names", return_value=_mock_return("imdb_names")),
                patch.object(acq, "load_mendeley_dataset", return_value=_mock_return("mendeley")),
                patch.object(acq, "load_github_bollywood", return_value=_mock_return("github_bollywood")),
                patch.object(acq, "load_kaggle_indian_movies", return_value=_mock_return("kaggle_indian_movies")),
            ):
                result = acq.load_all()

            # Result keys must be a subset of expected names
            assert set(result.keys()).issubset(EXPECTED_SOURCE_NAMES)

            # Only non-empty sources should appear
            assert set(result.keys()) == nonempty_sources

            # All returned DataFrames must be non-empty
            for key, df in result.items():
                assert isinstance(df, pd.DataFrame), f"{key} is not a DataFrame"
                assert not df.empty, f"{key} DataFrame should not be empty"
