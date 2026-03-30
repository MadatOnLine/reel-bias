"""Property-based tests for the Data Cleaning module.

Uses Hypothesis to verify universal correctness properties of the
DataCleaner class and helper functions.

Properties tested:
- Property 3: Sentinel Value Elimination (Validates: Requirement 2.1)
- Property 1: Year Range Integrity (Validates: Requirements 2.3, 4.1)
- Property 2: Period Bin Correctness (Validates: Requirements 4.2, 4.3)
- Property 14: Principals Filtering Integrity (Validates: Requirement 2.5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from data_cleaning import DataCleaner, IMDB_SENTINEL, assign_period_bin

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: random string that may or may not be the IMDB sentinel
_cell_value = st.one_of(
    st.just(IMDB_SENTINEL),
    st.text(min_size=0, max_size=20),
    st.integers(min_value=0, max_value=9999).map(str),
)

# Strategy: a valid IMDB tconst identifier
_tconst = st.from_regex(r"tt[0-9]{4,8}", fullmatch=True)

# Strategy: a year that may or may not be in [1975, 2025]
_any_year = st.integers(min_value=1900, max_value=2100)

# Strategy: a year guaranteed to be in [1975, 2025]
_valid_year = st.integers(min_value=1975, max_value=2025)

# Strategy: IMDB category values
_category = st.sampled_from(["actor", "actress", "director", "writer", "producer", "self"])


# ---------------------------------------------------------------------------
# Property 3: Sentinel Value Elimination
# ---------------------------------------------------------------------------


class TestSentinelValueElimination:
    """**Validates: Requirement 2.1**

    Property 3: After cleaning with clean_imdb_basics, no cell in the
    output DataFrame contains the IMDB sentinel string '\\N'.
    """

    @given(
        tconsts=st.lists(_tconst, min_size=1, max_size=10),
        extra_values=st.lists(_cell_value, min_size=1, max_size=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_no_sentinel_after_cleaning(
        self, tconsts: list[str], extra_values: list[str]
    ) -> None:
        """After clean_imdb_basics, no cell contains '\\N'."""
        n = len(tconsts)
        # Pad or trim extra_values to match tconsts length
        extras = (extra_values * ((n // len(extra_values)) + 1))[:n]

        df = pd.DataFrame(
            {
                "tconst": tconsts,
                "titleType": ["movie"] * n,
                "startYear": extras,
                "endYear": [IMDB_SENTINEL] * n,  # always include sentinels
                "runtimeMinutes": extras,
                "genres": ["Drama"] * n,
            }
        )

        cleaner = DataCleaner()
        result = cleaner.clean_imdb_basics(df)

        # Check every cell in the result for the sentinel
        for col in result.columns:
            for val in result[col]:
                if isinstance(val, str):
                    assert val != IMDB_SENTINEL, (
                        f"Sentinel '\\N' found in column '{col}'"
                    )


# ---------------------------------------------------------------------------
# Property 1: Year Range Integrity
# ---------------------------------------------------------------------------


class TestYearRangeIntegrity:
    """**Validates: Requirements 2.3, 4.1**

    Property 1: All years in the output of filter_indian_movies are in
    [1975, 2025].
    """

    @given(
        tconsts=st.lists(_tconst, min_size=1, max_size=15, unique=True),
        years=st.lists(_any_year, min_size=1, max_size=15),
    )
    @settings(max_examples=50, deadline=None)
    def test_all_years_in_range(
        self, tconsts: list[str], years: list[int]
    ) -> None:
        """After filter_indian_movies, every year is in [1975, 2025]."""
        n = min(len(tconsts), len(years))
        tconsts = tconsts[:n]
        years = years[:n]

        df = pd.DataFrame({"tconst": tconsts, "startYear": years})

        cleaner = DataCleaner()
        result = cleaner.filter_indian_movies(df)

        if not result.empty:
            result_years = pd.to_numeric(result["startYear"], errors="coerce").dropna()
            assert (result_years >= 1975).all(), (
                f"Found year(s) below 1975: {result_years[result_years < 1975].tolist()}"
            )
            assert (result_years <= 2025).all(), (
                f"Found year(s) above 2025: {result_years[result_years > 2025].tolist()}"
            )


# ---------------------------------------------------------------------------
# Property 2: Period Bin Correctness
# ---------------------------------------------------------------------------


class TestPeriodBinCorrectness:
    """**Validates: Requirements 4.2, 4.3**

    Property 2: Each movie's year falls within its assigned 5-year period
    start and end.
    """

    @given(year=_valid_year)
    @settings(max_examples=100, deadline=None)
    def test_year_within_assigned_period(self, year: int) -> None:
        """For any year in [1975, 2025], the year falls within the
        start and end of the period returned by assign_period_bin."""
        period = assign_period_bin(year)
        start_str, end_str = period.split("-")
        start, end = int(start_str), int(end_str)

        assert start <= year <= end, (
            f"Year {year} not in period {period} (range [{start}, {end}])"
        )

    @given(year=_valid_year)
    @settings(max_examples=100, deadline=None)
    def test_period_span_is_five_years(self, year: int) -> None:
        """Every period bin spans exactly 5 years (end - start == 4)."""
        period = assign_period_bin(year)
        start_str, end_str = period.split("-")
        start, end = int(start_str), int(end_str)

        assert end - start == 4, (
            f"Period {period} does not span 5 years (end - start = {end - start})"
        )


# ---------------------------------------------------------------------------
# Property 14: Principals Filtering Integrity
# ---------------------------------------------------------------------------


class TestPrincipalsFilteringIntegrity:
    """**Validates: Requirement 2.5**

    Property 14: All tconst values in cleaned principals exist in the
    valid Indian movie tconst set.
    """

    @given(
        valid_tconsts=st.frozensets(_tconst, min_size=1, max_size=10),
        extra_tconsts=st.lists(_tconst, min_size=1, max_size=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_all_tconsts_in_valid_set(
        self, valid_tconsts: frozenset[str], extra_tconsts: list[str]
    ) -> None:
        """After clean_imdb_principals, every tconst in the result is a
        member of the valid_tconsts set."""
        # Build a principals DataFrame with a mix of valid and extra tconsts
        all_tconsts = list(valid_tconsts) + extra_tconsts
        n = len(all_tconsts)

        df = pd.DataFrame(
            {
                "tconst": all_tconsts,
                "ordering": list(range(n)),
                "nconst": [f"nm{i:07d}" for i in range(n)],
                "category": (["actor", "actress", "director"] * ((n // 3) + 1))[:n],
                "job": [IMDB_SENTINEL] * n,
                "characters": ['["Character"]'] * n,
            }
        )

        cleaner = DataCleaner()
        result = cleaner.clean_imdb_principals(df, set(valid_tconsts))

        if not result.empty:
            result_tconsts = set(result["tconst"])
            assert result_tconsts.issubset(valid_tconsts), (
                f"Found tconst(s) not in valid set: "
                f"{result_tconsts - valid_tconsts}"
            )
