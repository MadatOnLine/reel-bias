"""Property-based tests for the Thematic Analysis module.

Uses Hypothesis to verify universal correctness properties of the
ThematicAnalyzer class.

Properties tested:
- Property 4: Genre Percentage Consistency (Validates: Requirement 5.3)
- Property 5: Genre Trend Sort Order (Validates: Requirement 5.4)
- Property 6: Genre Co-occurrence Symmetry (Validates: Requirement 6.2)
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from thematic_analysis import ThematicAnalyzer

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_GENRES = [
    "Drama", "Action", "Comedy", "Romance",
    "Thriller", "Horror", "Sci-Fi", "Musical",
]

_PERIODS = [
    "1975-1979", "1980-1984", "1985-1989", "1990-1994", "1995-1999",
    "2000-2004", "2005-2009", "2010-2014", "2015-2019", "2020-2024",
]

_genre = st.sampled_from(_GENRES)
_period = st.sampled_from(_PERIODS)


@st.composite
def _movie_genres(draw):
    """Draw 1-3 unique genres joined by comma."""
    n = draw(st.integers(min_value=1, max_value=3))
    genres = draw(
        st.lists(_genre, min_size=n, max_size=n, unique=True)
    )
    return ", ".join(genres)


@st.composite
def _movies_df(draw):
    """Draw a DataFrame of 1-30 movies with genres and period columns."""
    n = draw(st.integers(min_value=1, max_value=30))
    rows = []
    for _ in range(n):
        rows.append({
            "genres": draw(_movie_genres()),
            "period": draw(_period),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Property 4: Genre Percentage Consistency
# ---------------------------------------------------------------------------


class TestGenrePercentageConsistency:
    """**Validates: Requirement 5.3**

    Property 4: For any 5-year period in the genre trends output, the sum
    of all genre percentages for that period must equal 100.0 within a
    floating-point tolerance of 0.01.
    """

    @given(df=_movies_df())
    @settings(max_examples=50, deadline=None)
    def test_genre_percentages_sum_to_100(self, df: pd.DataFrame) -> None:
        analyzer = ThematicAnalyzer()
        result = analyzer.genre_trends_by_period(df)

        if result.empty:
            return

        for period in result["period"].unique():
            period_rows = result[result["period"] == period]
            pct_sum = period_rows["percentage"].sum()
            assert abs(pct_sum - 100.0) < 0.01, (
                f"Period {period}: genre percentages sum to {pct_sum}, "
                f"expected 100.0 ± 0.01"
            )


# ---------------------------------------------------------------------------
# Property 5: Genre Trend Sort Order
# ---------------------------------------------------------------------------


class TestGenreTrendSortOrder:
    """**Validates: Requirement 5.4**

    Property 5: Results must be sorted by period ascending, and within
    each period, by count descending.
    """

    @given(df=_movies_df())
    @settings(max_examples=50, deadline=None)
    def test_sorted_by_period_asc_count_desc(self, df: pd.DataFrame) -> None:
        analyzer = ThematicAnalyzer()
        result = analyzer.genre_trends_by_period(df)

        if result.empty:
            return

        # Periods should appear in ascending order
        periods = result["period"].tolist()
        unique_periods_in_order = list(dict.fromkeys(periods))
        assert unique_periods_in_order == sorted(unique_periods_in_order), (
            f"Periods not in ascending order: {unique_periods_in_order}"
        )

        # Within each period, counts should be descending
        for period in result["period"].unique():
            grp = result[result["period"] == period]
            counts = grp["count"].tolist()
            assert counts == sorted(counts, reverse=True), (
                f"Period {period}: counts not in descending order: {counts}"
            )


# ---------------------------------------------------------------------------
# Property 6: Genre Co-occurrence Symmetry
# ---------------------------------------------------------------------------


class TestGenreCooccurrenceSymmetry:
    """**Validates: Requirement 6.2**

    Property 6: For any genre co-occurrence matrix, the value at
    matrix[A][B] must equal the value at matrix[B][A] for all genre pairs.
    """

    @given(df=_movies_df())
    @settings(max_examples=50, deadline=None)
    def test_cooccurrence_matrix_is_symmetric(self, df: pd.DataFrame) -> None:
        analyzer = ThematicAnalyzer()
        matrix = analyzer.genre_cooccurrence_matrix(df)

        if matrix.empty:
            return

        for a in matrix.index:
            for b in matrix.columns:
                assert matrix.loc[a, b] == matrix.loc[b, a], (
                    f"Asymmetry: matrix[{a}][{b}]={matrix.loc[a, b]} != "
                    f"matrix[{b}][{a}]={matrix.loc[b, a]}"
                )
