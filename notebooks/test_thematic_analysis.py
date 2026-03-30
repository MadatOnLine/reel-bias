"""Unit tests for the ThematicAnalyzer class.

Validates: genre_trends_by_period, genre_cooccurrence_matrix,
topic_model_plots, keyword_trends, runtime_trends, rating_trends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thematic_analysis import ThematicAnalyzer


@pytest.fixture
def analyzer() -> ThematicAnalyzer:
    return ThematicAnalyzer()


# ---------------------------------------------------------------------------
# Helper to build a small master-like DataFrame
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ===================================================================
# 6.1  genre_trends_by_period
# ===================================================================


class TestGenreTrendsByPeriod:
    """Validates: Requirements 5.1, 5.2, 5.3, 5.4"""

    def test_basic_counts_and_percentage(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": "Drama,Action", "period": "2000-2004"},
            {"genres": "Drama", "period": "2000-2004"},
            {"genres": "Action", "period": "2000-2004"},
        ])
        result = analyzer.genre_trends_by_period(df)
        assert set(result.columns) == {"period", "genre", "count", "percentage"}

        # Drama appears 2 times, Action appears 2 times in period 2000-2004
        period_rows = result[result["period"] == "2000-2004"]
        assert len(period_rows) == 2
        total_pct = period_rows["percentage"].sum()
        assert abs(total_pct - 100.0) < 0.01

    def test_explodes_comma_separated_genres(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": "Drama,Action,Comedy", "period": "2000-2004"},
        ])
        result = analyzer.genre_trends_by_period(df)
        assert len(result) == 3
        assert set(result["genre"]) == {"Drama", "Action", "Comedy"}

    def test_explodes_list_genres(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": ["Drama", "Action"], "period": "2000-2004"},
        ])
        result = analyzer.genre_trends_by_period(df)
        assert len(result) == 2

    def test_percentages_sum_to_100(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": "Drama", "period": "2000-2004"},
            {"genres": "Action", "period": "2000-2004"},
            {"genres": "Comedy", "period": "2000-2004"},
            {"genres": "Drama", "period": "2005-2009"},
            {"genres": "Thriller", "period": "2005-2009"},
        ])
        result = analyzer.genre_trends_by_period(df)
        for period in result["period"].unique():
            pct_sum = result[result["period"] == period]["percentage"].sum()
            assert abs(pct_sum - 100.0) < 0.01, f"Period {period}: {pct_sum}"

    def test_sorted_by_period_asc_count_desc(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": "Drama", "period": "2005-2009"},
            {"genres": "Drama", "period": "2005-2009"},
            {"genres": "Action", "period": "2005-2009"},
            {"genres": "Comedy", "period": "2000-2004"},
            {"genres": "Comedy", "period": "2000-2004"},
            {"genres": "Drama", "period": "2000-2004"},
        ])
        result = analyzer.genre_trends_by_period(df)
        periods = result["period"].tolist()
        # Periods should be ascending
        assert periods == sorted(periods, key=lambda p: p)
        # Within each period, counts should be descending
        for period in result["period"].unique():
            grp = result[result["period"] == period]
            counts = grp["count"].tolist()
            assert counts == sorted(counts, reverse=True)

    def test_empty_dataframe(self, analyzer: ThematicAnalyzer) -> None:
        result = analyzer.genre_trends_by_period(pd.DataFrame())
        assert result.empty
        assert set(result.columns) == {"period", "genre", "count", "percentage"}

    def test_strips_whitespace_from_genres(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": " Drama , Action ", "period": "2000-2004"},
        ])
        result = analyzer.genre_trends_by_period(df)
        assert set(result["genre"]) == {"Drama", "Action"}

    def test_skips_rows_with_empty_genres(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": "", "period": "2000-2004"},
            {"genres": "Drama", "period": "2000-2004"},
        ])
        result = analyzer.genre_trends_by_period(df)
        assert len(result) == 1
        assert result.iloc[0]["genre"] == "Drama"
        assert result.iloc[0]["percentage"] == 100.0


# ===================================================================
# 6.2  genre_cooccurrence_matrix
# ===================================================================


class TestGenreCooccurrenceMatrix:
    """Validates: Requirements 6.1, 6.2"""

    def test_symmetric_matrix(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": "Drama,Action"},
            {"genres": "Drama,Comedy"},
            {"genres": "Action,Comedy"},
        ])
        matrix = analyzer.genre_cooccurrence_matrix(df)
        # Symmetry check
        for a in matrix.index:
            for b in matrix.columns:
                assert matrix.loc[a, b] == matrix.loc[b, a], (
                    f"matrix[{a}][{b}]={matrix.loc[a, b]} != "
                    f"matrix[{b}][{a}]={matrix.loc[b, a]}"
                )

    def test_counts_pairs_correctly(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": "Drama,Action"},
            {"genres": "Drama,Action"},
            {"genres": "Drama,Comedy"},
        ])
        matrix = analyzer.genre_cooccurrence_matrix(df)
        assert matrix.loc["Drama", "Action"] == 2
        assert matrix.loc["Drama", "Comedy"] == 1
        assert matrix.loc["Action", "Comedy"] == 0

    def test_diagonal_counts_movies_with_genre(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": "Drama,Action"},
            {"genres": "Drama"},
            {"genres": "Action,Comedy"},
        ])
        matrix = analyzer.genre_cooccurrence_matrix(df)
        assert matrix.loc["Drama", "Drama"] == 2
        assert matrix.loc["Action", "Action"] == 2
        assert matrix.loc["Comedy", "Comedy"] == 1

    def test_empty_dataframe(self, analyzer: ThematicAnalyzer) -> None:
        result = analyzer.genre_cooccurrence_matrix(pd.DataFrame())
        assert result.empty

    def test_single_genre_movies(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"genres": "Drama"},
            {"genres": "Action"},
        ])
        matrix = analyzer.genre_cooccurrence_matrix(df)
        assert matrix.loc["Drama", "Action"] == 0
        assert matrix.loc["Drama", "Drama"] == 1
        assert matrix.loc["Action", "Action"] == 1

    def test_handles_string_genres(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([{"genres": "Drama,Action,Comedy"}])
        matrix = analyzer.genre_cooccurrence_matrix(df)
        assert matrix.loc["Drama", "Action"] == 1
        assert matrix.loc["Drama", "Comedy"] == 1
        assert matrix.loc["Action", "Comedy"] == 1


# ===================================================================
# 6.3  topic_model_plots (LDA)
# ===================================================================


class TestTopicModelPlots:
    """Validates: Requirements 7.1, 7.2, 7.3, 13.5"""

    def test_fallback_when_no_plot_column(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"title": "Movie A", "genres": "Drama"},
            {"title": "Movie B", "genres": "Action"},
        ])
        result = analyzer.topic_model_plots(df, n_topics=3)
        assert result["fallback"] == "genre_only"
        assert result["coherence_score"] == 0.0

    def test_fallback_when_majority_lack_text(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"plot_summary": "A hero saves the day", "genres": "Action"},
            {"plot_summary": "", "genres": "Drama"},
            {"plot_summary": np.nan, "genres": "Comedy"},
            {"plot_summary": "", "genres": "Thriller"},
        ])
        result = analyzer.topic_model_plots(df, n_topics=2)
        assert result.get("fallback") == "genre_only"

    def test_runs_lda_when_enough_text(self, analyzer: ThematicAnalyzer) -> None:
        plots = [
            "A young hero embarks on a journey to save the kingdom from evil forces",
            "Two lovers meet in a small town and overcome family opposition",
            "A detective investigates a series of mysterious murders in the city",
            "A comedy about friends who go on a road trip across the country",
            "A drama about a family dealing with loss and finding hope again",
            "An action thriller about a spy who must stop a terrorist plot",
            "A romantic story set in the mountains during the monsoon season",
            "A crime drama following a police officer fighting corruption",
            "A musical celebration of love and dance in a vibrant village",
            "A suspenseful tale of betrayal and revenge in the underworld",
        ]
        df = _make_df([{"plot_summary": p, "genres": "Drama"} for p in plots])
        result = analyzer.topic_model_plots(df, n_topics=2)
        assert "fallback" not in result
        assert isinstance(result["topic_words"], dict)
        assert len(result["topic_words"]) == 2
        assert isinstance(result["document_topics"], pd.DataFrame)
        assert isinstance(result["coherence_score"], float)

    def test_configurable_n_topics(self, analyzer: ThematicAnalyzer) -> None:
        plots = [
            "A brave warrior fights against the evil king to save the village people",
            "Two young lovers elope from their families and travel across India",
            "A detective solves a murder mystery in the streets of Mumbai city",
            "Friends embark on a hilarious road trip through the countryside together",
            "A mother struggles to raise her children alone after losing her husband",
            "An undercover agent infiltrates a dangerous criminal gang operation",
            "A village teacher inspires students to dream big and achieve success",
            "A musician rises from poverty to become a famous singer in Bollywood",
            "A corrupt politician faces justice when a journalist exposes the truth",
            "A ghost haunts an old mansion and terrorizes the new family living there",
            "A brave warrior defends the kingdom against invading foreign armies",
            "Two childhood sweethearts reunite after years of separation abroad",
            "A police officer hunts down a serial killer terrorizing the city streets",
            "A group of misfits plan an elaborate heist of a national bank vault",
            "A daughter fights societal norms to pursue her dream of becoming a boxer",
            "A spy mission goes wrong when the agent discovers a double agent plot",
            "A rural doctor brings modern medicine to a remote tribal village community",
            "A dancer overcomes injury to perform at the national competition finals",
            "A whistleblower risks everything to expose corporate environmental crimes",
            "A supernatural force threatens the peace of a quiet coastal fishing town",
        ]
        df = _make_df([{"plot_summary": p, "genres": "Drama"} for p in plots])
        result = analyzer.topic_model_plots(df, n_topics=5)
        assert len(result["topic_words"]) == 5

    def test_uses_alternative_text_columns(self, analyzer: ThematicAnalyzer) -> None:
        plots = [
            "A brave warrior fights against the evil king to save the village",
            "Two young lovers elope from their families and travel across India",
            "A detective solves a murder mystery in the streets of Mumbai city",
            "Friends embark on a hilarious road trip through the countryside",
            "A mother struggles to raise her children alone after losing husband",
            "An undercover agent infiltrates a dangerous criminal gang operation",
            "A village teacher inspires students to dream big and achieve goals",
            "A musician rises from poverty to become a famous singer star",
            "A corrupt politician faces justice when a journalist exposes truth",
            "A ghost haunts an old mansion and terrorizes the new family there",
            "A brave soldier defends the kingdom against invading foreign armies",
            "Two childhood sweethearts reunite after years of separation abroad",
            "A police officer hunts down a serial killer terrorizing the city",
            "A group of misfits plan an elaborate heist of a national bank",
            "A daughter fights societal norms to pursue her dream of boxing",
        ]
        df = _make_df([{"description": p, "genres": "Drama"} for p in plots])
        result = analyzer.topic_model_plots(df, n_topics=2)
        # Should use 'description' column and not fall back
        assert "fallback" not in result


# ===================================================================
# 6.4  keyword_trends, runtime_trends, rating_trends
# ===================================================================


class TestKeywordTrends:
    """Validates: Requirements 5.1, 5.2"""

    def test_returns_keywords_per_period(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"title": "Love Story", "period": "2000-2004"},
            {"title": "Love Again", "period": "2000-2004"},
            {"title": "Action Hero", "period": "2005-2009"},
            {"title": "Action King", "period": "2005-2009"},
        ])
        result = analyzer.keyword_trends(df, top_n=5)
        assert set(result.columns) == {"period", "keyword", "tfidf_score"}
        assert "2000-2004" in result["period"].values
        assert "2005-2009" in result["period"].values

    def test_sorted_by_period_then_score(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"title": f"Movie {i}", "period": "2000-2004"} for i in range(10)
        ] + [
            {"title": f"Film {i}", "period": "2005-2009"} for i in range(10)
        ])
        result = analyzer.keyword_trends(df, top_n=5)
        if not result.empty:
            periods = result["period"].tolist()
            assert periods == sorted(periods)

    def test_empty_dataframe(self, analyzer: ThematicAnalyzer) -> None:
        result = analyzer.keyword_trends(pd.DataFrame())
        assert result.empty

    def test_missing_columns(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([{"genres": "Drama"}])
        result = analyzer.keyword_trends(df)
        assert result.empty


class TestRuntimeTrends:
    """Validates: Requirements 5.1, 5.2"""

    def test_computes_stats_per_period(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"runtimeMinutes": 120, "period": "2000-2004"},
            {"runtimeMinutes": 150, "period": "2000-2004"},
            {"runtimeMinutes": 90, "period": "2005-2009"},
            {"runtimeMinutes": 110, "period": "2005-2009"},
        ])
        result = analyzer.runtime_trends(df)
        assert set(result.columns) == {"period", "mean", "median", "std"}
        assert len(result) == 2
        row_2000 = result[result["period"] == "2000-2004"].iloc[0]
        assert row_2000["mean"] == 135.0
        assert row_2000["median"] == 135.0

    def test_handles_runtime_column_name(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"runtime": 100, "period": "2000-2004"},
            {"runtime": 120, "period": "2000-2004"},
        ])
        result = analyzer.runtime_trends(df)
        assert len(result) == 1

    def test_empty_dataframe(self, analyzer: ThematicAnalyzer) -> None:
        result = analyzer.runtime_trends(pd.DataFrame())
        assert result.empty

    def test_no_runtime_column(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([{"title": "Movie", "period": "2000-2004"}])
        result = analyzer.runtime_trends(df)
        assert result.empty

    def test_coerces_non_numeric_runtime(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"runtimeMinutes": "120", "period": "2000-2004"},
            {"runtimeMinutes": "not_a_number", "period": "2000-2004"},
            {"runtimeMinutes": "90", "period": "2000-2004"},
        ])
        result = analyzer.runtime_trends(df)
        assert len(result) == 1
        assert result.iloc[0]["mean"] == 105.0


class TestRatingTrends:
    """Validates: Requirements 5.1, 5.2"""

    def test_computes_stats_per_period(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"rating": 7.0, "period": "2000-2004"},
            {"rating": 8.0, "period": "2000-2004"},
            {"rating": 6.0, "period": "2005-2009"},
            {"rating": 9.0, "period": "2005-2009"},
        ])
        result = analyzer.rating_trends(df)
        assert set(result.columns) == {"period", "mean", "median", "std"}
        assert len(result) == 2

    def test_empty_dataframe(self, analyzer: ThematicAnalyzer) -> None:
        result = analyzer.rating_trends(pd.DataFrame())
        assert result.empty

    def test_no_rating_column(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([{"title": "Movie", "period": "2000-2004"}])
        result = analyzer.rating_trends(df)
        assert result.empty

    def test_handles_nan_ratings(self, analyzer: ThematicAnalyzer) -> None:
        df = _make_df([
            {"rating": 7.0, "period": "2000-2004"},
            {"rating": np.nan, "period": "2000-2004"},
            {"rating": 8.0, "period": "2000-2004"},
        ])
        result = analyzer.rating_trends(df)
        assert len(result) == 1
        assert result.iloc[0]["mean"] == 7.5
