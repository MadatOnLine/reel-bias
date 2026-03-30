"""Smoke tests for the Visualizer class.

Verifies each visualization method executes without error on sample data
and returns a matplotlib Figure object. Uses the Agg backend for headless
testing.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from visualization import Visualizer


@pytest.fixture
def viz() -> Visualizer:
    return Visualizer()


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

def _genre_trends_data() -> pd.DataFrame:
    """Sample genre_trends DataFrame with period/genre/count/percentage."""
    rows = [
        {"period": "2000-2004", "genre": "Drama", "count": 50, "percentage": 50.0},
        {"period": "2000-2004", "genre": "Action", "count": 30, "percentage": 30.0},
        {"period": "2000-2004", "genre": "Comedy", "count": 20, "percentage": 20.0},
        {"period": "2005-2009", "genre": "Drama", "count": 40, "percentage": 40.0},
        {"period": "2005-2009", "genre": "Action", "count": 35, "percentage": 35.0},
        {"period": "2005-2009", "genre": "Thriller", "count": 25, "percentage": 25.0},
        {"period": "2010-2014", "genre": "Action", "count": 45, "percentage": 45.0},
        {"period": "2010-2014", "genre": "Drama", "count": 30, "percentage": 30.0},
        {"period": "2010-2014", "genre": "Comedy", "count": 25, "percentage": 25.0},
    ]
    return pd.DataFrame(rows)


def _cooccurrence_data() -> pd.DataFrame:
    """Sample symmetric co-occurrence matrix."""
    genres = ["Drama", "Action", "Comedy"]
    data = [[10, 5, 3], [5, 8, 2], [3, 2, 6]]
    return pd.DataFrame(data, index=genres, columns=genres)


def _topic_words_data() -> dict[int, list[str]]:
    return {
        0: ["love", "family", "marriage", "village", "mother"],
        1: ["police", "crime", "murder", "city", "gang"],
        2: ["song", "dance", "music", "festival", "celebration"],
    }


def _role_distribution_data() -> pd.DataFrame:
    """Sample crosstab: role_type (rows) x gender (columns)."""
    data = {"male": [40, 20, 15, 10], "female": [25, 30, 5, 8], "unknown": [5, 5, 3, 2]}
    return pd.DataFrame(data, index=["lead", "supporting", "villain", "comic"])


def _temporal_bias_data() -> pd.DataFrame:
    rows = [
        {"period": "2000-2004", "attribute": "inferred_gender", "value": "male", "count": 60, "percentage": 60.0},
        {"period": "2000-2004", "attribute": "inferred_gender", "value": "female", "count": 40, "percentage": 40.0},
        {"period": "2005-2009", "attribute": "inferred_gender", "value": "male", "count": 55, "percentage": 55.0},
        {"period": "2005-2009", "attribute": "inferred_gender", "value": "female", "count": 45, "percentage": 45.0},
        {"period": "2010-2014", "attribute": "inferred_gender", "value": "male", "count": 50, "percentage": 50.0},
        {"period": "2010-2014", "attribute": "inferred_gender", "value": "female", "count": 50, "percentage": 50.0},
    ]
    return pd.DataFrame(rows)


def _name_frequency_data() -> pd.DataFrame:
    rows = [
        {"name": "Raj", "count": 25, "period": "2000-2004", "inferred_gender": "male", "role_type": "lead"},
        {"name": "Priya", "count": 20, "period": "2000-2004", "inferred_gender": "female", "role_type": "lead"},
        {"name": "Vijay", "count": 18, "period": "2000-2004", "inferred_gender": "male", "role_type": "supporting"},
        {"name": "Simran", "count": 15, "period": "2005-2009", "inferred_gender": "female", "role_type": "lead"},
        {"name": "Arjun", "count": 12, "period": "2005-2009", "inferred_gender": "male", "role_type": "villain"},
    ]
    return pd.DataFrame(rows)


def _significance_results() -> dict:
    return {
        "chi2_statistic": 15.3,
        "p_value": 0.002,
        "degrees_of_freedom": 4,
        "effect_size": 0.35,
        "significant": True,
        "test_used": "chi_square",
    }


# ===================================================================
# 10.1  plot_genre_trends, plot_genre_heatmap, plot_topic_wordclouds
# ===================================================================


class TestPlotGenreTrends:
    """Validates: Requirements 11.1, 11.6, 11.7"""

    def test_returns_figure(self, viz: Visualizer) -> None:
        fig = viz.plot_genre_trends(_genre_trends_data())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_data(self, viz: Visualizer) -> None:
        fig = viz.plot_genre_trends(pd.DataFrame())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_dpi_is_300(self, viz: Visualizer) -> None:
        fig = viz.plot_genre_trends(_genre_trends_data())
        assert fig.dpi == 300


class TestPlotGenreHeatmap:
    """Validates: Requirements 11.2, 11.6, 11.7"""

    def test_returns_figure(self, viz: Visualizer) -> None:
        fig = viz.plot_genre_heatmap(_cooccurrence_data())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_data(self, viz: Visualizer) -> None:
        fig = viz.plot_genre_heatmap(pd.DataFrame())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_dpi_is_300(self, viz: Visualizer) -> None:
        fig = viz.plot_genre_heatmap(_cooccurrence_data())
        assert fig.dpi == 300


class TestPlotTopicWordclouds:
    """Validates: Requirements 11.3, 11.6, 11.7"""

    def test_returns_figure(self, viz: Visualizer) -> None:
        fig = viz.plot_topic_wordclouds(_topic_words_data())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_data(self, viz: Visualizer) -> None:
        fig = viz.plot_topic_wordclouds({})
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_single_topic(self, viz: Visualizer) -> None:
        fig = viz.plot_topic_wordclouds({0: ["word1", "word2", "word3"]})
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_dpi_is_300(self, viz: Visualizer) -> None:
        fig = viz.plot_topic_wordclouds(_topic_words_data())
        assert fig.dpi == 300


# ===================================================================
# 10.2  plot_role_distribution, plot_temporal_bias, plot_name_frequency
# ===================================================================


class TestPlotRoleDistribution:
    """Validates: Requirements 11.4, 11.6, 11.7, 11.8"""

    def test_returns_figure(self, viz: Visualizer) -> None:
        fig = viz.plot_role_distribution(_role_distribution_data(), group_by="gender")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_data(self, viz: Visualizer) -> None:
        fig = viz.plot_role_distribution(pd.DataFrame(), group_by="gender")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_significance(self, viz: Visualizer) -> None:
        fig = viz.plot_role_distribution(
            _role_distribution_data(),
            group_by="gender",
            significance_results=_significance_results(),
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_without_significance(self, viz: Visualizer) -> None:
        fig = viz.plot_role_distribution(
            _role_distribution_data(),
            group_by="religion",
            significance_results={"significant": False, "p_value": 0.5},
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_dpi_is_300(self, viz: Visualizer) -> None:
        fig = viz.plot_role_distribution(_role_distribution_data(), group_by="gender")
        assert fig.dpi == 300


class TestPlotTemporalBias:
    """Validates: Requirements 11.5, 11.6, 11.7, 11.8"""

    def test_returns_figure(self, viz: Visualizer) -> None:
        fig = viz.plot_temporal_bias(_temporal_bias_data())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_data(self, viz: Visualizer) -> None:
        fig = viz.plot_temporal_bias(pd.DataFrame())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_significance(self, viz: Visualizer) -> None:
        fig = viz.plot_temporal_bias(
            _temporal_bias_data(),
            significance_results=_significance_results(),
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_dpi_is_300(self, viz: Visualizer) -> None:
        fig = viz.plot_temporal_bias(_temporal_bias_data())
        assert fig.dpi == 300


class TestPlotNameFrequency:
    """Validates: Requirements 11.4, 11.6, 11.7, 11.8"""

    def test_returns_figure(self, viz: Visualizer) -> None:
        fig = viz.plot_name_frequency(_name_frequency_data())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_data(self, viz: Visualizer) -> None:
        fig = viz.plot_name_frequency(pd.DataFrame())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_significance(self, viz: Visualizer) -> None:
        fig = viz.plot_name_frequency(
            _name_frequency_data(),
            significance_results=_significance_results(),
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_dpi_is_300(self, viz: Visualizer) -> None:
        fig = viz.plot_name_frequency(_name_frequency_data())
        assert fig.dpi == 300


# ===================================================================
# 10.3  plot_summary_dashboard
# ===================================================================


class TestPlotSummaryDashboard:
    """Validates: Requirements 11.6, 11.7"""

    def test_returns_figure_with_all_data(self, viz: Visualizer) -> None:
        fig = viz.plot_summary_dashboard({
            "genre_trends": _genre_trends_data(),
            "bias_results": _significance_results(),
            "temporal_bias": _temporal_bias_data(),
        })
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_with_empty_dict(self, viz: Visualizer) -> None:
        fig = viz.plot_summary_dashboard({})
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_with_partial_data(self, viz: Visualizer) -> None:
        fig = viz.plot_summary_dashboard({
            "genre_trends": _genre_trends_data(),
        })
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_dpi_is_300(self, viz: Visualizer) -> None:
        fig = viz.plot_summary_dashboard({
            "genre_trends": _genre_trends_data(),
            "bias_results": _significance_results(),
            "temporal_bias": _temporal_bias_data(),
        })
        assert fig.dpi == 300

    def test_handles_non_significant_results(self, viz: Visualizer) -> None:
        fig = viz.plot_summary_dashboard({
            "bias_results": {"significant": False, "p_value": 0.5, "effect_size": 0.05, "chi2_statistic": 1.2},
        })
        assert isinstance(fig, matplotlib.figure.Figure)
