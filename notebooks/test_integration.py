"""Integration tests for the end-to-end Indian Movies IMDB Research pipeline.

Generates a synthetic sample dataset of 100 movies with realistic columns,
runs the full pipeline (cleaning → merging → thematic analysis → bias
analysis → statistical tests → visualizations), and verifies correctness.

Uses matplotlib Agg backend for headless testing.

Validates: Requirement 14.2
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from data_cleaning import DataCleaner, add_period_column
from thematic_analysis import ThematicAnalyzer
from bias_analysis import BiasAnalyzer
from visualization import Visualizer

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

GENRES_POOL = ["Drama", "Action", "Comedy", "Romance", "Thriller"]
INDIAN_NAMES = [
    "Ram Sharma", "Fatima Khan", "Priya Iyer", "Vijay Kumar",
    "Lakshmi Nair", "Ahmed Ali", "Simran Kaur", "Arjun Reddy",
    "Devi Patil", "John David", "Meera Banerjee", "Raj Singh",
    "Anita Gupta", "Hussain Sheikh", "Grace Joseph", "Ganesh Rao",
    "Rani Devi", "Harpreet Kaur", "Suresh Menon", "Pooja Verma",
]
ROLE_TYPES = ["lead", "supporting", "villain", "comic", "other"]


def _make_sample_movies(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic movies DataFrame mimicking IMDB basics."""
    rng = np.random.RandomState(seed)
    years = rng.randint(1975, 2026, size=n)
    ratings = np.round(rng.uniform(1.0, 10.0, size=n), 1)
    genres = [
        ",".join(rng.choice(GENRES_POOL, size=rng.randint(1, 4), replace=False))
        for _ in range(n)
    ]
    return pd.DataFrame({
        "tconst": [f"tt{i+1:07d}" for i in range(n)],
        "primaryTitle": [f"Movie {i+1}" for i in range(n)],
        "startYear": years,
        "genres": genres,
        "averageRating": ratings,
        "titleType": ["movie"] * n,
    })


def _make_sample_principals(movie_tconsts: list[str], seed: int = 42) -> pd.DataFrame:
    """Generate a principals-like DataFrame with character names."""
    rng = np.random.RandomState(seed)
    rows = []
    for tconst in movie_tconsts:
        n_chars = rng.randint(2, 6)
        for _ in range(n_chars):
            name = rng.choice(INDIAN_NAMES)
            role = rng.choice(ROLE_TYPES)
            rows.append({
                "tconst": tconst,
                "character_name": name,
                "role_type": role,
                "category": rng.choice(["actor", "actress"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_movies() -> pd.DataFrame:
    return _make_sample_movies(100)


@pytest.fixture(scope="module")
def sample_principals(sample_movies: pd.DataFrame) -> pd.DataFrame:
    return _make_sample_principals(sample_movies["tconst"].tolist())


@pytest.fixture(scope="module")
def cleaner() -> DataCleaner:
    return DataCleaner()


@pytest.fixture(scope="module")
def master_df(sample_movies: pd.DataFrame, cleaner: DataCleaner) -> pd.DataFrame:
    """Run standardize_columns + merge_datasets to produce master_df."""
    std = cleaner.standardize_columns(sample_movies, source="imdb_basics")
    merged = cleaner.merge_datasets({"imdb_basics": std})
    return merged


@pytest.fixture(scope="module")
def classified_df(sample_principals: pd.DataFrame) -> pd.DataFrame:
    """Classify character names from the principals DataFrame."""
    analyzer = BiasAnalyzer()
    return analyzer.classify_character_names(sample_principals["character_name"])


@pytest.fixture(scope="module")
def classified_with_meta(
    sample_principals: pd.DataFrame, classified_df: pd.DataFrame, master_df: pd.DataFrame
) -> pd.DataFrame:
    """Classified names merged with role_type and period info."""
    combined = pd.concat(
        [sample_principals.reset_index(drop=True), classified_df.reset_index(drop=True)],
        axis=1,
    )
    # Add period from master_df
    period_map = master_df.set_index("tconst")["period"].to_dict()
    combined["period"] = combined["tconst"].map(period_map)
    return combined


# ===================================================================
# 1. Pipeline: standardize_columns
# ===================================================================

class TestStandardizeColumns:
    """Verify DataCleaner.standardize_columns on the sample dataset."""

    def test_output_has_unified_columns(
        self, sample_movies: pd.DataFrame, cleaner: DataCleaner
    ) -> None:
        std = cleaner.standardize_columns(sample_movies, source="imdb_basics")
        for col in ["tconst", "title", "year", "genres", "rating", "source"]:
            assert col in std.columns, f"Missing column: {col}"

    def test_row_count_preserved(
        self, sample_movies: pd.DataFrame, cleaner: DataCleaner
    ) -> None:
        std = cleaner.standardize_columns(sample_movies, source="imdb_basics")
        assert len(std) == len(sample_movies)

    def test_source_column_filled(
        self, sample_movies: pd.DataFrame, cleaner: DataCleaner
    ) -> None:
        std = cleaner.standardize_columns(sample_movies, source="imdb_basics")
        assert (std["source"] == "imdb_basics").all()


# ===================================================================
# 2. Pipeline: merge_datasets → master_df
# ===================================================================

class TestMergeDatasets:
    """Verify merge_datasets produces a valid master_df."""

    def test_master_has_required_columns(self, master_df: pd.DataFrame) -> None:
        for col in ["tconst", "title", "year", "period", "genres", "rating", "source"]:
            assert col in master_df.columns, f"Missing column: {col}"

    def test_master_row_count(self, master_df: pd.DataFrame) -> None:
        assert len(master_df) == 100

    def test_years_in_range(self, master_df: pd.DataFrame) -> None:
        valid = master_df["year"].dropna()
        assert (valid >= 1975).all()
        assert (valid <= 2025).all()

    def test_period_column_populated(self, master_df: pd.DataFrame) -> None:
        assert master_df["period"].notna().all()


# ===================================================================
# 3. Thematic analysis: genre_trends_by_period
# ===================================================================

class TestGenreTrendsByPeriod:
    """Verify ThematicAnalyzer.genre_trends_by_period output."""

    def test_output_shape(self, master_df: pd.DataFrame) -> None:
        ta = ThematicAnalyzer()
        result = ta.genre_trends_by_period(master_df)
        assert not result.empty
        assert set(result.columns) == {"period", "genre", "count", "percentage"}

    def test_percentages_sum_to_100(self, master_df: pd.DataFrame) -> None:
        ta = ThematicAnalyzer()
        result = ta.genre_trends_by_period(master_df)
        for period, group in result.groupby("period"):
            total = group["percentage"].sum()
            assert abs(total - 100.0) < 0.01, (
                f"Period {period}: percentages sum to {total}"
            )


# ===================================================================
# 4. Thematic analysis: genre_cooccurrence_matrix
# ===================================================================

class TestGenreCooccurrenceMatrix:
    """Verify ThematicAnalyzer.genre_cooccurrence_matrix symmetry."""

    def test_symmetry(self, master_df: pd.DataFrame) -> None:
        ta = ThematicAnalyzer()
        matrix = ta.genre_cooccurrence_matrix(master_df)
        assert not matrix.empty
        # Check matrix[A][B] == matrix[B][A]
        for row in matrix.index:
            for col in matrix.columns:
                assert matrix.loc[row, col] == matrix.loc[col, row], (
                    f"Asymmetry at ({row}, {col})"
                )


# ===================================================================
# 5. Bias analysis: classify_character_names
# ===================================================================

class TestClassifyCharacterNames:
    """Verify BiasAnalyzer.classify_character_names output."""

    def test_output_size_matches_input(
        self, sample_principals: pd.DataFrame, classified_df: pd.DataFrame
    ) -> None:
        assert len(classified_df) == len(sample_principals)

    def test_required_columns(self, classified_df: pd.DataFrame) -> None:
        for col in [
            "name", "inferred_gender", "inferred_religion",
            "inferred_region", "classification_confidence",
        ]:
            assert col in classified_df.columns

    def test_confidence_bounds(self, classified_df: pd.DataFrame) -> None:
        assert (classified_df["classification_confidence"] >= 0.0).all()
        assert (classified_df["classification_confidence"] <= 1.0).all()

    def test_gender_values_valid(self, classified_df: pd.DataFrame) -> None:
        valid = {"male", "female", "neutral", "unknown"}
        assert set(classified_df["inferred_gender"].unique()).issubset(valid)


# ===================================================================
# 6. Statistical significance tests
# ===================================================================

class TestStatisticalSignificance:
    """Verify BiasAnalyzer.statistical_significance_tests output."""

    def test_p_value_in_range(self, classified_with_meta: pd.DataFrame) -> None:
        ba = BiasAnalyzer()
        result = ba.statistical_significance_tests(
            classified_with_meta,
            attribute_col="inferred_gender",
            role_col="role_type",
        )
        assert 0.0 <= result["p_value"] <= 1.0

    def test_effect_size_in_range(self, classified_with_meta: pd.DataFrame) -> None:
        ba = BiasAnalyzer()
        result = ba.statistical_significance_tests(
            classified_with_meta,
            attribute_col="inferred_gender",
            role_col="role_type",
        )
        assert 0.0 <= result["effect_size"] <= 1.0

    def test_significance_matches_alpha(self, classified_with_meta: pd.DataFrame) -> None:
        ba = BiasAnalyzer()
        result = ba.statistical_significance_tests(
            classified_with_meta,
            attribute_col="inferred_gender",
            role_col="role_type",
            alpha=0.05,
        )
        assert result["significant"] == (result["p_value"] < 0.05)

    def test_required_keys(self, classified_with_meta: pd.DataFrame) -> None:
        ba = BiasAnalyzer()
        result = ba.statistical_significance_tests(
            classified_with_meta,
            attribute_col="inferred_gender",
            role_col="role_type",
        )
        for key in [
            "chi2_statistic", "p_value", "degrees_of_freedom",
            "effect_size", "significant", "test_used",
        ]:
            assert key in result, f"Missing key: {key}"


# ===================================================================
# 7. Visualization smoke tests (all methods execute without error)
# ===================================================================

class TestVisualizationsNoError:
    """Verify all Visualizer methods execute without error on pipeline data."""

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        master_df: pd.DataFrame,
        classified_with_meta: pd.DataFrame,
    ) -> None:
        self.master_df = master_df
        self.classified_with_meta = classified_with_meta
        self.viz = Visualizer()
        self.ta = ThematicAnalyzer()
        self.ba = BiasAnalyzer()

    def test_plot_genre_trends(self) -> None:
        genre_data = self.ta.genre_trends_by_period(self.master_df)
        fig = self.viz.plot_genre_trends(genre_data)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_genre_heatmap(self) -> None:
        cooccurrence = self.ta.genre_cooccurrence_matrix(self.master_df)
        fig = self.viz.plot_genre_heatmap(cooccurrence)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_topic_wordclouds(self) -> None:
        # Use fallback since our sample has no plot text
        topic_result = self.ta.topic_model_plots(self.master_df, n_topics=3)
        fig = self.viz.plot_topic_wordclouds(topic_result.get("topic_words", {}))
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_role_distribution(self) -> None:
        role_data = self.ba.role_distribution_by_gender(self.classified_with_meta)
        fig = self.viz.plot_role_distribution(role_data, group_by="gender")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_temporal_bias(self) -> None:
        bias_data = self.ba.temporal_bias_trends(self.classified_with_meta)
        fig = self.viz.plot_temporal_bias(bias_data)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_name_frequency(self) -> None:
        name_data = self.ba.name_frequency_analysis(self.classified_with_meta)
        fig = self.viz.plot_name_frequency(name_data)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_summary_dashboard(self) -> None:
        genre_trends = self.ta.genre_trends_by_period(self.master_df)
        bias_data = self.ba.temporal_bias_trends(self.classified_with_meta)
        sig = self.ba.statistical_significance_tests(
            self.classified_with_meta,
            attribute_col="inferred_gender",
            role_col="role_type",
        )
        fig = self.viz.plot_summary_dashboard({
            "genre_trends": genre_trends,
            "bias_results": sig,
            "temporal_bias": bias_data,
        })
        assert isinstance(fig, matplotlib.figure.Figure)
