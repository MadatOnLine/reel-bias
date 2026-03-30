"""Unit tests for the BiasAnalyzer.classify_character_names method.

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bias_analysis import BiasAnalyzer


@pytest.fixture
def analyzer() -> BiasAnalyzer:
    return BiasAnalyzer()


# ===================================================================
# 7.1  classify_character_names
# ===================================================================


class TestClassifyCharacterNames:
    """Core tests for the classify_character_names method."""

    # --- Output shape and columns (Req 8.7) ---

    def test_output_row_count_matches_input(self, analyzer: BiasAnalyzer) -> None:
        names = pd.Series(["Ram Sharma", "Fatima Begum", "Unknown"])
        result = analyzer.classify_character_names(names)
        assert len(result) == len(names)

    def test_output_columns(self, analyzer: BiasAnalyzer) -> None:
        names = pd.Series(["Ram"])
        result = analyzer.classify_character_names(names)
        expected_cols = {
            "name", "inferred_gender", "inferred_religion",
            "inferred_region", "classification_confidence",
        }
        assert set(result.columns) == expected_cols

    def test_empty_series(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series([], dtype=str))
        assert len(result) == 0
        expected_cols = {
            "name", "inferred_gender", "inferred_religion",
            "inferred_region", "classification_confidence",
        }
        assert set(result.columns) == expected_cols

    # --- Gender inference (Req 8.1) ---

    def test_male_pattern_kumar(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Vijay Kumar"]))
        assert result.iloc[0]["inferred_gender"] == "male"

    def test_male_pattern_raj(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Raj Kapoor"]))
        assert result.iloc[0]["inferred_gender"] == "male"

    def test_male_pattern_dev(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Mahadev"]))
        assert result.iloc[0]["inferred_gender"] == "male"

    def test_female_pattern_devi(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Lakshmi Devi"]))
        assert result.iloc[0]["inferred_gender"] == "female"

    def test_female_pattern_priya(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Priya"]))
        assert result.iloc[0]["inferred_gender"] == "female"

    def test_female_pattern_rani(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Rani Mukherjee"]))
        assert result.iloc[0]["inferred_gender"] == "female"

    # --- Religion inference (Req 8.2) ---

    def test_hindu_pattern_ram(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Ram Prasad"]))
        assert result.iloc[0]["inferred_religion"] == "hindu"

    def test_hindu_pattern_krishna(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Krishna Murthy"]))
        assert result.iloc[0]["inferred_religion"] == "hindu"

    def test_muslim_pattern_khan(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Rahim Khan"]))
        assert result.iloc[0]["inferred_religion"] == "muslim"

    def test_muslim_pattern_fatima(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Fatima Begum"]))
        assert result.iloc[0]["inferred_religion"] == "muslim"

    def test_sikh_pattern_kaur(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Harpreet Kaur"]))
        assert result.iloc[0]["inferred_religion"] == "sikh"

    def test_christian_pattern_john(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["John D'Souza"]))
        assert result.iloc[0]["inferred_religion"] == "christian"

    def test_christian_pattern_mary(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Mary Thomas"]))
        assert result.iloc[0]["inferred_religion"] == "christian"

    # --- Region inference (Req 8.3) ---

    def test_north_indian_sharma(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Amit Sharma"]))
        assert result.iloc[0]["inferred_region"] == "north_indian"

    def test_north_indian_gupta(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Rahul Gupta"]))
        assert result.iloc[0]["inferred_region"] == "north_indian"

    def test_south_indian_iyer(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Venkat Iyer"]))
        assert result.iloc[0]["inferred_region"] == "south_indian"

    def test_south_indian_reddy(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Srinivas Reddy"]))
        assert result.iloc[0]["inferred_region"] == "south_indian"

    def test_bengali_banerjee(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Arup Banerjee"]))
        assert result.iloc[0]["inferred_region"] == "bengali"

    def test_maharashtrian_patil(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Sachin Patil"]))
        assert result.iloc[0]["inferred_region"] == "maharashtrian"

    # --- Confidence bounds (Req 8.4) ---

    def test_confidence_in_range(self, analyzer: BiasAnalyzer) -> None:
        names = pd.Series(["Ram Sharma", "Unknown Person", "", None, "Priya Devi"])
        result = analyzer.classify_character_names(names)
        assert (result["classification_confidence"] >= 0.0).all()
        assert (result["classification_confidence"] <= 1.0).all()

    def test_matched_name_has_positive_confidence(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Ram Sharma"]))
        assert result.iloc[0]["classification_confidence"] > 0.0

    # --- Unknown / null / empty handling (Req 8.5) ---

    def test_null_name_returns_unknown(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series([None]))
        row = result.iloc[0]
        assert row["inferred_gender"] == "unknown"
        assert row["inferred_religion"] == "unknown"
        assert row["inferred_region"] == "unknown"
        assert row["classification_confidence"] == 0.0

    def test_empty_string_returns_unknown(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series([""]))
        row = result.iloc[0]
        assert row["inferred_gender"] == "unknown"
        assert row["inferred_religion"] == "unknown"
        assert row["inferred_region"] == "unknown"
        assert row["classification_confidence"] == 0.0

    def test_whitespace_only_returns_unknown(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["   "]))
        row = result.iloc[0]
        assert row["inferred_gender"] == "unknown"
        assert row["classification_confidence"] == 0.0

    def test_unrecognized_name_returns_unknown(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["Xyzabc"]))
        row = result.iloc[0]
        assert row["inferred_gender"] == "unknown"
        assert row["inferred_religion"] == "unknown"
        assert row["inferred_region"] == "unknown"
        assert row["classification_confidence"] == 0.0

    def test_nan_returns_unknown(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series([np.nan]))
        row = result.iloc[0]
        assert row["inferred_gender"] == "unknown"
        assert row["classification_confidence"] == 0.0

    # --- Determinism (Req 8.6) ---

    def test_deterministic_output(self, analyzer: BiasAnalyzer) -> None:
        names = pd.Series(["Ram Sharma", "Fatima Khan", "Priya Iyer", None, ""])
        result1 = analyzer.classify_character_names(names)
        result2 = analyzer.classify_character_names(names)
        pd.testing.assert_frame_equal(result1, result2)

    # --- Combined attribute classification ---

    def test_multiple_attributes_detected(self, analyzer: BiasAnalyzer) -> None:
        """A name like 'Ram Kumar Sharma' should match gender, religion, and region."""
        result = analyzer.classify_character_names(pd.Series(["Ram Kumar Sharma"]))
        row = result.iloc[0]
        assert row["inferred_gender"] == "male"  # kumar pattern
        assert row["inferred_religion"] == "hindu"  # ram pattern
        assert row["inferred_region"] == "north_indian"  # sharma pattern
        assert row["classification_confidence"] > 0.0

    def test_case_insensitive(self, analyzer: BiasAnalyzer) -> None:
        result = analyzer.classify_character_names(pd.Series(["RAM SHARMA"]))
        row = result.iloc[0]
        assert row["inferred_religion"] == "hindu"
        assert row["inferred_region"] == "north_indian"

    def test_mixed_batch(self, analyzer: BiasAnalyzer) -> None:
        names = pd.Series([
            "Ram Sharma",
            "Fatima Begum",
            "Harpreet Kaur",
            "John D'Souza",
            None,
            "",
            "Xyzabc",
        ])
        result = analyzer.classify_character_names(names)
        assert len(result) == 7
        # Verify all genders are valid
        valid_genders = {"male", "female", "neutral", "unknown"}
        assert set(result["inferred_gender"]).issubset(valid_genders)


# ===================================================================
# 7.6  Role distribution & bias analysis methods
# ===================================================================


class TestRoleDistributionByGender:
    """Tests for role_distribution_by_gender (Req 9.1)."""

    def test_crosstab_shape(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "role_type": ["lead", "lead", "supporting", "supporting"],
            "inferred_gender": ["male", "female", "male", "female"],
        })
        result = analyzer.role_distribution_by_gender(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_crosstab_values(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "role_type": ["lead", "lead", "lead", "supporting"],
            "inferred_gender": ["male", "male", "female", "male"],
        })
        result = analyzer.role_distribution_by_gender(df)
        assert result.loc["lead", "male"] == 2
        assert result.loc["lead", "female"] == 1
        assert result.loc["supporting", "male"] == 1

    def test_single_role_type(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "role_type": ["villain", "villain"],
            "inferred_gender": ["male", "female"],
        })
        result = analyzer.role_distribution_by_gender(df)
        assert len(result) == 1
        assert result.loc["villain", "male"] == 1


class TestRoleDistributionByReligion:
    """Tests for role_distribution_by_religion (Req 9.2)."""

    def test_crosstab_shape(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "role_type": ["lead", "lead", "supporting"],
            "inferred_religion": ["hindu", "muslim", "hindu"],
        })
        result = analyzer.role_distribution_by_religion(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_crosstab_values(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "role_type": ["lead", "lead", "villain", "villain"],
            "inferred_religion": ["hindu", "muslim", "hindu", "hindu"],
        })
        result = analyzer.role_distribution_by_religion(df)
        assert result.loc["lead", "hindu"] == 1
        assert result.loc["lead", "muslim"] == 1
        assert result.loc["villain", "hindu"] == 2


class TestTemporalBiasTrends:
    """Tests for temporal_bias_trends (Req 9.3)."""

    def test_output_columns(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "period": ["1980-1984", "1980-1984", "1985-1989"],
            "inferred_gender": ["male", "female", "male"],
            "inferred_religion": ["hindu", "muslim", "hindu"],
            "inferred_region": ["north_indian", "south_indian", "bengali"],
        })
        result = analyzer.temporal_bias_trends(df)
        assert set(result.columns) == {"period", "attribute", "value", "count", "percentage"}

    def test_percentages_sum_to_100(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "period": ["1980-1984"] * 4,
            "inferred_gender": ["male", "male", "female", "female"],
            "inferred_religion": ["hindu", "muslim", "hindu", "muslim"],
            "inferred_region": ["north_indian", "south_indian", "north_indian", "south_indian"],
        })
        result = analyzer.temporal_bias_trends(df)
        for (period, attr), grp in result.groupby(["period", "attribute"]):
            assert abs(grp["percentage"].sum() - 100.0) < 0.01

    def test_counts_are_correct(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "period": ["1980-1984", "1980-1984", "1980-1984"],
            "inferred_gender": ["male", "male", "female"],
            "inferred_religion": ["hindu", "hindu", "muslim"],
            "inferred_region": ["north_indian", "north_indian", "south_indian"],
        })
        result = analyzer.temporal_bias_trends(df)
        gender_rows = result[result["attribute"] == "inferred_gender"]
        male_row = gender_rows[gender_rows["value"] == "male"]
        assert male_row.iloc[0]["count"] == 2

    def test_empty_input(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame(columns=["period", "inferred_gender", "inferred_religion", "inferred_region"])
        result = analyzer.temporal_bias_trends(df)
        assert len(result) == 0
        assert set(result.columns) == {"period", "attribute", "value", "count", "percentage"}


class TestNameFrequencyAnalysis:
    """Tests for name_frequency_analysis (Req 9.3)."""

    def test_output_columns(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "name": ["Ram", "Ram", "Sita"],
            "period": ["1980-1984", "1980-1984", "1980-1984"],
            "inferred_gender": ["male", "male", "female"],
            "role_type": ["lead", "lead", "lead"],
        })
        result = analyzer.name_frequency_analysis(df)
        assert list(result.columns) == ["period", "inferred_gender", "role_type", "name", "count"]

    def test_top_names_correct(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "name": ["Ram", "Ram", "Ram", "Sita", "Sita"],
            "period": ["1980-1984"] * 5,
            "inferred_gender": ["male"] * 3 + ["female"] * 2,
            "role_type": ["lead"] * 5,
        })
        result = analyzer.name_frequency_analysis(df)
        male_lead = result[
            (result["inferred_gender"] == "male") & (result["role_type"] == "lead")
        ]
        assert male_lead.iloc[0]["name"] == "Ram"
        assert male_lead.iloc[0]["count"] == 3

    def test_limits_to_top_10(self, analyzer: BiasAnalyzer) -> None:
        names = [f"Name{i}" for i in range(15)]
        df = pd.DataFrame({
            "name": names,
            "period": ["1980-1984"] * 15,
            "inferred_gender": ["male"] * 15,
            "role_type": ["lead"] * 15,
        })
        result = analyzer.name_frequency_analysis(df)
        group = result[
            (result["period"] == "1980-1984")
            & (result["inferred_gender"] == "male")
            & (result["role_type"] == "lead")
        ]
        assert len(group) <= 10


class TestRepresentationIndex:
    """Tests for representation_index (Req 9.4)."""

    def test_output_columns(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "role_type": ["lead", "lead", "supporting", "supporting"],
            "inferred_gender": ["male", "female", "male", "female"],
        })
        result = analyzer.representation_index(df)
        expected_cols = {"attribute", "group", "role_type", "role_share", "population_share", "representation_index"}
        assert set(result.columns) == expected_cols

    def test_proportional_representation_equals_one(self, analyzer: BiasAnalyzer) -> None:
        """When groups are equally distributed across roles, index should be 1.0."""
        df = pd.DataFrame({
            "role_type": ["lead", "lead", "supporting", "supporting"],
            "inferred_gender": ["male", "female", "male", "female"],
        })
        result = analyzer.representation_index(df)
        for _, row in result.iterrows():
            assert abs(row["representation_index"] - 1.0) < 1e-9

    def test_overrepresentation(self, analyzer: BiasAnalyzer) -> None:
        """If males are 50% of population but 100% of leads, index > 1."""
        df = pd.DataFrame({
            "role_type": ["lead", "lead", "supporting", "supporting"],
            "inferred_gender": ["male", "male", "male", "female"],
        })
        result = analyzer.representation_index(df)
        male_lead = result[
            (result["group"] == "male") & (result["role_type"] == "lead")
        ]
        assert male_lead.iloc[0]["representation_index"] > 1.0

    def test_empty_input(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame(columns=["role_type", "inferred_gender"])
        result = analyzer.representation_index(df)
        assert len(result) == 0

    def test_role_share_and_population_share(self, analyzer: BiasAnalyzer) -> None:
        """Verify the math: role_share / population_share = representation_index."""
        df = pd.DataFrame({
            "role_type": ["lead", "lead", "lead", "supporting"],
            "inferred_gender": ["male", "male", "female", "male"],
        })
        result = analyzer.representation_index(df)
        for _, row in result.iterrows():
            expected = row["role_share"] / row["population_share"]
            assert abs(row["representation_index"] - expected) < 1e-9


# ===================================================================
# 8.1  statistical_significance_tests
# ===================================================================


class TestStatisticalSignificanceTests:
    """Tests for statistical_significance_tests (Reqs 10.1, 10.2, 10.3, 10.4, 13.4)."""

    def test_returns_required_keys(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "attr": ["a", "a", "b", "b"] * 5,
            "role": ["x", "y", "x", "y"] * 5,
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role")
        expected_keys = {
            "chi2_statistic", "p_value", "degrees_of_freedom",
            "effect_size", "significant", "test_used",
        }
        assert set(result.keys()) == expected_keys

    def test_p_value_in_range(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "attr": ["a", "a", "b", "b"] * 10,
            "role": ["x", "y", "x", "y"] * 10,
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role")
        assert 0.0 <= result["p_value"] <= 1.0

    def test_effect_size_in_range(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "attr": ["a", "a", "b", "b"] * 10,
            "role": ["x", "y", "x", "y"] * 10,
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role")
        assert 0.0 <= result["effect_size"] <= 1.0

    def test_significance_matches_alpha(self, analyzer: BiasAnalyzer) -> None:
        df = pd.DataFrame({
            "attr": ["a", "a", "b", "b"] * 10,
            "role": ["x", "y", "x", "y"] * 10,
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role", alpha=0.05)
        assert result["significant"] == (result["p_value"] < 0.05)

    def test_chi_square_used_for_normal_table(self, analyzer: BiasAnalyzer) -> None:
        """Large enough counts → chi-square test."""
        df = pd.DataFrame({
            "attr": ["a"] * 50 + ["b"] * 50,
            "role": ["x"] * 25 + ["y"] * 25 + ["x"] * 25 + ["y"] * 25,
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role")
        assert result["test_used"] == "chi_square"

    def test_fisher_exact_for_2x2_low_expected(self, analyzer: BiasAnalyzer) -> None:
        """2×2 table with very small counts → Fisher's exact test."""
        df = pd.DataFrame({
            "attr": ["a", "a", "b", "b"],
            "role": ["x", "x", "x", "y"],
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role")
        assert result["test_used"] == "fisher_exact"

    def test_chi_square_low_expected_for_larger_table(self, analyzer: BiasAnalyzer) -> None:
        """Larger-than-2×2 table with low expected → chi_square_low_expected."""
        df = pd.DataFrame({
            "attr": ["a", "a", "b", "b", "c", "c"],
            "role": ["x", "y", "x", "y", "x", "y"],
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role")
        # 3×2 table with 1 per cell → low expected
        assert result["test_used"] == "chi_square_low_expected"

    def test_degrees_of_freedom(self, analyzer: BiasAnalyzer) -> None:
        """dof should be (r-1)*(c-1)."""
        df = pd.DataFrame({
            "attr": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
            "role": (["x"] * 10 + ["y"] * 10) * 3,
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role")
        # 3 attrs × 2 roles → dof = (3-1)*(2-1) = 2
        assert result["degrees_of_freedom"] == 2

    def test_perfectly_independent_not_significant(self, analyzer: BiasAnalyzer) -> None:
        """Perfectly balanced table should not be significant."""
        df = pd.DataFrame({
            "attr": ["a"] * 100 + ["b"] * 100,
            "role": (["x"] * 50 + ["y"] * 50) * 2,
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role")
        assert result["significant"] is False
        assert result["chi2_statistic"] == pytest.approx(0.0, abs=1e-9)

    def test_strongly_associated_is_significant(self, analyzer: BiasAnalyzer) -> None:
        """Perfectly associated table should be significant."""
        df = pd.DataFrame({
            "attr": ["a"] * 50 + ["b"] * 50,
            "role": ["x"] * 50 + ["y"] * 50,
        })
        result = analyzer.statistical_significance_tests(df, "attr", "role")
        assert result["significant"] is True
        assert result["effect_size"] > 0.5
