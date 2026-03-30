"""Property-based tests for the Character & Role Bias Analysis module.

Uses Hypothesis to verify universal correctness properties of the
BiasAnalyzer.classify_character_names method.

Properties tested:
- Property 7: Character Classification Determinism (Validates: Requirement 8.6)
- Property 8: Classification Output Size Preservation (Validates: Requirement 8.7)
- Property 9: Classification Confidence Bounds (Validates: Requirements 8.4, 8.5)
- Property 10: Classification Attribute Validity (Validates: Requirements 8.1, 8.2, 8.3)
- Property 16: Representation Index Correctness (Validates: Requirement 9.4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from bias_analysis import BiasAnalyzer

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Indian name fragments for realistic inputs
_INDIAN_FIRST_NAMES = [
    "Ram", "Krishna", "Priya", "Devi", "Kumar", "Raj", "Fatima",
    "Harpreet", "John", "Mary", "Lakshmi", "Arjun", "Singh",
    "Begum", "Kaur", "Ahmed", "Ganesh", "Parvati",
]

_INDIAN_SURNAMES = [
    "Sharma", "Gupta", "Verma", "Iyer", "Reddy", "Rao",
    "Banerjee", "Mukherjee", "Patil", "Kulkarni", "Khan",
    "Das", "Sen", "Nair", "Menon", "Deshmukh", "Joshi",
]

_indian_first = st.sampled_from(_INDIAN_FIRST_NAMES)
_indian_surname = st.sampled_from(_INDIAN_SURNAMES)


@st.composite
def _indian_name(draw):
    """Draw a realistic Indian-style name (first + optional surname)."""
    first = draw(_indian_first)
    include_surname = draw(st.booleans())
    if include_surname:
        surname = draw(_indian_surname)
        return f"{first} {surname}"
    return first


# A name value that can be a random string, Indian name, None, or empty
_name_value = st.one_of(
    st.none(),
    st.just(""),
    st.just("   "),
    st.text(min_size=0, max_size=30),
    _indian_name(),
)


@st.composite
def _names_series(draw, min_size=0, max_size=50):
    """Draw a pd.Series of name values (mixed None, empty, text, Indian names)."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(st.lists(_name_value, min_size=n, max_size=n))
    return pd.Series(values, dtype=object)


# ---------------------------------------------------------------------------
# Property 7: Character Classification Determinism
# ---------------------------------------------------------------------------


class TestClassificationDeterminism:
    """**Validates: Requirement 8.6**

    Property 7: For any character name string, calling the classification
    function twice with the same input must produce identical inferred
    gender, religion, region, and confidence values.
    """

    @given(name=st.one_of(st.none(), st.text(min_size=0, max_size=50)))
    @settings(max_examples=100, deadline=None)
    def test_same_name_always_produces_identical_output(self, name) -> None:
        analyzer = BiasAnalyzer()
        series = pd.Series([name], dtype=object)

        result1 = analyzer.classify_character_names(series)
        result2 = analyzer.classify_character_names(series)

        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Property 8: Classification Output Size Preservation
# ---------------------------------------------------------------------------


class TestClassificationOutputSizePreservation:
    """**Validates: Requirement 8.7**

    Property 8: For any input Series of N character names, the
    classify_character_names function must return a DataFrame with
    exactly N rows.
    """

    @given(names=_names_series(min_size=0, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_output_row_count_equals_input(self, names: pd.Series) -> None:
        analyzer = BiasAnalyzer()
        result = analyzer.classify_character_names(names)
        assert len(result) == len(names), (
            f"Output rows ({len(result)}) != input rows ({len(names)})"
        )


# ---------------------------------------------------------------------------
# Property 9: Classification Confidence Bounds
# ---------------------------------------------------------------------------


class TestClassificationConfidenceBounds:
    """**Validates: Requirements 8.4, 8.5**

    Property 9: For any character name input (including empty, null, or
    unrecognized names), the classification confidence must be in the
    range [0.0, 1.0].
    """

    @given(name=st.one_of(st.none(), st.text(min_size=0, max_size=50)))
    @settings(max_examples=100, deadline=None)
    def test_confidence_always_in_bounds(self, name) -> None:
        analyzer = BiasAnalyzer()
        result = analyzer.classify_character_names(pd.Series([name], dtype=object))
        conf = result.iloc[0]["classification_confidence"]
        assert 0.0 <= conf <= 1.0, (
            f"Confidence {conf} out of bounds [0.0, 1.0] for name={name!r}"
        )


# ---------------------------------------------------------------------------
# Property 10: Classification Attribute Validity
# ---------------------------------------------------------------------------


VALID_GENDERS = {"male", "female", "neutral", "unknown"}
VALID_RELIGIONS = {"hindu", "muslim", "sikh", "christian", "unknown"}
VALID_REGIONS = {"north_indian", "south_indian", "bengali", "maharashtrian", "unknown"}


class TestClassificationAttributeValidity:
    """**Validates: Requirements 8.1, 8.2, 8.3**

    Property 10: For any classified character:
    - inferred_gender must be in {"male", "female", "neutral", "unknown"}
    - inferred_religion must be in {"hindu", "muslim", "sikh", "christian", "unknown"}
    - inferred_region must be in {"north_indian", "south_indian", "bengali",
      "maharashtrian", "unknown"}
    """

    @given(name=st.one_of(st.none(), st.text(min_size=0, max_size=50)))
    @settings(max_examples=100, deadline=None)
    def test_all_attributes_in_valid_sets(self, name) -> None:
        analyzer = BiasAnalyzer()
        result = analyzer.classify_character_names(pd.Series([name], dtype=object))
        row = result.iloc[0]

        assert row["inferred_gender"] in VALID_GENDERS, (
            f"Invalid gender {row['inferred_gender']!r} for name={name!r}"
        )
        assert row["inferred_religion"] in VALID_RELIGIONS, (
            f"Invalid religion {row['inferred_religion']!r} for name={name!r}"
        )
        assert row["inferred_region"] in VALID_REGIONS, (
            f"Invalid region {row['inferred_region']!r} for name={name!r}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 16
# ---------------------------------------------------------------------------

_ROLE_TYPES = ["lead", "supporting", "villain", "comic"]
_GENDERS = ["male", "female", "unknown"]


@st.composite
def _role_gender_dataframe(draw, min_size=2, max_size=60):
    """Draw a DataFrame with role_type and inferred_gender columns."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    roles = draw(st.lists(st.sampled_from(_ROLE_TYPES), min_size=n, max_size=n))
    genders = draw(st.lists(st.sampled_from(_GENDERS), min_size=n, max_size=n))
    return pd.DataFrame({"role_type": roles, "inferred_gender": genders})


# ---------------------------------------------------------------------------
# Property 16: Representation Index Correctness
# ---------------------------------------------------------------------------


class TestRepresentationIndexCorrectness:
    """**Validates: Requirement 9.4**

    Property 16: For any group in the Character_DataFrame, the
    representation index must equal the ratio of that group's role share
    to its population share.  Additionally, role_share and
    population_share must each be in [0.0, 1.0].
    """

    @given(df=_role_gender_dataframe(min_size=2, max_size=60))
    @settings(max_examples=100, deadline=None)
    def test_representation_index_equals_role_share_over_population_share(
        self, df: pd.DataFrame
    ) -> None:
        analyzer = BiasAnalyzer()
        result = analyzer.representation_index(df)

        for _, row in result.iterrows():
            # role_share and population_share must be in [0.0, 1.0]
            assert 0.0 <= row["role_share"] <= 1.0, (
                f"role_share {row['role_share']} out of [0.0, 1.0] "
                f"for group={row['group']!r}, role_type={row['role_type']!r}"
            )
            assert 0.0 <= row["population_share"] <= 1.0, (
                f"population_share {row['population_share']} out of [0.0, 1.0] "
                f"for group={row['group']!r}, role_type={row['role_type']!r}"
            )

            # representation_index == role_share / population_share
            expected = row["role_share"] / row["population_share"]
            assert abs(row["representation_index"] - expected) < 1e-9, (
                f"representation_index {row['representation_index']} != "
                f"role_share/population_share ({expected}) "
                f"for group={row['group']!r}, role_type={row['role_type']!r}"
            )


# ---------------------------------------------------------------------------
# Strategies for Property 11
# ---------------------------------------------------------------------------

_ATTRIBUTE_VALUES = ["a", "b", "c", "d"]
_ROLE_VALUES = ["x", "y", "z", "w"]


@st.composite
def _contingency_dataframe(draw):
    """Draw a DataFrame suitable for statistical_significance_tests.

    Generates a table with ``attribute_col`` and ``role_col`` where each
    column has 2-4 unique values and every (attribute, role) combination
    has at least 2 rows.

    **Validates: Requirements 10.1, 10.2, 10.3**
    """
    n_attrs = draw(st.integers(min_value=2, max_value=4))
    n_roles = draw(st.integers(min_value=2, max_value=4))
    attrs = _ATTRIBUTE_VALUES[:n_attrs]
    roles = _ROLE_VALUES[:n_roles]

    # Build at least 2 rows per (attr, role) combination
    rows_attr: list[str] = []
    rows_role: list[str] = []
    for a in attrs:
        for r in roles:
            count = draw(st.integers(min_value=2, max_value=8))
            rows_attr.extend([a] * count)
            rows_role.extend([r] * count)

    return pd.DataFrame({"attribute": rows_attr, "role": rows_role})


# ---------------------------------------------------------------------------
# Property 11: Statistical Test Output Validity
# ---------------------------------------------------------------------------


class TestStatisticalTestOutputValidity:
    """**Validates: Requirements 10.1, 10.2, 10.3**

    Property 11: For any valid contingency table input to the statistical
    significance test, the returned p-value must be in [0.0, 1.0], the
    effect size (Cramér's V) must be in [0.0, 1.0], and the significance
    boolean must be True if and only if p_value < alpha.
    """

    @given(df=_contingency_dataframe(), alpha=st.floats(min_value=0.001, max_value=0.999))
    @settings(max_examples=100, deadline=None)
    def test_p_value_and_effect_size_in_bounds_and_significance_correct(
        self, df: pd.DataFrame, alpha: float
    ) -> None:
        analyzer = BiasAnalyzer()
        result = analyzer.statistical_significance_tests(
            df, "attribute", "role", alpha=alpha
        )

        # p-value in [0.0, 1.0]
        assert 0.0 <= result["p_value"] <= 1.0, (
            f"p_value {result['p_value']} out of [0.0, 1.0]"
        )

        # Cramér's V in [0.0, 1.0]
        assert 0.0 <= result["effect_size"] <= 1.0, (
            f"effect_size {result['effect_size']} out of [0.0, 1.0]"
        )

        # significant iff p_value < alpha
        assert result["significant"] == (result["p_value"] < alpha), (
            f"significant={result['significant']} but p_value={result['p_value']}, "
            f"alpha={alpha}"
        )
