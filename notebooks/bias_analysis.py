"""Character & Role Bias Analysis module for the Indian Movies IMDB Research project.

Classifies character names by inferred gender, religion, and region using
name-pattern heuristics and lookup tables. Provides role distribution analysis,
temporal bias trends, and statistical significance testing.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 9.1, 9.2, 9.3, 9.4,
              10.1, 10.2, 10.3, 10.4, 13.4
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup tables for name-pattern heuristics
# ---------------------------------------------------------------------------

GENDER_INDICATORS: dict[str, list[str]] = {
    "male_patterns": ["kumar", "raj", "singh", "dev", "nath", "esh", "endra"],
    "female_patterns": ["devi", "kumari", "bai", "amma", "priya", "lata", "rani"],
}

RELIGION_INDICATORS: dict[str, list[str]] = {
    "hindu": ["ram", "krishna", "shiva", "lakshmi", "ganesh", "parvati", "arjun"],
    "muslim": ["khan", "ahmed", "fatima", "ali", "hussain", "begum", "sheikh"],
    "sikh": ["singh", "kaur", "gurpreet", "harpreet", "amarjeet"],
    "christian": ["john", "mary", "joseph", "david", "peter", "grace"],
}

REGION_INDICATORS: dict[str, list[str]] = {
    "north_indian": ["sharma", "verma", "gupta", "singh", "kumar"],
    "south_indian": ["iyer", "iyengar", "nair", "menon", "reddy", "rao"],
    "bengali": ["chatterjee", "banerjee", "mukherjee", "das", "sen"],
    "maharashtrian": ["patil", "deshmukh", "kulkarni", "joshi"],
}

# Valid attribute values
VALID_GENDERS = {"male", "female", "neutral", "unknown"}
VALID_RELIGIONS = {"hindu", "muslim", "sikh", "christian", "unknown"}
VALID_REGIONS = {"north_indian", "south_indian", "bengali", "maharashtrian", "unknown"}


class BiasAnalyzer:
    """Analyzes character names for gender, religious, and regional biases.

    Uses name-pattern heuristics and lookup tables to classify character names,
    then provides role distribution analysis and statistical testing.
    """

    def classify_character_names(self, names: pd.Series) -> pd.DataFrame:
        """Classify character names by inferred gender, religion, and region.

        Parameters
        ----------
        names : pd.Series
            Series of character name strings.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``name``, ``inferred_gender``,
            ``inferred_religion``, ``inferred_region``,
            ``classification_confidence``.
            Row count is always equal to ``len(names)``.

        Notes
        -----
        * Empty, null, or unrecognized names get ``"unknown"`` for all
          attributes and confidence ``0.0`` (Req 8.5).
        * Classification is deterministic: same input always yields same
          output (Req 8.6).
        * Confidence is the maximum of individual attribute confidences,
          clamped to [0.0, 1.0] (Req 8.4).

        Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
        """
        results: list[tuple[str, str, str, str, float]] = []

        for raw_name in names:
            # Handle null / empty
            if pd.isna(raw_name) or str(raw_name).strip() == "":
                results.append(("", "unknown", "unknown", "unknown", 0.0))
                continue

            name_str = str(raw_name).strip()
            name_lower = name_str.lower()
            tokens = name_lower.split()

            # --- Gender classification (Req 8.1) ---
            # Check female patterns first to avoid substring conflicts
            # (e.g. "dev" is a substring of "devi").
            gender = "unknown"
            gender_conf = 0.0
            for token in tokens:
                if gender != "unknown":
                    break
                for pattern in GENDER_INDICATORS["female_patterns"]:
                    if pattern in token:
                        gender = "female"
                        gender_conf = 0.7
                        break
                if gender != "unknown":
                    break
                for pattern in GENDER_INDICATORS["male_patterns"]:
                    if pattern in token:
                        gender = "male"
                        gender_conf = 0.7
                        break

            # --- Religion classification (Req 8.2) ---
            religion = "unknown"
            religion_conf = 0.0
            for rel, patterns in RELIGION_INDICATORS.items():
                if religion != "unknown":
                    break
                for token in tokens:
                    if any(p in token for p in patterns):
                        religion = rel
                        religion_conf = 0.6
                        break

            # --- Region classification (Req 8.3) ---
            region = "unknown"
            region_conf = 0.0
            for reg, patterns in REGION_INDICATORS.items():
                if region != "unknown":
                    break
                for token in tokens:
                    if any(p in token for p in patterns):
                        region = reg
                        region_conf = 0.6
                        break

            # Overall confidence = max of individual confidences (Req 8.4)
            avg_conf = max(gender_conf, religion_conf, region_conf)

            results.append((name_str, gender, religion, region, avg_conf))

        df = pd.DataFrame(
            results,
            columns=[
                "name",
                "inferred_gender",
                "inferred_religion",
                "inferred_region",
                "classification_confidence",
            ],
        )
        return df

    # ------------------------------------------------------------------
    # Role distribution analysis (Requirements 9.1, 9.2, 9.3, 9.4)
    # ------------------------------------------------------------------

    def role_distribution_by_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-tabulation of role_type vs inferred_gender.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``role_type`` and ``inferred_gender``.

        Returns
        -------
        pd.DataFrame
            Crosstab with role_type as rows and inferred_gender as columns.

        Validates: Requirement 9.1
        """
        return pd.crosstab(df["role_type"], df["inferred_gender"])

    def role_distribution_by_religion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-tabulation of role_type vs inferred_religion.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``role_type`` and ``inferred_religion``.

        Returns
        -------
        pd.DataFrame
            Crosstab with role_type as rows and inferred_religion as columns.

        Validates: Requirement 9.2
        """
        return pd.crosstab(df["role_type"], df["inferred_religion"])

    def temporal_bias_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track character attribute distributions across 5-year period bins.

        For each period and each attribute (inferred_gender, inferred_religion,
        inferred_region), computes the count and percentage of each value.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``period``, ``inferred_gender``,
            ``inferred_religion``, ``inferred_region``.

        Returns
        -------
        pd.DataFrame
            Columns: period, attribute, value, count, percentage.

        Validates: Requirement 9.3
        """
        attribute_cols = ["inferred_gender", "inferred_religion", "inferred_region"]
        parts: list[pd.DataFrame] = []

        for attr in attribute_cols:
            if attr not in df.columns:
                continue
            counts = (
                df.groupby(["period", attr])
                .size()
                .reset_index(name="count")
            )
            counts.rename(columns={attr: "value"}, inplace=True)
            counts["attribute"] = attr

            # Compute percentage within each period
            period_totals = counts.groupby("period")["count"].transform("sum")
            counts["percentage"] = (counts["count"] / period_totals * 100.0)

            parts.append(counts)

        if not parts:
            return pd.DataFrame(
                columns=["period", "attribute", "value", "count", "percentage"]
            )

        result = pd.concat(parts, ignore_index=True)
        result = result[["period", "attribute", "value", "count", "percentage"]]
        result.sort_values(["period", "attribute", "value"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def name_frequency_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Most common character names by period, gender, and role type.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``name``, ``period``, ``inferred_gender``,
            ``role_type``.

        Returns
        -------
        pd.DataFrame
            Columns: period, inferred_gender, role_type, name, count.
            Top 10 names per (period, inferred_gender, role_type) group.

        Validates: Requirement 9.3
        """
        group_cols = ["period", "inferred_gender", "role_type"]
        counts = (
            df.groupby(group_cols + ["name"])
            .size()
            .reset_index(name="count")
        )

        # Keep top 10 names per group
        top = (
            counts
            .sort_values(group_cols + ["count"], ascending=[True, True, True, False])
            .groupby(group_cols, sort=False)
            .head(10)
        )
        top.reset_index(drop=True, inplace=True)
        return top[group_cols + ["name", "count"]]

    def representation_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute representation index for each group and role type.

        For each (attribute, group, role_type):
            role_share   = group's count in that role / total count in that role
            population_share = group's total count / overall total count
            representation_index = role_share / population_share

        A value of 1.0 means proportional representation; >1 means
        over-represented in that role; <1 means under-represented.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``role_type`` and at least one of
            ``inferred_gender``, ``inferred_religion``, ``inferred_region``.

        Returns
        -------
        pd.DataFrame
            Columns: attribute, group, role_type, role_share,
            population_share, representation_index.

        Validates: Requirement 9.4
        """
        attribute_cols = ["inferred_gender", "inferred_religion", "inferred_region"]
        overall_total = len(df)
        if overall_total == 0:
            return pd.DataFrame(
                columns=[
                    "attribute", "group", "role_type",
                    "role_share", "population_share", "representation_index",
                ]
            )

        parts: list[pd.DataFrame] = []

        for attr in attribute_cols:
            if attr not in df.columns:
                continue

            # Population share per group
            pop_counts = df[attr].value_counts()

            # Counts per (group, role_type)
            role_counts = (
                df.groupby([attr, "role_type"])
                .size()
                .reset_index(name="group_role_count")
            )

            # Total per role_type
            role_totals = (
                df.groupby("role_type")
                .size()
                .reset_index(name="role_total")
            )

            merged = role_counts.merge(role_totals, on="role_type")
            merged["role_share"] = merged["group_role_count"] / merged["role_total"]
            merged["population_share"] = merged[attr].map(pop_counts) / overall_total
            merged["representation_index"] = (
                merged["role_share"] / merged["population_share"]
            )
            merged["attribute"] = attr
            merged.rename(columns={attr: "group"}, inplace=True)

            parts.append(
                merged[
                    [
                        "attribute", "group", "role_type",
                        "role_share", "population_share", "representation_index",
                    ]
                ]
            )

        if not parts:
            return pd.DataFrame(
                columns=[
                    "attribute", "group", "role_type",
                    "role_share", "population_share", "representation_index",
                ]
            )

        result = pd.concat(parts, ignore_index=True)
        result.sort_values(["attribute", "group", "role_type"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    # ------------------------------------------------------------------
    # Statistical significance testing (Requirements 10.1–10.4, 13.4)
    # ------------------------------------------------------------------

    def statistical_significance_tests(
        self,
        df: pd.DataFrame,
        attribute_col: str,
        role_col: str,
        alpha: float = 0.05,
    ) -> dict:
        """Run chi-square (or Fisher's exact) test of independence.

        Tests whether the distribution of *role_col* values is independent
        of *attribute_col* values.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns *attribute_col* and *role_col*.
        attribute_col : str
            Column with character attribute (e.g. ``"inferred_gender"``).
        role_col : str
            Column with role assignment (e.g. ``"role_type"``).
        alpha : float, optional
            Significance level, by default ``0.05``.

        Returns
        -------
        dict
            Keys: ``chi2_statistic``, ``p_value``, ``degrees_of_freedom``,
            ``effect_size`` (Cramér's V), ``significant`` (bool),
            ``test_used`` (``"chi_square"`` | ``"fisher_exact"`` |
            ``"chi_square_low_expected"``).

        Notes
        -----
        * Falls back to Fisher's exact test when the contingency table is
          2×2 and any expected cell frequency is < 5 (Req 10.4, 13.4).
        * For tables larger than 2×2 with low expected frequencies,
          chi-square is used anyway but ``test_used`` is set to
          ``"chi_square_low_expected"`` to flag the issue.
        * Cramér's V = sqrt(chi2 / (n * (min(r, c) - 1))).

        Validates: Requirements 10.1, 10.2, 10.3, 10.4, 13.4
        """
        # Build contingency table, dropping NaN
        contingency = pd.crosstab(df[attribute_col], df[role_col])
        n = contingency.values.sum()
        r, c = contingency.shape

        # Compute expected frequencies via chi2_contingency
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        low_expected = (expected < 5).any()

        # Decide which test to report
        if low_expected and r == 2 and c == 2:
            # Fisher's exact test for 2×2 tables with low expected freq
            odds_ratio, p_value = fisher_exact(contingency.values)
            test_used = "fisher_exact"
            # For Fisher's exact, dof is not applicable; keep chi2 for
            # Cramér's V calculation (still meaningful as effect size).
        elif low_expected:
            # Larger table with low expected — use chi-square but flag it
            test_used = "chi_square_low_expected"
        else:
            test_used = "chi_square"

        # Cramér's V
        min_dim = min(r, c)
        if min_dim <= 1 or n == 0:
            effect_size = 0.0
        else:
            effect_size = float(np.sqrt(chi2 / (n * (min_dim - 1))))
            # Clamp to [0, 1] for numerical safety
            effect_size = max(0.0, min(1.0, effect_size))

        return {
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "effect_size": effect_size,
            "significant": bool(p_value < alpha),
            "test_used": test_used,
        }
