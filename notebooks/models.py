"""Core data models for the Indian Movies IMDB Research project.

Defines dataclasses for movie records, character records, and analysis results,
plus type aliases for commonly used DataFrame types.

Requirements: 3.3 (Master_DataFrame required columns), 8.4 (classification confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MasterDataFrame = pd.DataFrame
"""A DataFrame representing the merged master dataset of Indian movies."""

CharacterDataFrame = pd.DataFrame
"""A DataFrame representing character records with inferred attributes."""


# ---------------------------------------------------------------------------
# Data-model dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MovieRecord:
    """Unified record for a single Indian movie across all data sources.

    Validation rules (from design doc):
    - year must be in [1975, 2025]
    - genres must be a non-empty list
    - rating must be in [0.0, 10.0] or NaN
    - tconst must match pattern ``tt\\d+`` for IMDB-sourced records
    """

    tconst: str
    title: str
    original_title: str
    year: int
    period: str
    runtime_minutes: int
    genres: list[str]
    rating: float
    num_votes: int
    plot_summary: str
    source: str


@dataclass
class CharacterRecord:
    """Record for a single character appearance in a movie.

    Validation rules (from design doc):
    - character_name must be a non-empty string
    - category must be one of: actor, actress, director, writer, self, other
    - inferred_gender must be one of: male, female, neutral, unknown
    - classification_confidence must be in [0.0, 1.0]
    """

    tconst: str
    nconst: str
    actor_name: str
    character_name: str
    category: str
    role_type: str
    inferred_gender: str
    inferred_religion: str
    inferred_region: str
    classification_confidence: float
    year: int
    period: str


@dataclass
class AnalysisResult:
    """Container for a single analysis output (genre trend, bias test, etc.).

    Carries the result DataFrame, associated statistics, and a plain-language
    conclusion suitable for inclusion in the notebook's final summary.
    """

    analysis_type: str
    description: str
    data: pd.DataFrame
    statistics: dict
    visualization_type: str
    conclusion: str
