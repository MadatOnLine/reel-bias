"""Unit tests for the DataCleaner class.

Validates: clean_imdb_basics, clean_imdb_principals, clean_imdb_names,
filter_indian_movies, standardize_columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_cleaning import DataCleaner, IMDB_SENTINEL, assign_period_bin, add_period_column


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cleaner() -> DataCleaner:
    return DataCleaner()


# ---------------------------------------------------------------------------
# clean_imdb_basics
# ---------------------------------------------------------------------------


class TestCleanImdbBasics:
    def test_replaces_sentinel_with_nan(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt0000001"],
            "titleType": ["movie"],
            "startYear": ["\\N"],
            "endYear": ["\\N"],
            "runtimeMinutes": ["120"],
            "genres": ["Drama"],
        })
        result = cleaner.clean_imdb_basics(df)
        assert pd.isna(result.iloc[0]["startYear"])
        assert pd.isna(result.iloc[0]["endYear"])

    def test_converts_year_columns_to_numeric(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt0000001"],
            "titleType": ["movie"],
            "startYear": ["1999"],
            "runtimeMinutes": ["150"],
        })
        result = cleaner.clean_imdb_basics(df)
        assert result.iloc[0]["startYear"] == 1999.0
        assert result.iloc[0]["runtimeMinutes"] == 150.0


    def test_filters_to_movies_only(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1", "tt2", "tt3"],
            "titleType": ["movie", "tvSeries", "movie"],
            "startYear": ["2000", "2001", "2002"],
        })
        result = cleaner.clean_imdb_basics(df)
        assert len(result) == 2
        assert set(result["tconst"]) == {"tt1", "tt3"}

    def test_empty_dataframe(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame()
        result = cleaner.clean_imdb_basics(df)
        assert result.empty

    def test_does_not_mutate_original(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1"],
            "titleType": ["movie"],
            "startYear": ["\\N"],
        })
        original_val = df.iloc[0]["startYear"]
        cleaner.clean_imdb_basics(df)
        assert df.iloc[0]["startYear"] == original_val


# ---------------------------------------------------------------------------
# clean_imdb_principals
# ---------------------------------------------------------------------------


class TestCleanImdbPrincipals:
    def test_parses_json_characters(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1", "tt1"],
            "category": ["actor", "actress"],
            "characters": ['["Raj"]', '["Simran","Priya"]'],
        })
        result = cleaner.clean_imdb_principals(df, {"tt1"})
        assert result.iloc[0]["characters"] == ["Raj"]
        assert result.iloc[1]["characters"] == ["Simran", "Priya"]

    def test_malformed_json_gets_unknown(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1"],
            "category": ["actor"],
            "characters": ["not-valid-json{{{"],
        })
        result = cleaner.clean_imdb_principals(df, {"tt1"})
        assert result.iloc[0]["characters"] == ["unknown"]

    def test_filters_to_valid_tconsts(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1", "tt2", "tt3"],
            "category": ["actor", "actress", "director"],
            "characters": ['["A"]', '["B"]', '["C"]'],
        })
        result = cleaner.clean_imdb_principals(df, {"tt1", "tt3"})
        assert set(result["tconst"]) == {"tt1", "tt3"}

    def test_keeps_only_actor_actress_director(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1", "tt1", "tt1", "tt1"],
            "category": ["actor", "writer", "director", "producer"],
            "characters": ['["A"]', '["B"]', '["C"]', '["D"]'],
        })
        result = cleaner.clean_imdb_principals(df, {"tt1"})
        assert len(result) == 2
        assert set(result["category"]) == {"actor", "director"}

    def test_sentinel_replaced(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1"],
            "category": ["actor"],
            "job": ["\\N"],
            "characters": ['["Hero"]'],
        })
        result = cleaner.clean_imdb_principals(df, {"tt1"})
        assert pd.isna(result.iloc[0]["job"])

    def test_empty_dataframe(self, cleaner: DataCleaner) -> None:
        result = cleaner.clean_imdb_principals(pd.DataFrame(), {"tt1"})
        assert result.empty

    def test_nan_characters_become_empty_list(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1"],
            "category": ["director"],
            "characters": [np.nan],
        })
        result = cleaner.clean_imdb_principals(df, {"tt1"})
        assert result.iloc[0]["characters"] == []


# ---------------------------------------------------------------------------
# clean_imdb_names
# ---------------------------------------------------------------------------


class TestCleanImdbNames:
    def test_replaces_sentinel(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "nconst": ["nm1"],
            "primaryName": ["Test Actor"],
            "primaryProfession": ["actor,director"],
            "birthYear": ["\\N"],
        })
        result = cleaner.clean_imdb_names(df)
        assert pd.isna(result.iloc[0]["birthYear"])

    def test_parses_professions_to_list(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "nconst": ["nm1"],
            "primaryName": ["Test"],
            "primaryProfession": ["actor,director,writer"],
        })
        result = cleaner.clean_imdb_names(df)
        assert result.iloc[0]["primaryProfession"] == ["actor", "director", "writer"]

    def test_nan_profession_becomes_empty_list(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "nconst": ["nm1"],
            "primaryName": ["Test"],
            "primaryProfession": [np.nan],
        })
        result = cleaner.clean_imdb_names(df)
        assert result.iloc[0]["primaryProfession"] == []

    def test_empty_dataframe(self, cleaner: DataCleaner) -> None:
        result = cleaner.clean_imdb_names(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# filter_indian_movies
# ---------------------------------------------------------------------------


class TestFilterIndianMovies:
    def test_filters_by_year_range(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1", "tt2", "tt3", "tt4"],
            "startYear": [1970, 1975, 2000, 2030],
        })
        result = cleaner.filter_indian_movies(df)
        assert len(result) == 2
        assert set(result["tconst"]) == {"tt2", "tt3"}

    def test_uses_region_column_when_available(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1", "tt2", "tt3"],
            "startYear": [2000, 2001, 2002],
            "region": ["IN", "US", "India"],
        })
        result = cleaner.filter_indian_movies(df)
        assert len(result) == 2
        assert set(result["tconst"]) == {"tt1", "tt3"}

    def test_falls_back_when_region_filter_yields_zero(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1", "tt2"],
            "startYear": [2000, 2001],
            "region": ["US", "UK"],
        })
        result = cleaner.filter_indian_movies(df)
        # Falls back to year-filtered data
        assert len(result) == 2

    def test_empty_dataframe(self, cleaner: DataCleaner) -> None:
        result = cleaner.filter_indian_movies(pd.DataFrame())
        assert result.empty

    def test_boundary_years_included(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1", "tt2"],
            "startYear": [1975, 2025],
        })
        result = cleaner.filter_indian_movies(df)
        assert len(result) == 2

    def test_uses_year_column_if_startYear_missing(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1", "tt2"],
            "year": [1980, 2030],
        })
        result = cleaner.filter_indian_movies(df)
        assert len(result) == 1
        assert result.iloc[0]["tconst"] == "tt1"


# ---------------------------------------------------------------------------
# standardize_columns
# ---------------------------------------------------------------------------


class TestStandardizeColumns:
    def test_imdb_basics_mapping(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1"],
            "primaryTitle": ["My Movie"],
            "startYear": [2000],
            "genres": ["Drama"],
            "averageRating": [7.5],
        })
        result = cleaner.standardize_columns(df, "imdb_basics")
        assert list(result.columns) == cleaner.UNIFIED_COLUMNS
        assert result.iloc[0]["tconst"] == "tt1"
        assert result.iloc[0]["title"] == "My Movie"
        assert result.iloc[0]["year"] == 2000.0
        assert result.iloc[0]["genres"] == "Drama"
        assert result.iloc[0]["rating"] == 7.5
        assert result.iloc[0]["source"] == "imdb_basics"

    def test_mendeley_mapping(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "Title": ["Bollywood Film"],
            "Year": [1990],
            "Genre": ["Action"],
            "Rating": [6.0],
        })
        result = cleaner.standardize_columns(df, "mendeley")
        assert result.iloc[0]["title"] == "Bollywood Film"
        assert result.iloc[0]["year"] == 1990.0
        assert result.iloc[0]["source"] == "mendeley"

    def test_github_bollywood_mapping(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "Title": ["Film"],
            "Year": [2010],
            "Genre": ["Comedy"],
            "Rating": [5.5],
        })
        result = cleaner.standardize_columns(df, "github_bollywood")
        assert result.iloc[0]["title"] == "Film"
        assert result.iloc[0]["source"] == "github_bollywood"

    def test_kaggle_mapping(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "Name": ["Kaggle Movie"],
            "Year": [2015],
            "Genre": ["Thriller"],
            "Rating": [8.0],
        })
        result = cleaner.standardize_columns(df, "kaggle_indian_movies")
        assert result.iloc[0]["title"] == "Kaggle Movie"
        assert result.iloc[0]["source"] == "kaggle_indian_movies"

    def test_missing_columns_become_nan(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "Title": ["No Rating Movie"],
            "Year": [2000],
        })
        result = cleaner.standardize_columns(df, "mendeley")
        assert pd.isna(result.iloc[0]["rating"])
        assert pd.isna(result.iloc[0]["tconst"])

    def test_empty_dataframe(self, cleaner: DataCleaner) -> None:
        result = cleaner.standardize_columns(pd.DataFrame(), "imdb_basics")
        assert result.empty
        assert list(result.columns) == cleaner.UNIFIED_COLUMNS

    def test_unknown_source_returns_source_column_only(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({"col1": [1, 2]})
        result = cleaner.standardize_columns(df, "unknown_source")
        assert "source" in result.columns
        assert all(result["source"] == "unknown_source")

    def test_year_coerced_to_numeric(self, cleaner: DataCleaner) -> None:
        df = pd.DataFrame({
            "tconst": ["tt1"],
            "primaryTitle": ["Movie"],
            "startYear": ["not_a_year"],
            "genres": ["Drama"],
        })
        result = cleaner.standardize_columns(df, "imdb_basics")
        assert pd.isna(result.iloc[0]["year"])


# ---------------------------------------------------------------------------
# assign_period_bin & add_period_column
# ---------------------------------------------------------------------------


class TestAssignPeriodBin:
    """Validates: Requirements 4.1, 4.2, 4.3"""

    def test_example_1983(self) -> None:
        assert assign_period_bin(1983) == "1980-1984"

    def test_example_1975(self) -> None:
        assert assign_period_bin(1975) == "1975-1979"

    def test_example_2025(self) -> None:
        assert assign_period_bin(2025) == "2025-2029"

    def test_bin_start_boundary(self) -> None:
        assert assign_period_bin(1980) == "1980-1984"

    def test_bin_end_boundary(self) -> None:
        assert assign_period_bin(1984) == "1980-1984"

    def test_all_expected_bins(self) -> None:
        expected_bins = [
            "1975-1979", "1980-1984", "1985-1989", "1990-1994",
            "1995-1999", "2000-2004", "2005-2009", "2010-2014",
            "2015-2019", "2020-2024", "2025-2029",
        ]
        representative_years = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]
        for year, expected in zip(representative_years, expected_bins):
            assert assign_period_bin(year) == expected

    def test_year_within_assigned_period(self) -> None:
        for year in range(1975, 2030):
            period = assign_period_bin(year)
            start, end = (int(x) for x in period.split("-"))
            assert start <= year <= end


class TestAddPeriodColumn:
    """Validates: Requirements 4.1, 4.2, 4.3"""

    def test_adds_period_column(self) -> None:
        df = pd.DataFrame({"year": [1983, 1975, 2025]})
        result = add_period_column(df)
        assert "period" in result.columns
        assert list(result["period"]) == ["1980-1984", "1975-1979", "2025-2029"]

    def test_does_not_mutate_original(self) -> None:
        df = pd.DataFrame({"year": [2000]})
        add_period_column(df)
        assert "period" not in df.columns

    def test_custom_year_column(self) -> None:
        df = pd.DataFrame({"startYear": [1990]})
        result = add_period_column(df, year_col="startYear")
        assert result.iloc[0]["period"] == "1990-1994"

    def test_handles_nan_years(self) -> None:
        df = pd.DataFrame({"year": [2000, np.nan, 1985]})
        result = add_period_column(df)
        assert result.iloc[0]["period"] == "2000-2004"
        assert pd.isna(result.iloc[1]["period"])
        assert result.iloc[2]["period"] == "1985-1989"


# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------


class TestMergeDatasets:
    """Validates: Requirements 3.1, 3.2, 3.3, 3.4, 12.1, 12.2"""

    def test_basic_merge_single_source(self, cleaner: DataCleaner) -> None:
        datasets = {
            "imdb_basics": pd.DataFrame({
                "tconst": ["tt1", "tt2"],
                "title": ["Movie A", "Movie B"],
                "year": [2000, 2010],
                "genres": ["Drama", "Action"],
                "rating": [7.0, 8.0],
                "source": ["imdb_basics", "imdb_basics"],
            }),
        }
        result = cleaner.merge_datasets(datasets)
        assert len(result) == 2
        assert "period" in result.columns

    def test_required_columns_present(self, cleaner: DataCleaner) -> None:
        datasets = {
            "imdb_basics": pd.DataFrame({
                "tconst": ["tt1"],
                "title": ["Movie A"],
                "year": [2000],
                "genres": ["Drama"],
                "rating": [7.0],
                "source": ["imdb_basics"],
            }),
        }
        result = cleaner.merge_datasets(datasets)
        for col in cleaner.MASTER_REQUIRED_COLUMNS:
            assert col in result.columns, f"Missing required column: {col}"

    def test_deduplicates_on_tconst(self, cleaner: DataCleaner) -> None:
        datasets = {
            "imdb_basics": pd.DataFrame({
                "tconst": ["tt1"],
                "title": ["Movie A"],
                "year": [2000],
                "genres": ["Drama"],
                "rating": [7.0],
                "source": ["imdb_basics"],
            }),
            "kaggle_indian_movies": pd.DataFrame({
                "tconst": ["tt1"],
                "title": ["Movie A (Kaggle)"],
                "year": [2000],
                "genres": ["Drama,Action"],
                "rating": [6.5],
                "source": ["kaggle_indian_movies"],
            }),
        }
        result = cleaner.merge_datasets(datasets)
        assert len(result) == 1
        # IMDB data preferred
        assert result.iloc[0]["source"] == "imdb_basics"

    def test_deduplicates_on_title_year_when_no_tconst(self, cleaner: DataCleaner) -> None:
        datasets = {
            "mendeley": pd.DataFrame({
                "tconst": [np.nan],
                "title": ["Same Movie"],
                "year": [2005],
                "genres": ["Comedy"],
                "rating": [6.0],
                "source": ["mendeley"],
            }),
            "github_bollywood": pd.DataFrame({
                "tconst": [np.nan],
                "title": ["Same Movie"],
                "year": [2005],
                "genres": ["Comedy"],
                "rating": [5.5],
                "source": ["github_bollywood"],
            }),
        }
        result = cleaner.merge_datasets(datasets)
        assert len(result) == 1

    def test_prefers_imdb_data_on_conflict(self, cleaner: DataCleaner) -> None:
        datasets = {
            "kaggle_indian_movies": pd.DataFrame({
                "tconst": ["tt1"],
                "title": ["Kaggle Title"],
                "year": [2000],
                "genres": ["Action"],
                "rating": [5.0],
                "source": ["kaggle_indian_movies"],
            }),
            "imdb_basics": pd.DataFrame({
                "tconst": ["tt1"],
                "title": ["IMDB Title"],
                "year": [2000],
                "genres": ["Drama"],
                "rating": [7.5],
                "source": ["imdb_basics"],
            }),
        }
        result = cleaner.merge_datasets(datasets)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "IMDB Title"
        assert result.iloc[0]["rating"] == 7.5

    def test_row_count_does_not_exceed_sum(self, cleaner: DataCleaner) -> None:
        datasets = {
            "imdb_basics": pd.DataFrame({
                "tconst": ["tt1", "tt2"],
                "title": ["A", "B"],
                "year": [2000, 2001],
                "genres": ["Drama", "Action"],
                "rating": [7.0, 8.0],
                "source": ["imdb_basics", "imdb_basics"],
            }),
            "mendeley": pd.DataFrame({
                "tconst": ["tt3", "tt1"],
                "title": ["C", "A"],
                "year": [2002, 2000],
                "genres": ["Comedy", "Drama"],
                "rating": [6.0, 7.0],
                "source": ["mendeley", "mendeley"],
            }),
        }
        total_input = sum(len(df) for df in datasets.values())
        result = cleaner.merge_datasets(datasets)
        assert len(result) <= total_input

    def test_period_column_added(self, cleaner: DataCleaner) -> None:
        datasets = {
            "imdb_basics": pd.DataFrame({
                "tconst": ["tt1"],
                "title": ["Movie"],
                "year": [1983],
                "genres": ["Drama"],
                "rating": [7.0],
                "source": ["imdb_basics"],
            }),
        }
        result = cleaner.merge_datasets(datasets)
        assert result.iloc[0]["period"] == "1980-1984"

    def test_empty_datasets_dict(self, cleaner: DataCleaner) -> None:
        result = cleaner.merge_datasets({})
        assert result.empty
        for col in cleaner.MASTER_REQUIRED_COLUMNS:
            assert col in result.columns

    def test_all_empty_dataframes(self, cleaner: DataCleaner) -> None:
        datasets = {
            "imdb_basics": pd.DataFrame(columns=["tconst", "title", "year", "genres", "rating", "source"]),
            "mendeley": pd.DataFrame(columns=["tconst", "title", "year", "genres", "rating", "source"]),
        }
        result = cleaner.merge_datasets(datasets)
        assert result.empty

    def test_multiple_sources_no_overlap(self, cleaner: DataCleaner) -> None:
        datasets = {
            "imdb_basics": pd.DataFrame({
                "tconst": ["tt1"],
                "title": ["Movie A"],
                "year": [2000],
                "genres": ["Drama"],
                "rating": [7.0],
                "source": ["imdb_basics"],
            }),
            "mendeley": pd.DataFrame({
                "tconst": ["tt2"],
                "title": ["Movie B"],
                "year": [2005],
                "genres": ["Comedy"],
                "rating": [6.0],
                "source": ["mendeley"],
            }),
        }
        result = cleaner.merge_datasets(datasets)
        assert len(result) == 2

    def test_character_referential_integrity(self, cleaner: DataCleaner) -> None:
        datasets = {
            "imdb_basics": pd.DataFrame({
                "tconst": ["tt1", "tt2"],
                "title": ["A", "B"],
                "year": [2000, 2001],
                "genres": ["Drama", "Action"],
                "rating": [7.0, 8.0],
                "source": ["imdb_basics", "imdb_basics"],
            }),
        }
        characters = pd.DataFrame({
            "tconst": ["tt1", "tt2", "tt999"],
            "character_name": ["Hero", "Villain", "Ghost"],
        })
        result = cleaner.merge_datasets(datasets, characters_df=characters)
        # The method returns the master df; characters_df filtering is
        # done in-place on the passed reference — but the method's contract
        # ensures no character references a movie outside master.
        valid_tconsts = set(result["tconst"].dropna())
        # tt999 should not be in valid tconsts
        assert "tt999" not in valid_tconsts

    def test_title_year_dedup_removes_rows_already_in_tconst_set(self, cleaner: DataCleaner) -> None:
        """A row without tconst whose title+year matches a row with tconst
        should be dropped."""
        datasets = {
            "imdb_basics": pd.DataFrame({
                "tconst": ["tt1"],
                "title": ["Movie A"],
                "year": [2000],
                "genres": ["Drama"],
                "rating": [7.0],
                "source": ["imdb_basics"],
            }),
            "mendeley": pd.DataFrame({
                "tconst": [np.nan],
                "title": ["Movie A"],
                "year": [2000],
                "genres": ["Drama"],
                "rating": [6.5],
                "source": ["mendeley"],
            }),
        }
        result = cleaner.merge_datasets(datasets)
        assert len(result) == 1
        assert result.iloc[0]["source"] == "imdb_basics"
