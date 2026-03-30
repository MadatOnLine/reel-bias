"""Thematic Analysis module for the Indian Movies IMDB Research project.

Analyses how movie themes (genres, topics, keywords) have changed over
5-year periods.

Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 7.1, 7.2, 7.3, 13.5
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ThematicAnalyzer:
    """Analyses thematic content evolution in Indian cinema."""

    # ------------------------------------------------------------------
    # 6.1  Genre trends by period
    # ------------------------------------------------------------------

    def genre_trends_by_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute genre frequency and percentage per 5-year period.

        * Explodes multi-genre entries so each genre is counted individually.
        * If ``genres`` is a string it is split on comma.
        * Returns a DataFrame with columns: ``period``, ``genre``, ``count``,
          ``percentage``.
        * Percentages within each period sum to 100.0 (tolerance 0.01).
        * Sorted by period ascending, then count descending within each period.

        Validates: Requirements 5.1, 5.2, 5.3, 5.4
        """
        if df.empty:
            return pd.DataFrame(columns=["period", "genre", "count", "percentage"])

        work = df[["genres", "period"]].copy()

        # Normalise genres to list[str]
        def _to_list(val):
            if isinstance(val, list):
                return [g.strip() for g in val if g and str(g).strip()]
            if isinstance(val, str):
                return [g.strip() for g in val.split(",") if g.strip()]
            return []

        work["genres"] = work["genres"].apply(_to_list)

        # Drop rows with empty genre lists or missing period
        work = work[work["genres"].map(len) > 0]
        work = work.dropna(subset=["period"])

        # Explode so each genre gets its own row
        exploded = work.explode("genres").rename(columns={"genres": "genre"})

        # Count per (period, genre)
        counts = (
            exploded.groupby(["period", "genre"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

        # Percentage within each period
        period_totals = counts.groupby("period")["count"].transform("sum")
        counts["percentage"] = (counts["count"] / period_totals) * 100.0

        # Sort: period ascending, count descending within each period
        counts = counts.sort_values(
            ["period", "count"], ascending=[True, False]
        ).reset_index(drop=True)

        return counts

    # ------------------------------------------------------------------
    # 6.2  Genre co-occurrence matrix
    # ------------------------------------------------------------------

    def genre_cooccurrence_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a symmetric genre co-occurrence matrix.

        Counts how often each pair of genres appears on the same movie.
        ``matrix[A][B] == matrix[B][A]``.

        Validates: Requirements 6.1, 6.2
        """
        if df.empty:
            return pd.DataFrame()

        # Normalise genres to list[str]
        def _to_list(val):
            if isinstance(val, list):
                return [g.strip() for g in val if g and str(g).strip()]
            if isinstance(val, str):
                return [g.strip() for g in val.split(",") if g.strip()]
            return []

        genre_lists = df["genres"].apply(_to_list)

        # Collect all unique genres
        all_genres = sorted({g for gl in genre_lists for g in gl})

        if not all_genres:
            return pd.DataFrame()

        matrix = pd.DataFrame(
            0, index=all_genres, columns=all_genres, dtype=int
        )

        for genres in genre_lists:
            unique = sorted(set(genres))
            for a, b in combinations(unique, 2):
                matrix.loc[a, b] += 1
                matrix.loc[b, a] += 1
            # Also count self-co-occurrence (diagonal)
            for g in unique:
                matrix.loc[g, g] += 1

        return matrix

    # ------------------------------------------------------------------
    # 6.3  Topic modelling (LDA)
    # ------------------------------------------------------------------

    def topic_model_plots(
        self, df: pd.DataFrame, n_topics: int = 10
    ) -> dict:
        """Run LDA topic modelling on plot summaries / descriptions.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least one text column among:
            ``plot_summary``, ``plot``, ``description``, ``overview``.
        n_topics : int
            Number of LDA topics (default 10).

        Returns
        -------
        dict
            Keys: ``topic_words``, ``document_topics``, ``coherence_score``.
            If the majority of movies lack plot text the dict also contains
            ``fallback`` = ``"genre_only"`` and topic modelling is skipped.

        Validates: Requirements 7.1, 7.2, 7.3, 13.5
        """
        # Identify the text column
        text_col = None
        for candidate in ("plot_summary", "plot", "description", "overview"):
            if candidate in df.columns:
                text_col = candidate
                break

        # Gather text, treating NaN / empty as missing
        if text_col is not None:
            texts = df[text_col].fillna("").astype(str).str.strip()
            has_text = texts.str.len() > 0
        else:
            has_text = pd.Series([False] * len(df), dtype=bool)

        # Fall back if majority lack plot text
        if has_text.sum() <= len(df) / 2:
            logger.info(
                "topic_model_plots: majority of movies lack plot text "
                "(%d / %d) — falling back to genre-only analysis",
                has_text.sum(),
                len(df),
            )
            return {
                "topic_words": {},
                "document_topics": pd.DataFrame(),
                "coherence_score": 0.0,
                "fallback": "genre_only",
            }

        # Filter to documents with text
        docs = texts[has_text].tolist()

        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        # Adapt min_df for small corpora
        adaptive_min_df = min(2, len(docs))
        vectorizer = CountVectorizer(
            max_df=0.95, min_df=adaptive_min_df, stop_words="english"
        )
        try:
            dtm = vectorizer.fit_transform(docs)
        except ValueError:
            # Not enough documents after filtering
            return {
                "topic_words": {},
                "document_topics": pd.DataFrame(),
                "coherence_score": 0.0,
                "fallback": "genre_only",
            }

        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42
        )
        doc_topic_matrix = lda.fit_transform(dtm)

        feature_names = vectorizer.get_feature_names_out()

        # Extract top words per topic
        topic_words: dict[int, list[str]] = {}
        for idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            topic_words[idx] = [feature_names[i] for i in top_indices]

        # Document-topic DataFrame
        document_topics = pd.DataFrame(
            doc_topic_matrix,
            columns=[f"topic_{i}" for i in range(n_topics)],
        )

        # Simple coherence proxy: mean log-likelihood
        coherence_score = float(lda.score(dtm)) / dtm.shape[0]

        return {
            "topic_words": topic_words,
            "document_topics": document_topics,
            "coherence_score": coherence_score,
        }

    # ------------------------------------------------------------------
    # 6.4  Keyword / runtime / rating trends
    # ------------------------------------------------------------------

    def keyword_trends(
        self, df: pd.DataFrame, top_n: int = 20
    ) -> pd.DataFrame:
        """Extract top keywords from movie titles per 5-year period.

        Uses ``TfidfVectorizer`` to rank terms within each period.

        Returns DataFrame with columns: ``period``, ``keyword``, ``tfidf_score``.

        Validates: Requirements 5.1, 5.2
        """
        if df.empty or "title" not in df.columns or "period" not in df.columns:
            return pd.DataFrame(columns=["period", "keyword", "tfidf_score"])

        from sklearn.feature_extraction.text import TfidfVectorizer

        work = df[["title", "period"]].dropna().copy()
        work["title"] = work["title"].astype(str).str.strip()
        work = work[work["title"].str.len() > 0]

        if work.empty:
            return pd.DataFrame(columns=["period", "keyword", "tfidf_score"])

        rows: list[dict] = []
        for period, group in work.groupby("period"):
            titles = group["title"].tolist()
            if not titles:
                continue
            vec = TfidfVectorizer(stop_words="english", max_features=top_n)
            try:
                tfidf_matrix = vec.fit_transform(titles)
            except ValueError:
                continue
            feature_names = vec.get_feature_names_out()
            scores = tfidf_matrix.mean(axis=0).A1
            for fname, score in zip(feature_names, scores):
                rows.append(
                    {"period": period, "keyword": fname, "tfidf_score": float(score)}
                )

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(
                ["period", "tfidf_score"], ascending=[True, False]
            ).reset_index(drop=True)
        return result

    def runtime_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runtime distribution stats per 5-year period.

        Returns DataFrame with columns: ``period``, ``mean``, ``median``, ``std``.

        Validates: Requirements 5.1, 5.2
        """
        if df.empty:
            return pd.DataFrame(columns=["period", "mean", "median", "std"])

        # Identify runtime column
        rt_col = None
        for candidate in ("runtimeMinutes", "runtime_minutes", "runtime", "duration"):
            if candidate in df.columns:
                rt_col = candidate
                break

        if rt_col is None or "period" not in df.columns:
            return pd.DataFrame(columns=["period", "mean", "median", "std"])

        work = df[["period", rt_col]].copy()
        work[rt_col] = pd.to_numeric(work[rt_col], errors="coerce")
        work = work.dropna(subset=["period", rt_col])

        if work.empty:
            return pd.DataFrame(columns=["period", "mean", "median", "std"])

        stats = (
            work.groupby("period")[rt_col]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        stats = stats.sort_values("period").reset_index(drop=True)
        return stats

    def rating_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rating distribution stats per 5-year period.

        Returns DataFrame with columns: ``period``, ``mean``, ``median``, ``std``.

        Validates: Requirements 5.1, 5.2
        """
        if df.empty:
            return pd.DataFrame(columns=["period", "mean", "median", "std"])

        rt_col = None
        for candidate in ("rating", "averageRating", "imdb_rating"):
            if candidate in df.columns:
                rt_col = candidate
                break

        if rt_col is None or "period" not in df.columns:
            return pd.DataFrame(columns=["period", "mean", "median", "std"])

        work = df[["period", rt_col]].copy()
        work[rt_col] = pd.to_numeric(work[rt_col], errors="coerce")
        work = work.dropna(subset=["period", rt_col])

        if work.empty:
            return pd.DataFrame(columns=["period", "mean", "median", "std"])

        stats = (
            work.groupby("period")[rt_col]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        stats = stats.sort_values("period").reset_index(drop=True)
        return stats
