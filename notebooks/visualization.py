"""Visualization module for the Indian Movies IMDB Research project.

Produces publication-quality charts, plots, and visual outputs for genre
trends, bias analysis, topic modeling, and summary dashboards.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent styling defaults
_STYLE_DEFAULTS = {
    "figure.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}


def _apply_style() -> None:
    """Apply consistent matplotlib/seaborn styling."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    matplotlib.rcParams.update(_STYLE_DEFAULTS)


class Visualizer:
    """Produces all charts, plots, and visual outputs for the notebook.

    Every public method:
    - Returns the ``matplotlib.figure.Figure`` object for flexibility.
    - Calls ``plt.close()`` on the figure after creation to avoid memory leaks.
    - Uses 300 DPI for export-ready quality.
    - Applies consistent matplotlib/seaborn styling.
    """

    def __init__(self) -> None:
        _apply_style()

    # ------------------------------------------------------------------
    # 10.1  Genre / thematic visualizations
    # ------------------------------------------------------------------

    def plot_genre_trends(self, genre_data: pd.DataFrame) -> matplotlib.figure.Figure:
        """Stacked area chart of genre popularity over 5-year periods.

        Parameters
        ----------
        genre_data : pd.DataFrame
            Columns: ``period``, ``genre``, ``count``, ``percentage``.

        Returns
        -------
        matplotlib.figure.Figure

        Validates: Requirements 11.1, 11.6, 11.7
        """
        _apply_style()

        if genre_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.set_title("Genre Trends Over Time (no data)")
            plt.close(fig)
            return fig

        # Pivot to get genres as columns, periods as rows
        pivot = genre_data.pivot_table(
            index="period", columns="genre", values="percentage", fill_value=0.0
        )
        pivot = pivot.sort_index()

        # Keep top genres by total percentage to avoid clutter
        top_genres = pivot.sum().nlargest(10).index.tolist()
        plot_data = pivot[top_genres]

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        plot_data.plot.area(ax=ax, alpha=0.7, linewidth=0.5)
        ax.set_title("Genre Popularity Over 5-Year Periods")
        ax.set_xlabel("Period")
        ax.set_ylabel("Percentage (%)")
        ax.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_genre_heatmap(self, cooccurrence: pd.DataFrame) -> matplotlib.figure.Figure:
        """Heatmap of genre co-occurrence matrix.

        Parameters
        ----------
        cooccurrence : pd.DataFrame
            Symmetric genre co-occurrence matrix (genres as both index and columns).

        Returns
        -------
        matplotlib.figure.Figure

        Validates: Requirements 11.2, 11.6, 11.7
        """
        _apply_style()

        if cooccurrence.empty:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.set_title("Genre Co-occurrence Heatmap (no data)")
            plt.close(fig)
            return fig

        size = max(8, len(cooccurrence) * 0.6)
        fig, ax = plt.subplots(figsize=(size, size), dpi=300)
        sns.heatmap(
            cooccurrence,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Genre Co-occurrence Heatmap")
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_topic_wordclouds(
        self, topic_words: dict[int, list[str]]
    ) -> matplotlib.figure.Figure:
        """Word clouds for each LDA topic.

        Parameters
        ----------
        topic_words : dict[int, list[str]]
            Mapping from topic index to list of top words.

        Returns
        -------
        matplotlib.figure.Figure

        Validates: Requirements 11.3, 11.6, 11.7
        """
        _apply_style()

        if not topic_words:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
            ax.set_title("Topic Word Clouds (no data)")
            ax.axis("off")
            plt.close(fig)
            return fig

        from wordcloud import WordCloud

        n_topics = len(topic_words)
        cols = min(3, n_topics)
        rows = (n_topics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=300)

        # Ensure axes is always a flat iterable
        if n_topics == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        for idx, (topic_id, words) in enumerate(sorted(topic_words.items())):
            ax = axes[idx]
            # Build frequency dict: higher-ranked words get higher weight
            freq = {w: len(words) - i for i, w in enumerate(words)}
            wc = WordCloud(
                width=400, height=300, background_color="white", colormap="viridis"
            ).generate_from_frequencies(freq)
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(f"Topic {topic_id}")
            ax.axis("off")

        # Hide unused axes
        for idx in range(n_topics, len(axes)):
            axes[idx].axis("off")

        fig.suptitle("LDA Topic Word Clouds", fontsize=14, y=1.02)
        fig.tight_layout()
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # 10.2  Bias / role visualizations
    # ------------------------------------------------------------------

    def plot_role_distribution(
        self,
        role_data: pd.DataFrame,
        group_by: str,
        significance_results: dict[str, Any] | None = None,
    ) -> matplotlib.figure.Figure:
        """Grouped bar chart of role distribution by specified attribute.

        Parameters
        ----------
        role_data : pd.DataFrame
            Crosstab DataFrame with role_type as rows and attribute values
            as columns.
        group_by : str
            Attribute name used for grouping (e.g. ``"gender"``,
            ``"religion"``, ``"region"``).
        significance_results : dict, optional
            If provided and ``significance_results["significant"]`` is True,
            a significance marker is annotated on the chart.

        Returns
        -------
        matplotlib.figure.Figure

        Validates: Requirements 11.4, 11.6, 11.7, 11.8
        """
        _apply_style()

        if role_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.set_title(f"Role Distribution by {group_by.title()} (no data)")
            plt.close(fig)
            return fig

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        role_data.plot.bar(ax=ax, width=0.8)
        ax.set_title(f"Role Distribution by {group_by.title()}")
        ax.set_xlabel("Role Type")
        ax.set_ylabel("Count")
        ax.legend(title=group_by.title(), bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)

        # Annotate with significance marker if results are significant
        if significance_results and significance_results.get("significant"):
            p_val = significance_results.get("p_value", 0)
            marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            ax.annotate(
                f"{marker} p={p_val:.4f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
            )

        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_temporal_bias(
        self,
        bias_data: pd.DataFrame,
        significance_results: dict[str, Any] | None = None,
    ) -> matplotlib.figure.Figure:
        """Line charts showing bias metric changes over 5-year periods.

        Parameters
        ----------
        bias_data : pd.DataFrame
            Columns: ``period``, ``attribute``, ``value``, ``count``,
            ``percentage``.
        significance_results : dict, optional
            If provided and ``significance_results["significant"]`` is True,
            a significance marker is annotated on the chart.

        Returns
        -------
        matplotlib.figure.Figure

        Validates: Requirements 11.5, 11.6, 11.7, 11.8
        """
        _apply_style()

        if bias_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.set_title("Temporal Bias Trends (no data)")
            plt.close(fig)
            return fig

        attributes = bias_data["attribute"].unique()
        n_attrs = len(attributes)
        fig, axes = plt.subplots(
            n_attrs, 1, figsize=(12, 5 * n_attrs), dpi=300, squeeze=False
        )

        for idx, attr in enumerate(sorted(attributes)):
            ax = axes[idx, 0]
            attr_data = bias_data[bias_data["attribute"] == attr]
            for value in sorted(attr_data["value"].unique()):
                subset = attr_data[attr_data["value"] == value].sort_values("period")
                ax.plot(subset["period"], subset["percentage"], marker="o", label=value)
            ax.set_title(f"Temporal Trends: {attr}")
            ax.set_xlabel("Period")
            ax.set_ylabel("Percentage (%)")
            ax.legend(title="Value", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=45)

            # Significance annotation
            if significance_results and significance_results.get("significant"):
                p_val = significance_results.get("p_value", 0)
                marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                ax.annotate(
                    f"{marker} p={p_val:.4f}",
                    xy=(0.95, 0.95),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                    color="red",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5
                    ),
                )

        fig.suptitle("Bias Metric Changes Over Time", fontsize=14)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_name_frequency(
        self,
        name_data: pd.DataFrame,
        significance_results: dict[str, Any] | None = None,
    ) -> matplotlib.figure.Figure:
        """Horizontal bar charts of most common character names.

        Parameters
        ----------
        name_data : pd.DataFrame
            Must contain columns ``name`` and ``count``. May also contain
            ``period``, ``inferred_gender``, ``role_type`` for faceting.
        significance_results : dict, optional
            If provided and ``significance_results["significant"]`` is True,
            a significance marker is annotated on the chart.

        Returns
        -------
        matplotlib.figure.Figure

        Validates: Requirements 11.4, 11.6, 11.7, 11.8
        """
        _apply_style()

        if name_data.empty:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.set_title("Most Common Character Names (no data)")
            plt.close(fig)
            return fig

        # Aggregate across all groups to get overall top names
        top = (
            name_data.groupby("name")["count"]
            .sum()
            .nlargest(20)
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        top.plot.barh(ax=ax, color=sns.color_palette("colorblind", len(top)))
        ax.set_title("Most Common Character Names")
        ax.set_xlabel("Total Count")
        ax.set_ylabel("Character Name")

        # Significance annotation
        if significance_results and significance_results.get("significant"):
            p_val = significance_results.get("p_value", 0)
            marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            ax.annotate(
                f"{marker} p={p_val:.4f}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
            )

        fig.tight_layout()
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # 10.3  Summary dashboard
    # ------------------------------------------------------------------

    def plot_summary_dashboard(
        self, all_results: dict[str, Any]
    ) -> matplotlib.figure.Figure:
        """Multi-panel summary figure with key findings.

        Parameters
        ----------
        all_results : dict
            Expected keys (all optional):
            - ``genre_trends`` : pd.DataFrame with period/genre/count/percentage
            - ``bias_results`` : dict with chi2_statistic, p_value, etc.
            - ``temporal_bias`` : pd.DataFrame with period/attribute/value/count/percentage

        Returns
        -------
        matplotlib.figure.Figure

        Validates: Requirements 11.6, 11.7
        """
        _apply_style()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

        # --- Panel 1: Genre trends (top-left) ---
        ax1 = axes[0, 0]
        genre_trends = all_results.get("genre_trends")
        if genre_trends is not None and not genre_trends.empty:
            pivot = genre_trends.pivot_table(
                index="period", columns="genre", values="percentage", fill_value=0.0
            ).sort_index()
            top_genres = pivot.sum().nlargest(5).index.tolist()
            pivot[top_genres].plot.area(ax=ax1, alpha=0.7, linewidth=0.5)
            ax1.set_title("Top 5 Genre Trends")
            ax1.set_xlabel("Period")
            ax1.set_ylabel("Percentage (%)")
            ax1.legend(fontsize=7, loc="upper left")
            ax1.tick_params(axis="x", rotation=45)
        else:
            ax1.set_title("Genre Trends (no data)")
            ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)

        # --- Panel 2: Bias test summary (top-right) ---
        ax2 = axes[0, 1]
        bias_results = all_results.get("bias_results")
        if bias_results and isinstance(bias_results, dict):
            labels = []
            values = []
            for key in ("chi2_statistic", "p_value", "effect_size"):
                if key in bias_results:
                    labels.append(key.replace("_", " ").title())
                    values.append(bias_results[key])
            if labels:
                colors = ["#4c72b0", "#dd8452", "#55a868"]
                ax2.barh(labels, values, color=colors[: len(labels)])
                ax2.set_title("Statistical Test Results")
                ax2.set_xlabel("Value")
                # Annotate significance
                if bias_results.get("significant"):
                    ax2.annotate(
                        "* Significant",
                        xy=(0.95, 0.95),
                        xycoords="axes fraction",
                        ha="right",
                        va="top",
                        fontsize=10,
                        fontweight="bold",
                        color="red",
                    )
            else:
                ax2.set_title("Bias Test Results (no data)")
                ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
        else:
            ax2.set_title("Bias Test Results (no data)")
            ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)

        # --- Panel 3: Temporal bias snapshot (bottom-left) ---
        ax3 = axes[1, 0]
        temporal_bias = all_results.get("temporal_bias")
        if temporal_bias is not None and not temporal_bias.empty:
            # Show first attribute's trends
            first_attr = sorted(temporal_bias["attribute"].unique())[0]
            attr_data = temporal_bias[temporal_bias["attribute"] == first_attr]
            for value in sorted(attr_data["value"].unique()):
                subset = attr_data[attr_data["value"] == value].sort_values("period")
                ax3.plot(subset["period"], subset["percentage"], marker="o", label=value)
            ax3.set_title(f"Temporal Trends: {first_attr}")
            ax3.set_xlabel("Period")
            ax3.set_ylabel("Percentage (%)")
            ax3.legend(fontsize=7, loc="upper left")
            ax3.tick_params(axis="x", rotation=45)
        else:
            ax3.set_title("Temporal Bias (no data)")
            ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)

        # --- Panel 4: Key findings text (bottom-right) ---
        ax4 = axes[1, 1]
        ax4.axis("off")
        findings = ["Key Findings Summary", "=" * 30]
        if bias_results and isinstance(bias_results, dict):
            sig = "Yes" if bias_results.get("significant") else "No"
            findings.append(f"Statistically significant: {sig}")
            if "p_value" in bias_results:
                findings.append(f"p-value: {bias_results['p_value']:.4f}")
            if "effect_size" in bias_results:
                findings.append(f"Effect size (Cramér's V): {bias_results['effect_size']:.4f}")
        if genre_trends is not None and not genre_trends.empty:
            top_genre = genre_trends.groupby("genre")["count"].sum().idxmax()
            findings.append(f"Most popular genre overall: {top_genre}")
        if not findings[2:]:
            findings.append("No analysis results available.")
        ax4.text(
            0.1, 0.9, "\n".join(findings),
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax4.set_title("Summary")

        fig.suptitle("Indian Cinema Research — Summary Dashboard", fontsize=16, y=1.01)
        fig.tight_layout()
        plt.close(fig)
        return fig
