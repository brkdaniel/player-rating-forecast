"""Figures for the README and the notebooks.

Every figure is rendered twice, for GitHub's light and dark themes. Colours follow one rule set:
thin marks, hairline solid gridlines, text in ink colours (never in series colours), the accent
hue for the series the story is about and grey for context.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .models.hierarchical import VarianceComponents  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "accent": "#2a78d6",
        "accent_2": "#eb6834",
        "context": "#a3a29b",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "accent": "#3987e5",
        "accent_2": "#d95926",
        "context": "#6f6e69",
    },
}
MARKER_SIZE = 8  # points; with a 2 pt ring in the surface colour


def _axes(theme: dict, figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["secondary"], labelsize=9, length=0, pad=6)
    ax.grid(True, color=theme["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    return fig, ax


def _titles(fig, ax, theme, title: str, subtitle: str | None = None) -> None:
    """Title and subtitle, aligned with the left edge of the axis labels."""
    fig.canvas.draw()
    box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(box.x0, 0.965, title, color=theme["ink"], fontsize=12, fontweight="bold", va="top")
    if subtitle:
        fig.text(box.x0, 0.905, subtitle, color=theme["secondary"], fontsize=9, va="top")


def _legend_below(ax, theme, n_columns: int = 2) -> None:
    """Legend under the x-axis label, where it cannot cover data."""
    legend = ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=n_columns,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(theme["secondary"])


def _dot(ax, x, y, color, theme, size=MARKER_SIZE, zorder=4):
    ax.plot(
        x,
        y,
        "o",
        markersize=size,
        markerfacecolor=color,
        markeredgecolor=theme["surface"],
        markeredgewidth=2,
        zorder=zorder,
    )


def save_both_themes(draw: Callable[[dict], plt.Figure], path_stem: Path) -> list[Path]:
    """Render ``draw(theme)`` for both themes as ``<stem>_light.png`` and ``<stem>_dark.png``."""
    paths = []
    for name, theme in THEMES.items():
        fig = draw(theme)
        path = path_stem.parent / f"{path_stem.name}_{name}.png"
        fig.savefig(path, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------------------------


def model_comparison(results: pd.DataFrame, components: VarianceComponents, selected: str):
    """Test R² with 95% intervals per model, against the theoretical ceiling."""
    ordered = results.sort_values("R2")

    def draw(theme):
        fig, ax = _axes(theme, figsize=(8, 0.42 * len(ordered) + 1.9))
        for row, (name, r) in enumerate(ordered.iterrows()):
            color = theme["accent"] if name == selected else theme["context"]
            ax.plot(
                [r["R2_low"], r["R2_high"]],
                [row, row],
                color=color,
                linewidth=2,
                solid_capstyle="round",
                zorder=3,
            )
            _dot(ax, r["R2"], row, color, theme)
            ax.text(
                r["R2_high"] + 0.004,
                row,
                f"{round(r['R2'], 3) + 0.0:.3f}",  # + 0.0 turns -0.0 into 0.0
                va="center",
                fontsize=8.5,
                color=theme["ink"] if name == selected else theme["secondary"],
            )
        for value, label in (
            (components.ceiling_r2, f"theoretical ceiling {components.ceiling_r2:.2f}"),
            (components.static_r2, f"static attributes only {components.static_r2:.2f}"),
        ):
            ax.axvline(value, color=theme["muted"], linewidth=1, zorder=1)
            ax.text(
                value,
                len(ordered) - 0.35,
                label,
                rotation=0,
                ha="right",
                va="bottom",
                fontsize=8,
                color=theme["secondary"],
            )
        ax.set_yticks(range(len(ordered)), ordered.index)
        ax.set_ylim(-0.6, len(ordered) + 0.2)
        ax.set_xlim(min(-0.01, ordered["R2_low"].min() - 0.01), components.ceiling_r2 + 0.03)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(
            "R² on the held-out final week (95% interval, matches resampled)",
            color=theme["secondary"],
            fontsize=9,
        )
        _titles(
            fig,
            ax,
            theme,
            "Every reasonable model ends up close to the predictability ceiling",
            "Blue: the selected model. Whiskers: 95% intervals from resampling matches",
        )
        fig.subplots_adjust(top=0.80)
        return fig

    return draw


def reliability_curve(history: pd.DataFrame, components: VarianceComponents, min_rows: int = 50):
    """Observed error by length of history versus the mixed-model expectation.

    Buckets with fewer than ``min_rows`` predictions are left out: their RMSE is too noisy.
    """
    history = history[history["n_rows"] >= min_rows]

    def draw(theme):
        fig, ax = _axes(theme)
        n = np.linspace(0, max(history["mean_prior_apps"].max(), 30), 200)
        ax.plot(
            n,
            np.sqrt(components.expected_mse(n)),
            color=theme["accent"],
            linewidth=2,
            solid_capstyle="round",
            label="Expected by the mixed model",
            zorder=3,
        )
        _dot(ax, history["mean_prior_apps"], history["RMSE"], theme["accent_2"], theme)
        ax.plot(
            [],
            [],
            "o",
            color=theme["accent_2"],
            markersize=MARKER_SIZE,
            label="Observed (walk-forward predictions)",
        )
        floor, no_model = np.sqrt(components.noise), np.sqrt(components.total)
        for level, label, va in (
            (floor, f"match-to-match noise alone: {floor:.3f}", "top"),
            (no_model, f"no model (always predict the average): {no_model:.3f}", "bottom"),
        ):
            ax.axhline(level, color=theme["muted"], linewidth=1, zorder=1)
            offset = -0.004 if va == "top" else 0.004
            ax.text(
                n.max(),
                level + offset,
                label,
                ha="right",
                va=va,
                fontsize=8,
                color=theme["secondary"],
            )
        ax.set_ylim(0.56, no_model + 0.03)
        ax.set_xlabel("Previous appearances of the player", color=theme["secondary"], fontsize=9)
        ax.set_ylabel("RMSE (rating points)", color=theme["secondary"], fontsize=9)
        _legend_below(ax, theme)
        _titles(
            fig,
            ax,
            theme,
            "Match-to-match noise sets a floor that more history cannot remove",
            f"Dots: observed walk-forward error (groups of at least {min_rows} predictions)",
        )
        fig.subplots_adjust(top=0.82)
        return fig

    return draw


def ablation_chart(ablation: pd.DataFrame, n_selected_groups: int):
    """Walk-forward R² as feature groups are added one at a time."""

    def draw(theme):
        fig, ax = _axes(theme, figsize=(8, 0.45 * len(ablation) + 1.9))
        rows = np.arange(len(ablation))[::-1]
        ax.plot(ablation["R2"], rows, color=theme["axis"], linewidth=1, zorder=2)
        r2, change = ablation["R2"].to_numpy(), ablation["R2 change"].to_numpy()
        for position, row in enumerate(rows):
            kept = position < n_selected_groups
            _dot(ax, r2[position], row, theme["accent"] if kept else theme["context"], theme)
            label = f"{r2[position]:.3f}" + (
                "" if position == 0 else f"  ({change[position]:+.3f})"
            )
            ax.text(
                r2[position] + 0.0015,
                row,
                label,
                va="center",
                fontsize=8.5,
                color=theme["ink"] if kept else theme["secondary"],
            )
        ax.set_yticks(rows, [f"+ {name}" for name in ablation["added group"]])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(ablation["R2"].min() - 0.01, ablation["R2"].max() + 0.015)
        ax.set_xlabel(
            "Walk-forward R² (ridge, groups added cumulatively)",
            color=theme["secondary"],
            fontsize=9,
        )
        _titles(
            fig,
            ax,
            theme,
            "Three feature groups carry all the signal",
            "Blue: kept in the final model. Grey: adding them makes validation error worse",
        )
        fig.subplots_adjust(top=0.80)
        return fig

    return draw


def shrinkage_curve(curve: pd.DataFrame, k_theory: float):
    """Validation error for different shrinkage constants k, against the mixed-model value."""

    def draw(theme):
        fig, ax = _axes(theme)
        ax.plot(curve["k"], curve["MAE"], color=theme["accent"], linewidth=2, zorder=3)
        _dot(ax, curve["k"], curve["MAE"], theme["accent"], theme, size=6)
        best = curve.loc[curve["MAE"].idxmin()]
        ax.axvline(k_theory, color=theme["muted"], linewidth=1, zorder=1)
        ax.text(
            k_theory * 1.06,
            curve["MAE"].max(),
            f"mixed-model k = {k_theory:.1f}",
            fontsize=8.5,
            color=theme["secondary"],
            va="top",
        )
        _dot(ax, best["k"], best["MAE"], theme["ink"], theme, size=9, zorder=5)
        ax.text(
            0.98,
            0.55,
            f"lowest validation error: k = {best['k']:g}",
            transform=ax.transAxes,
            ha="right",
            fontsize=8.5,
            color=theme["ink"],
        )
        ax.set_xscale("log")
        ax.set_xticks(curve["k"], [f"{k:g}" for k in curve["k"]])
        ax.minorticks_off()
        ax.set_xlabel("Shrinkage constant k (log scale)", color=theme["secondary"], fontsize=9)
        ax.set_ylabel("Walk-forward MAE", color=theme["secondary"], fontsize=9)
        _titles(
            fig,
            ax,
            theme,
            "The shrinkage strength picked by theory is the one validation picks",
            "k = noise variance / player variance, estimated on the training weeks only",
        )
        fig.subplots_adjust(top=0.82)
        return fig

    return draw


def coverage_chart(coverage: pd.DataFrame):
    """Empirical coverage of the 80% and 90% conformal intervals, overall and per position."""
    groups = list(dict.fromkeys(coverage["group"]))

    def draw(theme):
        fig, ax = _axes(theme, figsize=(8, 0.5 * len(groups) + 1.9))
        rows = {group: row for row, group in enumerate(groups[::-1])}
        for level, color, offset in ((0.8, theme["accent"], 0.12), (0.9, theme["accent_2"], -0.12)):
            part = coverage[coverage["level"] == level]
            ax.axvline(level, color=theme["muted"], linewidth=1, zorder=1)
            ys = [rows[g] + offset for g in part["group"]]
            _dot(ax, part["coverage"], ys, color, theme, size=7)
            ax.plot([], [], "o", color=color, markersize=7, label=f"{level:.0%} interval")
        ax.set_yticks(list(rows.values()), list(rows))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.70, 0.97)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(
            "Share of test ratings inside the interval (vertical lines: target)",
            color=theme["secondary"],
            fontsize=9,
        )
        _legend_below(ax, theme)
        _titles(
            fig,
            ax,
            theme,
            "Conformal intervals hit their target coverage on unseen matches",
            "On target overall; goalkeepers are over-covered because their ratings are less noisy",
        )
        fig.subplots_adjust(top=0.80)
        return fig

    return draw


def render_all(metrics_dir: Path, figures_dir: Path) -> list[Path]:
    """Draw every report figure from the files the pipeline saved in ``metrics_dir``."""
    import json

    from .experiments import V2_GROUPS

    components = json.loads((metrics_dir / "variance_components.json").read_text())
    components = VarianceComponents(
        static=components["static"], player=components["player"], noise=components["noise"]
    )
    summary = json.loads((metrics_dir / "run_summary.json").read_text())
    results = pd.read_csv(metrics_dir / "test_results.csv", index_col="model")
    figures_dir.mkdir(parents=True, exist_ok=True)
    drawings = {
        "model_comparison": model_comparison(results, components, summary["selected_model"]),
        "error_vs_history": reliability_curve(
            pd.read_csv(metrics_dir / "error_by_history_walk_forward.csv"), components
        ),
        "ablation": ablation_chart(pd.read_csv(metrics_dir / "ablation.csv"), len(V2_GROUPS)),
        "shrinkage_k": shrinkage_curve(
            pd.read_csv(metrics_dir / "shrinkage_curve.csv"), components.shrinkage_k
        ),
        "interval_coverage": coverage_chart(pd.read_csv(metrics_dir / "conformal_coverage.csv")),
    }
    paths = []
    for name, draw in drawings.items():
        paths += save_both_themes(draw, figures_dir / name)
    return paths


if __name__ == "__main__":
    from .config import FIGURES_DIR, METRICS_DIR

    for path in render_all(METRICS_DIR, FIGURES_DIR):
        print("wrote", path)
