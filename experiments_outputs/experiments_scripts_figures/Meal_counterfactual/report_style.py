"""Shared plot style for the SSM-CGM report meal-flag figures.

Import this module and call apply_style() at the top of every figure script.
All figures share the same rcParams, color palette, and helpers.

Color semantics (use exactly these hex values, no others):
    C_LEAKY   = "#BA2828"  crimson  -- glucose-defined / leaky / strong baseline
    C_WEARABLE= "#5BBABA"  teal     -- wearable pre-rise / causally correct
    C_NAVY    = "#003366"  navy     -- second reference or primary headline
    C_GRAY    = "#888888"  gray     -- null / pure / de-emphasised
    C_EVENT   = "#FF0000"  bright red -- event markers and threshold walls only
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---- palette ----
C_LEAKY    = "#BA2828"   # glucose-defined arm, leaky oracle
C_WEARABLE = "#5BBABA"   # wearable pre-rise arm, hero
C_NAVY     = "#003366"   # second reference, delta_obs band
C_GRAY     = "#888888"   # null, de-emphasised
C_EVENT    = "#FF0000"   # event markers, threshold walls

# convenience aliases matching style_reference_figure.py names
C_GRID      = C_LEAKY
C_RETRIEVAL = C_NAVY
C_GEN_PURE  = C_GRAY
C_GEN_HERO  = C_WEARABLE


def apply_style() -> None:
    """Apply shared rcParams. Call once at the top of each plotting script."""
    mpl.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          10,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.framealpha":  0.7,
        "legend.edgecolor":   "#cccccc",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "--",
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "pdf.fonttype":       42,   # embed fonts as TrueType in PDF
    })


def despine(ax: plt.Axes) -> None:
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def bar_value_labels(
    ax: plt.Axes,
    bars,
    values,
    fmt: str = "{:.1f}",
    offset: float | None = None,
    fontsize: int = 9,
    color: str = "black",
) -> None:
    """Annotate a bar chart with bold value labels above each bar.

    Parameters
    ----------
    ax     : the axes containing the bars
    bars   : container returned by ax.bar()
    values : sequence of numeric values (same order as bars)
    fmt    : format string for the label
    offset : vertical offset in data units; defaults to 1% of y range
    """
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    if offset is None:
        offset = y_range * 0.015
    for bar, val in zip(bars, values):
        if not np.isfinite(float(val)):
            continue
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        label_y = y + offset if y >= 0 else y - offset * 3
        ax.text(
            x, label_y, fmt.format(val),
            ha="center", va="bottom",
            fontweight="bold", fontsize=fontsize, color=color,
        )


def hline(ax: plt.Axes, y: float, label: str, ls: str = ":", lw: float = 2,
          color: str = C_NAVY, text_x: float = 0.98, fontsize: int = 8) -> None:
    """Draw a labeled horizontal reference line."""
    ax.axhline(y, ls=ls, lw=lw, color=color, zorder=1)
    ax.text(
        text_x, y, f" {label}",
        transform=ax.get_yaxis_transform(),
        va="center", ha="right", fontsize=fontsize, color=color,
    )


def save_fig(fig: plt.Figure, path: str | None = None, name: str = "figure") -> None:
    """Save figure as vector PDF (and a companion PNG for quick preview)."""
    import os
    base = path or name
    pdf_path = base if base.endswith(".pdf") else base + ".pdf"
    png_path = pdf_path.replace(".pdf", ".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
    print(f"[save] {os.path.abspath(pdf_path)}")
    print(f"[save] {os.path.abspath(png_path)}")
