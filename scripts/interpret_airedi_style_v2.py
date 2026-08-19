"""Shared figure design system for the AI-READI interpretability v2 suite.

Every v2 figure script imports from this module instead of hardcoding colors,
fonts, or spine/grid settings, so the whole figure set stays visually
consistent. See the project figure design style guide for the full rationale;
this module is its executable form.

Two color systems, kept deliberately separate:
  ROLE colors    -- semantic meaning (history, event, positive/negative, ...),
                    used in case-study time series and signed-value bar charts.
                    These are the five STRATUM_COLORS plus black/white.
  GROUP colors   -- categorical identity (which modality group a bar
                    represents), used in attribution bar charts. There are
                    more groups than base colors, so the group palette is
                    built by systematically tinting/shading the four
                    non-reserved STRATUM_COLORS (crimson, navy, teal, gray).
                    Bright red (#FF0000) is reserved for event markers only
                    and never used as a categorical fill.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]

# -- semantic role colors (case-study time series, signed bars) --------------
HISTORY_COLOR = "#888888"
EVENT_COLOR = "#FF0000"
REFERENCE_COLOR = "#003366"
REFERENCE_BAND_ALPHA = 0.15
CORRECTED_COLOR = "#5BBABA"
CORRECTED_BAND_ALPHA = 0.15
OBSERVED_COLOR = "#000000"
POSITIVE_COLOR = "#BA2828"
NEGATIVE_COLOR = "#003366"
NULL_COLOR = "#888888"
ANNOTATION_COLOR = "#888888"

PRIMARY_LINEWIDTH = 2.0
SECONDARY_LINEWIDTH = 1.0
MARKER_SIZE = 3.5
EVENT_MARKER = "^"
SHADED_BAND_ALPHA = 0.20

# -- categorical group palette, tints/shades of crimson/navy/teal/gray -------
_TINT_TOWARD_WHITE = 0.45
_SHADE_TOWARD_BLACK = 0.35


def _tint(hex_color: str, amount: float) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    r, g, b = (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
    return mcolors.to_hex((r, g, b))


def _shade(hex_color: str, amount: float) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    r, g, b = (r * (1 - amount), g * (1 - amount), b * (1 - amount))
    return mcolors.to_hex((r, g, b))


CRIMSON, NAVY, TEAL, GRAY = "#BA2828", "#003366", "#5BBABA", "#888888"

# 11 broad attribution groups, fixed color per group across every figure.
GROUP_COLORS_V2 = {
    "glucose history": CRIMSON,
    "heart rate / physiology": NAVY,
    "wearable activity / exercise": TEAL,
    "sleep": _tint(NAVY, _TINT_TOWARD_WHITE),
    "stress": _tint(CRIMSON, _TINT_TOWARD_WHITE),
    "data quality / missingness": GRAY,
    "time / calendar": _shade(GRAY, _SHADE_TOWARD_BLACK),
    "static clinical": _shade(NAVY, _SHADE_TOWARD_BLACK),
    "medication metadata": _shade(TEAL, _SHADE_TOWARD_BLACK),
    "site / cohort": _tint(GRAY, _TINT_TOWARD_WHITE),
    "meal proxy (diagnostic only)": _shade(CRIMSON, _SHADE_TOWARD_BLACK),
    # 12th group, environment-trained checkpoint only. A paler teal tint, distinct
    # from both "wearable activity / exercise" (raw TEAL) and "medication metadata"
    # (shaded TEAL) so all three stay visually separable in the same legend.
    "environmental exposure": _tint(TEAL, 0.70),
}

# 7 static-clinical domains (Phase 2), fixed color per domain.
STATIC_DOMAIN_COLORS_V2 = {
    "glycaemic_severity": CRIMSON,
    "anthropometrics": TEAL,
    "cardiovascular": NAVY,
    "metabolic_laboratory": _tint(NAVY, _TINT_TOWARD_WHITE),
    "demographics": GRAY,
    "medication_metadata": _shade(TEAL, _SHADE_TOWARD_BLACK),
    "site_and_cohort": _tint(GRAY, _TINT_TOWARD_WHITE),
}


def pretty_label(slug: str) -> str:
    """Convert an internal slug (domain/subgroup/variable name) to a readable,
    sentence-case display label: underscores to spaces, first letter capitalized."""
    text = str(slug).replace("_", " ").strip()
    return sentence_case(text)


def sentence_case(text: str) -> str:
    """First letter capitalized, everything else left as-is (preserves
    acronyms and units like mg/dL, HR, AUROC)."""
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def apply_whitegrid_style(ax, axis: str = "both"):
    """Seaborn whitegrid look: white panel, light gray gridlines, thin black
    frame on all four spines (never removed)."""
    ax.set_facecolor("white")
    ax.grid(True, axis=axis, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="black")


def style_legend(ax, ncol: int = 1, loc: str = "best"):
    leg = ax.legend(loc=loc, ncol=ncol, fontsize=8, frameon=True, facecolor="white",
                    edgecolor="black", framealpha=1.0)
    if leg is not None:
        leg.get_frame().set_linewidth(0.6)
    return leg


def annotate_muted(ax, x, y, text, ha="left", va="bottom", fontsize=7.5):
    ax.annotate(text, (x, y), color=ANNOTATION_COLOR, fontsize=fontsize, ha=ha, va=va)


def signed_bar_colors(values):
    return [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]


def set_title(ax_or_fig, text: str, fontsize: float = 10.5, pad: float = 6.0):
    if hasattr(ax_or_fig, "set_title"):
        ax_or_fig.set_title(sentence_case(text), fontsize=fontsize, fontweight="bold", loc="left", pad=pad)
    else:
        ax_or_fig.suptitle(sentence_case(text), fontsize=fontsize, fontweight="bold", x=0.01, ha="left")


def annotate_above_axes(ax, text: str, points_above_axes: float = 4.0, fontsize: float = 7.5):
    """Place muted subtitle/context text directly above the axes, using a fixed
    point offset (not axes-fraction) so it never collides with the title, which
    matplotlib also positions in points above the axes."""
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, points_above_axes),
               textcoords="offset points", color=ANNOTATION_COLOR, fontsize=fontsize, ha="left", va="bottom")
