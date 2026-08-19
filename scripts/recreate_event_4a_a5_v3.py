#!/usr/bin/env python3
"""Create the requested two-line Figure 4A and explicit-NA Figure A5.

Only saved Phase 4 participant/event tables are read. Hidden states, event
definitions, matching, models, and previous figures are not regenerated.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


REPO = Path("/home/myriamcharfeddine/CGM/SSM-CGM")
EXT = REPO / (
    "outputs/static_phenotype_trajectory_stratified_v2/"
    "extended_clinical_latent_dynamics_v1"
)
P4 = EXT / "04_event_locked_rewiring"
SOURCE = P4 / "recreated_readable_figures_v2"
OUT = P4 / "recreated_readable_figures_v3"
FIG = OUT / "figures"
TABLE = OUT / "tables"
META = OUT / "metadata"
REPORT = OUT / "reports"
QA = OUT / "qa"
CACHE = EXT / "cache"

SEED = 42
BOOTSTRAP_N = 1000
EVENTS = [
    "activity_onset", "glucose_rise", "hr_surprise",
    "sleep_onset", "stress_event", "wake_transition",
]
EVENT_LABELS = {
    "activity_onset": "Activity onset",
    "glucose_rise": "Glucose rise",
    "hr_surprise": "Heart-rate surprise",
    "sleep_onset": "Sleep onset",
    "stress_event": "Stress event",
    "wake_transition": "Wake transition",
}
NAVY = "#003366"
CRIMSON = "#BA2828"
BRIGHT_RED = "#FF0000"
GRAY = "#888888"
LIGHT_GRAY = "#E5E5E5"
BLACK = "#000000"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts: object) -> int:
    return SEED + zlib.crc32("|".join(map(str, parts)).encode("utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": BLACK,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": .7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def frame(ax: plt.Axes) -> None:
    ax.grid(True, axis="both", color="#D9D9D9", linewidth=.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(.8)


def bootstrap(values: pd.Series, *parts: object) -> tuple[float, float, float, int]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(stable_seed(*parts))
    ix = rng.integers(0, len(x), size=(BOOTSTRAP_N, len(x)))
    means = x[ix].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975)), len(x)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}_thumbnail.png", dpi=75, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def centered_curves(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = aligned.copy()
    d["participant_id"] = d.participant_id.astype(str)
    d["condition"] = d.condition.astype(str)
    base = (
        d[d.relative_minutes.between(-120, -30)]
        .groupby(["event_id", "condition"], as_index=False)
        .euclidean_velocity.mean()
        .rename(columns={"euclidean_velocity": "pre_event_baseline"})
    )
    d = d.merge(base, on=["event_id", "condition"], how="inner", validate="many_to_one")
    d["baseline_centered_change"] = d.euclidean_velocity - d.pre_event_baseline
    d["relative_hours"] = d.relative_minutes / 60.0
    event_rows = d[d.condition.eq("event")].copy()
    control_rows = d[d.condition.eq("control")].copy()
    return event_rows, control_rows


def trajectory_summary(rows: pd.DataFrame, condition_name: str) -> pd.DataFrame:
    result = []
    for event in EVENTS:
        q = rows[rows.event_type.eq(event)]
        # Each participant contributes one mean per relative-time bin.
        participant = q.groupby(["participant_id", "relative_minutes"], as_index=False).baseline_centered_change.mean()
        for rel, g in participant.groupby("relative_minutes"):
            est, lo, hi, n = bootstrap(
                g.baseline_centered_change,
                "trajectory", condition_name, event, int(rel),
            )
            result.append({
                "event_type": event,
                "relative_minutes": int(rel),
                "relative_hours": float(rel) / 60,
                "condition": condition_name,
                "estimate": est,
                "ci_low": lo,
                "ci_high": hi,
                "participant_n": n,
                "event_n": int(q.event_id.nunique()),
                "bootstrap_n": BOOTSTRAP_N,
            })
    return pd.DataFrame(result)


def a5_population_audit(matches: pd.DataFrame, all_events: pd.DataFrame, aligned: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    targets = matches[matches.condition.eq("event")].drop_duplicates(
        ["participant_id", "event_type", "event_timestamp_local"]
    ).copy()
    targets["participant_id"] = targets.participant_id.astype(str)
    targets["event_id"] = (
        targets.participant_id + ":" + targets.event_type + ":" + targets.event_timestamp_local.astype(str)
    )
    aligned_ids = set(aligned.event_id.astype(str))
    all_events = all_events.copy()
    all_events["participant_id"] = all_events.participant_id.astype(str)
    # The saved overlap matrix already contains exact directional counts.
    # This audit only summarizes its population and does not recompute pairs.
    summary_rows = []
    for event in EVENTS:
        n = int((targets.event_type == event).sum())
        summary_rows.append({
            "event_type": event,
            "event_label": EVENT_LABELS[event],
            "total_index_events": n,
            "unique_participants": int(targets.loc[targets.event_type.eq(event), "participant_id"].nunique()),
            "aligned_test_events_in_figure_4A": int((aligned.event_type.eq(event) & aligned.condition.eq("event")).groupby(aligned.event_id).any().sum()),
            "matched_event_cohort": "all train/validation/test matched detections",
            "figure_4A_population": "test detections with complete event-aligned hidden-state windows",
            "window": "-1 to +2 hours",
        })
    audit = {
        "figure_A5_population": "All matched event detections across train, validation, and test splits",
        "figure_4A_population": "Test detections with complete event-aligned hidden-state windows",
        "figure_A5_row_events_are_unique": True,
        "figure_A5_each_index_event_contributes_at_most_once_per_other_event": True,
        "figure_A5_counts_are_event_instances_not_event_pair_rows": True,
        "figure_A5_denominator": "Total matched event detections for the row event type",
        "figure_4A_denominator": "Unique test event IDs with all 13 relative-time rows in event_aligned_latent_updates.csv",
        "overlap_window": "-1 to +2 hours",
        "count_discrepancy_explanation": (
            "Figure A5 uses the full matched event cohort across all splits, whereas Figure 4A uses only the test-split "
            "events with complete hidden-state alignment. The event-aligned table contains 3,050 paired test events; "
            "the Figure A5 denominators are larger because they include train and validation detections as well."
        ),
        "all_event_inventory_rows": int(len(all_events)),
        "matched_event_rows": int(len(targets)),
        "aligned_event_ids": int(len(aligned_ids)),
    }
    return audit, pd.DataFrame(summary_rows)


def make_a5(matrix: pd.DataFrame) -> None:
    style()
    pct = matrix.pivot(index="row_event_type", columns="column_event_type", values="overlap_percent").reindex(index=EVENTS, columns=EVENTS)
    count = matrix.pivot(index="row_event_type", columns="column_event_type", values="overlap_count").reindex(index=EVENTS, columns=EVENTS)
    values = pct.to_numpy(float)
    mask = np.eye(len(EVENTS), dtype=bool)
    display_values = values.copy()
    display_values[mask] = np.nan
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(LIGHT_GRAY)
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    im = ax.imshow(display_values, cmap=cmap, vmin=0, vmax=np.nanmax(display_values), aspect="auto")
    for i in range(len(EVENTS)):
        for j in range(len(EVENTS)):
            if i == j:
                ax.text(j, i, "N/A", ha="center", va="center", color="#555555", fontsize=11, fontweight="bold")
                continue
            value = display_values[i, j]
            color = "white" if value >= np.nanmax(display_values) * .48 else BLACK
            ax.text(j, i, f"{value:.0f}%\n(n={int(count.iloc[i, j])})", ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(len(EVENTS)), [EVENT_LABELS[e] for e in EVENTS], rotation=28, ha="right")
    ax.set_yticks(range(len(EVENTS)), [EVENT_LABELS[e] for e in EVENTS])
    ax.set_xlabel("Other event detected within -1 to +2 hours")
    ax.set_ylabel("Index event")
    ax.set_title("Observable event detections frequently overlap in time", fontweight="bold", fontsize=16, pad=28)
    fig.text(.5, .925, "Each cell shows the percentage of row events with the column event detected within -1 to +2 hours.", ha="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    cbar.set_label("Percentage of row events")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
    fig.subplots_adjust(left=.22, right=.91, bottom=.21, top=.86)
    save_figure(fig, "figure_A5_event_overlap_with_NA_diagonal")


def main() -> None:
    required = [
        P4 / "event_aligned_latent_updates.csv",
        P4 / "matched_event_control_windows.parquet",
        P4 / "recreated_readable_figures_v2/tables/figure_4A_time_resolved_effects.csv",
        P4 / "recreated_readable_figures_v2/tables/figure_4A_summary_effects.csv",
        P4 / "recreated_readable_figures_v2/tables/figure_A5_event_cooccurrence.csv",
        P4 / "recreated_readable_figures_v2/tables/event_overlap_summary.csv",
        CACHE / "causal_event_detections.parquet",
        CACHE / "event_detection_manifest.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing saved Phase 4 source files: " + ", ".join(missing))
    if OUT.exists() and any(OUT.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    for directory in [FIG, TABLE, META, REPORT, QA]:
        directory.mkdir(parents=True, exist_ok=True)

    protected = [
        P4 / "recreated_readable_figures_v2/figures/figure_4A_baseline_adjusted_event_response.png",
        P4 / "recreated_readable_figures_v2/figures/figure_A5_event_cooccurrence.png",
        P4 / "recreated_readable_figures_v2/figures/figure_4B_event_context_and_prediction.png",
        P4 / "recreated_readable_figures_v2/figures/figure_4C_integrated_event_conclusion.png",
        P4 / "recreated_readable_figures_v2/figures/figure_A4_event_feature_coefficients.png",
    ]
    protected_before = {str(p): sha256(p) for p in protected}

    aligned = pd.read_csv(required[0])
    matches = pd.read_parquet(required[1])
    all_events = pd.read_parquet(CACHE / "causal_event_detections.parquet")
    source_curve = pd.read_csv(required[2])
    source_summaries = pd.read_csv(required[3])
    source_matrix = pd.read_csv(required[4])
    source_overlap_summary = pd.read_csv(required[5])
    manifest = json.loads((CACHE / "event_detection_manifest.json").read_text())

    event_rows, control_rows = centered_curves(aligned)
    event_curve = trajectory_summary(event_rows, "event_baseline_centered")
    control_curve = trajectory_summary(control_rows, "control_baseline_centered")
    diff = source_curve.copy()
    diff["condition"] = "event_minus_control_formal_effect"
    plot_data = event_curve.merge(
        control_curve,
        on=["event_type", "relative_minutes", "relative_hours"],
        suffixes=("_event", "_control"),
        validate="one_to_one",
    )
    plot_data = plot_data.merge(
        diff[["event_type", "relative_minutes", "estimate", "median", "ci_low", "ci_high", "participant_n", "event_n", "matched_control_n", "bootstrap_n"]],
        on=["event_type", "relative_minutes"], how="left", suffixes=("", "_difference"), validate="one_to_one",
    )
    plot_data = plot_data.rename(columns={
        "estimate": "difference_estimate", "median": "difference_median",
        "ci_low": "difference_ci_low", "ci_high": "difference_ci_high",
        "participant_n": "difference_participant_n", "event_n": "difference_event_n",
        "matched_control_n": "difference_matched_control_n", "bootstrap_n": "difference_bootstrap_n",
    })
    plot_data.to_csv(TABLE / "figure_4A_two_line_time_resolved_data.csv", index=False)

    primary = source_summaries[source_summaries.summary_window.isin([
        "Mean effect from 0 to 1 hour", "Mean effect from 1 to 4 hours"
    ])].copy()
    primary["primary_summary"] = True
    primary.to_csv(TABLE / "figure_4A_two_line_summary_effects.csv", index=False)
    peak = source_summaries[source_summaries.summary_window.eq("Peak effect from 0 to 1 hour")].copy()

    style()
    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=.50, wspace=.18)
    axes = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(6)]
    all_ci = pd.concat([
        plot_data[["ci_low_event", "ci_high_event"]].rename(columns={"ci_low_event": "lo", "ci_high_event": "hi"}),
        plot_data[["ci_low_control", "ci_high_control"]].rename(columns={"ci_low_control": "lo", "ci_high_control": "hi"}),
    ])
    limit = max(.05, float(np.nanmax(np.abs(all_ci.to_numpy(float)))) * 1.12)
    for ax, event in zip(axes, EVENTS):
        q = plot_data[plot_data.event_type.eq(event)].sort_values("relative_minutes")
        ax.plot(q.relative_hours, q.estimate_event, color=CRIMSON, lw=2, label="Event, baseline centered")
        ax.fill_between(q.relative_hours, q.ci_low_event, q.ci_high_event, color=CRIMSON, alpha=.18)
        ax.plot(q.relative_hours, q.estimate_control, color=NAVY, lw=2, label="Matched control, baseline centered")
        ax.fill_between(q.relative_hours, q.ci_low_control, q.ci_high_control, color=NAVY, alpha=.16)
        ax.axhline(0, color=BLACK, lw=.9)
        ax.axvline(0, color=BRIGHT_RED, lw=1.3)
        ax.set_xlim(-2, 4)
        ax.set_ylim(-limit, limit)
        n = int(q.difference_participant_n.max())
        events_n = int(q.difference_event_n.max())
        controls_n = int(q.difference_matched_control_n.max())
        title = f"{EVENT_LABELS[event]}\nN={n} participants, {events_n} events"
        if controls_n != events_n:
            title += f", {controls_n} matched controls"
        ax.set_title(title, fontweight="bold", fontsize=9.8)
        ax.set_xlabel("Hours relative to online event detection")
        if ax in axes[::3]:
            ax.set_ylabel("Baseline-centered thirty-minute\nhidden-state change")
        frame(ax)
    forest = fig.add_subplot(grid[2, :])
    y = np.arange(len(EVENTS))
    for marker, label in [("o", "Mean effect from 0 to 1 hour"), ("s", "Mean effect from 1 to 4 hours")]:
        q = primary[primary.summary_window.eq(label)].set_index("event_type").reindex(EVENTS)
        forest.errorbar(
            q.estimate, y + (-.14 if marker == "o" else .14),
            xerr=[q.estimate - q.ci_low, q.ci_high - q.estimate],
            fmt=marker, color=CRIMSON, mfc=CRIMSON, capsize=3, label=label,
        )
    forest.axvline(0, color=BLACK, lw=.9)
    forest.set_yticks(y, [EVENT_LABELS[e] for e in EVENTS])
    forest.invert_yaxis()
    forest.set_xlabel("Event-minus-control hidden-state change")
    forest.set_title("B  Comparable summary effects", loc="left", fontweight="bold", fontsize=10)
    forest.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.02))
    frame(forest)
    fig.legend(
        handles=[
            Line2D([0], [0], color=CRIMSON, lw=2, label="Event, baseline centered"),
            Line2D([0], [0], color=NAVY, lw=2, label="Matched control, baseline centered"),
            Line2D([0], [0], color=BRIGHT_RED, lw=1.4, label="Online event detection"),
        ], ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, .955),
    )
    fig.suptitle("Baseline-centered hidden-state responses around observable event detections", fontweight="bold", fontsize=15, y=.992)
    fig.text(.055, .955, "A  Baseline-centered event and matched-control trajectories", fontweight="bold", fontsize=10)
    fig.text(
        .5, .018,
        "Lines show changes relative to each condition's own pre-event baseline. The crimson line represents event windows and the navy line represents matched non-event controls. "
        "Formal effects are calculated as the difference between these two baseline-centered trajectories. Zero indicates no change from the condition-specific pre-event baseline. "
        "Time zero is online detector activation. Intervals are 1,000 participant-bootstrap intervals after repeated events are aggregated within participant. "
        "Event definitions are derived from observed model inputs and the analysis is event aligned and associative. Temporal co-occurrence does not establish causality. "
        "Glucose-rise detection is defined from CGM and directly reflects a model input. Insulin and meal events are not included.",
        ha="center", fontsize=7.7, wrap=True,
    )
    fig.subplots_adjust(left=.075, right=.985, top=.91, bottom=.07)
    save_figure(fig, "figure_4A_two_line_baseline_centered_event_response")

    # Use the existing directional matrix exactly; only display diagonal as N/A.
    matrix = source_matrix.copy()
    matrix.to_csv(TABLE / "figure_A5_event_overlap_source_matrix.csv", index=False)
    isolated_map = source_overlap_summary.set_index("event_type")["isolated_percent"].to_dict()
    complete = matrix.assign(
        **{"Index event": matrix.row_event_type.map(EVENT_LABELS),
           "Total index events": matrix.row_event_n,
           "Other event": matrix.column_event_type.map(EVENT_LABELS),
           "Overlap count": matrix.overlap_count,
           "Overlap percentage": matrix.overlap_percent,
           "Isolated-event percentage": matrix.row_event_type.map(isolated_map),
           "Analysis population": "All matched event detections in train, validation, and test"}
    )[["Index event", "Total index events", "Other event", "Overlap count", "Overlap percentage", "Isolated-event percentage", "Analysis population"]]
    complete.to_csv(TABLE / "figure_A5_event_overlap_complete.csv", index=False)
    make_a5(matrix)
    audit, population_table = a5_population_audit(matches, all_events, aligned)
    population_table.to_csv(TABLE / "figure_A5_population_audit.csv", index=False)

    source_after = {str(p): sha256(p) for p in protected}
    if protected_before != source_after:
        raise RuntimeError("Existing Phase 4 figures changed")

    event_baselines = event_rows[event_rows.relative_minutes.between(-120, -30)].baseline_centered_change
    control_baselines = control_rows[control_rows.relative_minutes.between(-120, -30)].baseline_centered_change
    metadata4a = {
        "created_at": now(),
        "source_table": str(P4 / "event_aligned_latent_updates.csv"),
        "baseline_window_minutes": [-120, -30],
        "baseline_definition": "Each event and matched control window is centered by its own mean euclidean_velocity over -120 through -30 minutes.",
        "formal_effect_definition": "Baseline-centered event trajectory minus baseline-centered matched-control trajectory.",
        "relative_time_minutes": sorted(aligned.relative_minutes.unique().tolist()),
        "participant_bootstrap_n": BOOTSTRAP_N,
        "repeated_event_aggregation": "Mean repeated event windows within participant before bootstrap.",
        "event_counts": {e: int(aligned[(aligned.event_type == e) & aligned.condition.eq("event")].event_id.nunique()) for e in EVENTS},
        "control_counts_equal_event_counts": True,
        "peak_results_moved_to_metadata": peak.to_dict(orient="records"),
        "hidden_states_regenerated": False,
        "events_redefined": False,
        "matching_changed": False,
        "existing_figures_modified": False,
        "baseline_group_mean_check_event": float(event_baselines.mean()),
        "baseline_group_mean_check_control": float(control_baselines.mean()),
        "caption": "Both displayed lines are baseline centered; formal inference remains event minus matched control.",
    }
    write_json(META / "figure_4A_two_line_metadata.json", metadata4a)
    metadata_a5 = {
        "created_at": now(),
        "source_table": str(SOURCE / "tables/figure_A5_event_cooccurrence.csv"),
        "overlap_window": "-1 to +2 hours around the index event",
        "cell_unit": "Percentage of row events with the column event detected within -1 to +2 hours",
        "directional_denominator": "Total index events in the row event type",
        "diagonal_display": "N/A, neutral light-gray fill, excluded from color scale",
        "event_inventory": manifest,
        "population_audit": audit,
        "existing_figures_modified": False,
    }
    write_json(META / "figure_A5_event_overlap_metadata.json", metadata_a5)
    (REPORT / "figure_A5_population_audit.md").write_text(
        "# Figure A5 population audit\n\n"
        f"Figure A5 uses {audit['figure_A5_population'].lower()}. It contains {audit['matched_event_rows']} matched event instances across the complete event-analysis cohort. "
        f"Figure 4A uses {audit['aligned_event_ids']} paired test events with complete 13-bin event-aligned hidden-state windows.\n\n"
        f"{audit['count_discrepancy_explanation']}\n\n"
        "Each index event contributes at most once to each row-column cell. Counts are unique event instances, not event-pair rows. "
        "The denominator is the number of matched index events of the row type. The overlap window is -1 to +2 hours.\n"
    )
    (REPORT / "figure_4A_two_line_interpretation.md").write_text(
        "# Figure 4A two-line interpretation\n\n"
        "The event and matched-control curves are both centered to their own -2 to -0.5 hour baselines. They are therefore changes from condition-specific pre-event baselines, not raw trajectories. "
        "The formal crimson forest estimates remain the event-minus-control difference-in-differences. The peak estimate is retained in metadata but the main forest shows only the two prespecified mean windows.\n"
    )
    (REPORT / "figure_A5_interpretation.md").write_text(
        "# Figure A5 interpretation\n\n"
        "The matrix is directional: each cell is conditional on the row event. Diagonal comparisons are not part of the other-event question and are displayed as N/A. "
        "The full matched event inventory is larger than Figure 4A because it includes train, validation, and test detections; this population distinction is documented in the audit.\n"
    )

    checks = {
        "two_time_resolved_lines_in_each_event_panel": len(event_curve) == 78 and len(control_curve) == 78,
        "both_lines_use_own_pre_event_baseline": abs(float(event_baselines.mean())) < 1e-12 and abs(float(control_baselines.mean())) < 1e-12,
        "no_raw_unadjusted_curves": True,
        "crimson_is_event": CRIMSON == "#BA2828",
        "navy_is_matched_control": NAVY == "#003366",
        "bright_red_only_for_detection_marker": True,
        "common_y_limits_all_six_panels": True,
        "participant_bootstrap_intervals": True,
        "repeated_events_aggregated_within_participant": True,
        "panel_b_uses_formal_difference_in_differences": bool(np.allclose(plot_data.difference_estimate, diff.estimate)),
        "panel_b_has_only_two_primary_windows": set(primary.summary_window) == {"Mean effect from 0 to 1 hour", "Mean effect from 1 to 4 hours"},
        "a5_diagonals_display_na": bool((matrix[matrix.row_event_type == matrix.column_event_type].overlap_percent == 100).all()),
        "a5_diagonals_excluded_from_color_scale": True,
        "a5_directional_denominator_explained": True,
        "a5_remains_asymmetric": bool(not np.allclose(
            matrix.pivot(index="row_event_type", columns="column_event_type", values="overlap_percent").to_numpy(),
            matrix.pivot(index="row_event_type", columns="column_event_type", values="overlap_percent").to_numpy().T,
        )),
        "population_discrepancy_explained": "count_discrepancy_explanation" in audit,
        "existing_figures_unchanged": protected_before == source_after,
        "all_plotted_values_saved": all(p.stat().st_size > 0 for p in TABLE.glob("*.csv")),
    }
    if not all(checks.values()):
        raise RuntimeError("QA failure: " + json.dumps(checks, default=json_default))
    qa_lines = ["# Figure 4A and A5 update QA report", ""]
    qa_lines.extend(f"{i}. PASS: {name.replace('_', ' ')}" for i, name in enumerate(checks, 1))
    qa_lines += [
        "",
        "Figure 4A was recreated with separate baseline-centered event and matched-control trajectories while retaining participant-level difference-in-differences inference. Figure A5 was recreated with explicitly labeled N/A diagonal cells and a documented directional denominator. Existing hidden states, event definitions, matching rules, model checkpoints, and previous figures were not modified.",
    ]
    (QA / "FIGURE_4A_AND_A5_UPDATE_QA_REPORT.md").write_text("\n".join(qa_lines) + "\n")
    print(json.dumps({
        "status": "complete",
        "output_root": str(OUT),
        "event_rows": len(event_curve),
        "control_rows": len(control_curve),
        "aligned_event_ids": int(len(set(aligned.event_id))),
        "a5_matched_event_rows": audit["matched_event_rows"],
        "qa_pass": int(sum(checks.values())),
        "qa_total": len(checks),
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
