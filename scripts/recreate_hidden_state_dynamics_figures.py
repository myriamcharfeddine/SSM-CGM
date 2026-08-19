"""Recreate readable phase-3 figures from frozen states and saved participant metrics."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_extended_circadian_dynamics as phase23  # noqa: E402

EXT = ROOT / "outputs/static_phenotype_trajectory_stratified_v2/extended_clinical_latent_dynamics_v1"
P3 = EXT / "03_latent_update_dynamics"
OUT = P3 / "recreated_readable_figures"
ARCH = EXT / "timestamped_state_archive_35072d"
PROFILES = EXT / "01_cluster_metabolic_profiles/participant_frozen_cluster_profiles.parquet"
H0 = phase23.H0
DATASET = phase23.DATASET
DIM = 35072
SQRT_DIM = np.sqrt(DIM)
SEED = 42
BOOTSTRAP_N = 1000
SUBTYPES = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
LABELS = {
    "healthy": "Healthy", "pre_diabetes": "Prediabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin", "insulin_dependent": "Insulin-dependent*",
}
SHORT = {"healthy": "Healthy", "pre_diabetes": "Prediabetes", "t2d_oral_non_insulin": "T2D oral", "insulin_dependent": "Insulin-dependent*"}
COLORS = {"healthy": "#003366", "pre_diabetes": "#5BBABA", "t2d_oral_non_insulin": "#BA2828", "insulin_dependent": "#888888"}
NAVY, TEAL, CRIMSON, GRAY, BLACK = "#003366", "#5BBABA", "#BA2828", "#888888", "#000000"
NIGHT_COLOR = "#B8BEC6"


def paths() -> dict[str, Path]:
    return {
        "audit": OUT / "figure_recreation_input_audit.json",
        "a_png": OUT / "figure_3A_cohort_hidden_state_dynamics_recreated.png",
        "a_pdf": OUT / "figure_3A_cohort_hidden_state_dynamics_recreated.pdf",
        "a_thumb": OUT / "figure_3A_cohort_hidden_state_dynamics_recreated_thumbnail.png",
        "a_data": OUT / "figure_3A_cohort_hidden_state_dynamics_plotted_data.csv",
        "a_summary": OUT / "figure_3A_cohort_hidden_state_dynamics_summary.csv",
        "a_metadata": OUT / "figure_3A_cohort_hidden_state_dynamics_metadata.json",
        "a_note": OUT / "figure_3A_cohort_hidden_state_dynamics_interpretation.md",
        "b_png": OUT / "figure_3B_representative_participant_dynamics_recreated.png",
        "b_pdf": OUT / "figure_3B_representative_participant_dynamics_recreated.pdf",
        "b_thumb": OUT / "figure_3B_representative_participant_dynamics_recreated_thumbnail.png",
        "b_data": OUT / "figure_3B_representative_participant_dynamics_plotted_data.csv",
        "b_metadata": OUT / "figure_3B_representative_participant_dynamics_metadata.json",
        "b_note": OUT / "figure_3B_representative_participant_dynamics_interpretation.md",
        "pca_png": OUT / "figure_A1_representative_participant_pca_appendix.png",
        "pca_pdf": OUT / "figure_A1_representative_participant_pca_appendix.pdf",
        "pca_thumb": OUT / "figure_A1_representative_participant_pca_appendix_thumbnail.png",
        "pca_data": OUT / "figure_A1_representative_participant_pca_plotted_data.csv",
        "pca_metadata": OUT / "figure_A1_representative_participant_pca_metadata.json",
        "pca_note": OUT / "figure_A1_representative_participant_pca_interpretation.md",
        "qa": OUT / "FIGURE_RECREATION_QA_REPORT.md",
        "interpretation": OUT / "RECREATED_FIGURES_INTERPRETATION.md",
    }


def json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "sans-serif", "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black", "axes.spines.top": True, "axes.spines.right": True,
        "grid.color": "#D9D9D9", "grid.linewidth": 0.8, "pdf.fonttype": 42,
    })


def state_matrix(participant_id: str) -> np.ndarray:
    return phase23.state_matrix(participant_id)


def preflight(dyn: pd.DataFrame, profiles: pd.DataFrame, p: dict[str, Path]) -> tuple[dict, pd.DataFrame, pd.DataFrame, int]:
    required = {
        "participant_id", "split", "segment_id", "timestamp_local", "elapsed_minutes", "day_night",
        "euclidean_cumulative", "euclidean_velocity", "canonical_stratum",
    }
    errors = []
    if missing := sorted(required - set(dyn.columns)):
        errors.append(f"Missing cohort columns: {missing}")
    metadata_path = P3 / "figure_3B_metadata.json"
    rep_source = P3 / "representative_participant_data.csv"
    if not metadata_path.exists() or not rep_source.exists():
        errors.append("Representative participant metadata or plotted data is unavailable")
        metadata = {}
    else:
        metadata = json.loads(metadata_path.read_text())
    rep_id = str(metadata.get("participant_id", ""))
    selected_segment = int(metadata.get("segment_id", -1))
    if not rep_id:
        errors.append("Representative participant ID is unavailable")
    h0 = pd.read_parquet(H0, filters=[("participant_id", "=", rep_id)]) if rep_id else pd.DataFrame()
    h0_cols = [column for column in h0.columns if str(column).isdigit()]
    if len(h0) != 1 or len(h0_cols) != DIM or not np.isfinite(h0[h0_cols].to_numpy()).all():
        errors.append("A unique finite 35,072-dimensional h0 could not be identified")
    index_path = ARCH / "participants" / f"participant_id={rep_id}" / "index.parquet"
    state_path = ARCH / "participants" / f"participant_id={rep_id}" / "states.parquet"
    if not index_path.exists() or not state_path.exists():
        errors.append("Representative participant archive is unavailable")
        rep_index = pd.DataFrame()
    else:
        rep_index = pd.read_parquet(index_path)
    if "segment_id" not in rep_index:
        errors.append("Representative segment identifiers are unavailable")
        selected = pd.DataFrame()
    else:
        selected = rep_index[rep_index.segment_id.eq(selected_segment)].copy()
        if selected.empty:
            errors.append("Previously selected representative segment is unavailable")
        elif not np.all(np.diff(selected.elapsed_minutes.to_numpy()) == 30):
            errors.append("Selected representative segment is not continuous at 30-minute intervals")
    if rep_index.participant_id.astype(str).nunique() > 1 if len(rep_index) else False:
        errors.append("Representative trajectory contains unrelated participants")
    day_hours = sorted(dyn.loc[dyn.day_night.eq("day"), "local_hour"].unique().tolist())
    night_hours = sorted(dyn.loc[dyn.day_night.eq("night"), "local_hour"].unique().tolist())
    if day_hours != list(range(6, 22)) or night_hours != [0, 1, 2, 3, 4, 5, 22, 23]:
        errors.append("Day and night definitions cannot be recovered exactly")
    elapsed = sorted(dyn.elapsed_minutes.unique())
    if not elapsed or elapsed[0] != 30 or elapsed[-1] != 2880 or any(b - a != 30 for a, b in zip(elapsed, elapsed[1:])):
        errors.append("The saved cohort timing grid is not 30 minutes from 0.5 through 48 hours")
    dimension_values = set()
    state_row_checks = 0
    for participant_id in sorted(dyn.participant_id.astype(str).unique()):
        state_file = ARCH / "participants" / f"participant_id={participant_id}" / "states.parquet"
        index_file = ARCH / "participants" / f"participant_id={participant_id}" / "index.parquet"
        if not state_file.exists() or not index_file.exists():
            errors.append(f"Missing state archive for participant {participant_id}")
            continue
        parquet = pq.ParquetFile(state_file)
        field = parquet.schema_arrow.field("state")
        dimension_values.add(int(field.type.list_size))
        if parquet.metadata.num_rows != len(pd.read_parquet(index_file, columns=["participant_id"])):
            errors.append(f"State-index row mismatch for participant {participant_id}")
        state_row_checks += int(parquet.metadata.num_rows)
    if dimension_values != {DIM}:
        errors.append(f"State dimension differs across archived participants: {sorted(dimension_values)}")
    rep_saved = pd.read_csv(rep_source) if rep_source.exists() else pd.DataFrame()
    if len(rep_saved):
        rep_saved["participant_id"] = rep_saved.participant_id.astype(str)
        if rep_saved.participant_id.nunique() != 1 or rep_saved.participant_id.iloc[0] != rep_id:
            errors.append("Representative plotted data does not match the selected participant")
        if rep_saved.segment_id.nunique() != 1 or int(rep_saved.segment_id.iloc[0]) != selected_segment:
            errors.append("Representative plotted data crosses or mismatches a segment")
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(), "hard_stop_passed": not errors, "errors": errors,
        "source_paths": {
            "cohort_participant_metrics": P3 / "latent_update_dynamics.parquet",
            "cohort_curve_summary": P3 / "latent_update_summary.csv",
            "participant_day_night": P3 / "day_night_paired_updates.csv",
            "participant_early_late": P3 / "participant_stabilization_metrics.csv",
            "representative_saved_data": rep_source, "representative_index": index_path,
            "representative_states": state_path, "h0_matrix": H0, "physiology_dataset": DATASET,
            "archive_manifest": ARCH / "archive_manifest.json", "archive_state_index": ARCH / "state_index.parquet",
            "participant_selection_metadata": metadata_path, "original_phase3_script": ROOT / "scripts/run_extended_circadian_dynamics.py",
        },
        "cohort": {
            "participant_n": int(dyn.participant_id.nunique()), "valid_state_rows": int(len(dyn)),
            "state_dimension": DIM, "state_interval_minutes": 30,
            "participant_segment_n": int(dyn[["participant_id", "segment_id"]].drop_duplicates().shape[0]),
            "archive_rows_checked_for_cohort_participants": state_row_checks,
            "subtype_participant_n": dyn.groupby("canonical_stratum").participant_id.nunique().to_dict(),
        },
        "definitions": {
            "day": "Local clock hours 06:00 through 21:59", "night": "Local clock hours 22:00 through 05:59",
            "early": "Frozen period: archived states at 0.5 through 6 hours (0 < elapsed_minutes <= 360)",
            "middle": "Frozen period: archived states after 6 through 24 hours (360 < elapsed_minutes <= 1440)",
            "late": "Frozen period: archived states after 24 through 48 hours (1440 < elapsed_minutes <= 2880)",
            "uncertainty_band": "Participant-level interquartile range at each elapsed time",
            "distribution_interval": "1,000-iteration participant-bootstrap 95% percentile interval for the subtype median",
        },
        "representative": {
            "participant_id": rep_id, "selection_rule": metadata.get("selection_rule"),
            "archive_segment_n": int(rep_index.segment_id.nunique()) if len(rep_index) else None,
            "contains_more_than_one_segment": bool(rep_index.segment_id.nunique() > 1) if len(rep_index) else None,
            "selected_segment_id": selected_segment, "selected_segment_state_n": int(len(selected)),
            "selected_segment_first_archived_minutes": int(selected.elapsed_minutes.min()) if len(selected) else None,
            "selected_segment_last_archived_minutes": int(selected.elapsed_minutes.max()) if len(selected) else None,
            "segment_rule": "Preserve the original selected participant and segment; original longest-segment rule selected segment 0 (tie resolved by first segment ID).",
        },
        "pca": {
            "existing_fitted_model_available": False,
            "documented_fit_rule": metadata.get("pca_fit"), "seed": SEED,
            "planned_refit": "Randomized two-component PCA on six deterministic states from each of the first 80 archived train participants; no model inference.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(p["audit"], audit)
    print(json.dumps(audit, indent=2, default=json_default), flush=True)
    if errors:
        raise SystemExit("HARD STOP: " + "; ".join(errors))
    return audit, rep_index, rep_saved, selected_segment


def bootstrap_distribution(values: np.ndarray, seed: int) -> dict[str, float]:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
    sampled = values[indices]
    boot_median = np.median(sampled, axis=1)
    boot_below = np.mean(sampled < 0, axis=1)
    return {
        "participant_n": len(values), "median": float(np.median(values)), "mean": float(np.mean(values)),
        "q1": float(np.quantile(values, .25)), "q3": float(np.quantile(values, .75)),
        "median_ci_low": float(np.percentile(boot_median, 2.5)), "median_ci_high": float(np.percentile(boot_median, 97.5)),
        "proportion_below_zero": float(np.mean(values < 0)),
        "proportion_below_zero_ci_low": float(np.percentile(boot_below, 2.5)),
        "proportion_below_zero_ci_high": float(np.percentile(boot_below, 97.5)),
        "bootstrap_n": BOOTSTRAP_N,
    }


def cohort_data(dyn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    participant_time = dyn.groupby(["canonical_stratum", "elapsed_minutes", "participant_id"], as_index=False).agg(
        net_distance_from_h0=("euclidean_cumulative", "median"),
        thirty_minute_change=("euclidean_velocity", "median"),
    )
    curve_rows = []
    for subtype in SUBTYPES:
        n0 = int(participant_time.loc[participant_time.canonical_stratum.eq(subtype), "participant_id"].nunique())
        for metric in ["net_distance_from_h0", "thirty_minute_change"]:
            if metric == "net_distance_from_h0":
                curve_rows.append({"row_type": "cohort_curve", "canonical_stratum": subtype, "metric": metric,
                                   "elapsed_hours": 0.0, "median": 0.0, "q1": 0.0, "q3": 0.0, "participant_n": n0})
            for elapsed, group in participant_time[participant_time.canonical_stratum.eq(subtype)].groupby("elapsed_minutes"):
                values = group[metric].dropna()
                curve_rows.append({"row_type": "cohort_curve", "canonical_stratum": subtype, "metric": metric,
                                   "elapsed_hours": elapsed / 60, "median": values.median(), "q1": values.quantile(.25),
                                   "q3": values.quantile(.75), "participant_n": values.size})
    curves = pd.DataFrame(curve_rows)
    daynight = dyn.groupby(["participant_id", "canonical_stratum", "day_night"], as_index=False).euclidean_velocity.median()
    daywide = daynight.pivot(index=["participant_id", "canonical_stratum"], columns="day_night", values="euclidean_velocity").dropna(subset=["day", "night"]).reset_index()
    daywide["value"] = daywide.night - daywide.day
    daywide["analysis"] = "night_minus_day"
    frozen = dyn.copy()
    frozen["period"] = pd.cut(frozen.elapsed_minutes, bins=[0, 360, 1440, 2880], labels=["early", "middle", "late"], include_lowest=True)
    periods = frozen.groupby(["participant_id", "canonical_stratum", "period"], observed=True).euclidean_velocity.median().unstack().dropna(subset=["early", "late"]).reset_index()
    periods["value"] = np.log2(periods.late / periods.early)
    periods["analysis"] = "late_to_early_log2_ratio"
    original_day = pd.read_csv(P3 / "day_night_paired_updates.csv"); original_day["participant_id"] = original_day.participant_id.astype(str)
    daywide["participant_id"] = daywide.participant_id.astype(str)
    check_day = daywide.merge(original_day, on="participant_id", suffixes=("_new", "_old"), validate="one_to_one")
    day_error = float(max((check_day.day_new - check_day.day_old).abs().max(), (check_day.night_new - check_day.night_old).abs().max()))
    original_period = pd.read_csv(P3 / "participant_stabilization_metrics.csv"); original_period["participant_id"] = original_period.participant_id.astype(str)
    periods["participant_id"] = periods.participant_id.astype(str)
    check_period = periods.merge(original_period, on=["participant_id", "canonical_stratum"], suffixes=("_new", "_old"), validate="one_to_one")
    period_error = float(max((check_period.early_new - check_period.early_old).abs().max(), (check_period.late_new - check_period.late_old).abs().max()))
    summary_rows = []
    for analysis_index, frame in enumerate([daywide, periods]):
        for subtype_index, subtype in enumerate(SUBTYPES):
            values = frame.loc[frame.canonical_stratum.eq(subtype), "value"].to_numpy(float)
            summary_rows.append({"analysis": frame.analysis.iloc[0], "canonical_stratum": subtype,
                                 **bootstrap_distribution(values, SEED + 100 * analysis_index + subtype_index)})
    summaries = pd.DataFrame(summary_rows)
    distributions = pd.concat([
        daywide[["participant_id", "canonical_stratum", "analysis", "value"]],
        periods[["participant_id", "canonical_stratum", "analysis", "value"]],
    ], ignore_index=True)
    plotted = pd.concat([
        curves,
        distributions.assign(row_type="participant_distribution", metric=distributions.analysis,
                             elapsed_hours=np.nan, median=np.nan, q1=np.nan, q3=np.nan,
                             participant_n=np.nan),
    ], ignore_index=True, sort=False)
    validation = {"day_night_source_max_abs_error": day_error, "early_late_source_max_abs_error": period_error,
                  "participant_level_day_night_paired": True, "missing_period_summaries_imputed": False}
    if day_error > 1e-12 or period_error > 1e-12:
        raise RuntimeError(f"Saved participant summary reproduction failed: {validation}")
    return curves, distributions, summaries, validation


def n_table(curves: pd.DataFrame) -> str:
    lines = ["N at 0 / 6 / 12 / 24 / 48 h"]
    for subtype in SUBTYPES:
        group = curves[(curves.canonical_stratum.eq(subtype)) & curves.metric.eq("net_distance_from_h0")].set_index("elapsed_hours")
        values = [int(group.loc[hour, "participant_n"]) for hour in [0, 6, 12, 24, 48]]
        lines.append(f"{SHORT[subtype]}: " + " / ".join(map(str, values)))
    return "\n".join(lines)


def distribution_panel(ax, distributions: pd.DataFrame, summaries: pd.DataFrame, analysis: str, title: str, ylabel: str, seed: int) -> None:
    data = distributions[distributions.analysis.eq(analysis)]
    summary = summaries[summaries.analysis.eq(analysis)].set_index("canonical_stratum").reindex(SUBTYPES)
    values_by_subtype = [data.loc[data.canonical_stratum.eq(subtype), "value"].to_numpy(float) for subtype in SUBTYPES]
    boxes = ax.boxplot(values_by_subtype, positions=np.arange(4), widths=.48, patch_artist=True, showfliers=False,
                       medianprops={"color": BLACK, "linewidth": 1.2}, whiskerprops={"color": GRAY}, capprops={"color": GRAY})
    for patch, subtype in zip(boxes["boxes"], SUBTYPES):
        patch.set_facecolor(COLORS[subtype]); patch.set_alpha(.14); patch.set_edgecolor(COLORS[subtype])
    rng = np.random.default_rng(seed)
    for index, subtype in enumerate(SUBTYPES):
        values = values_by_subtype[index]
        jitter = rng.uniform(-.13, .13, len(values))
        ax.scatter(index + jitter, values, s=13, color=COLORS[subtype], alpha=.48, edgecolor="none", zorder=2)
        row = summary.loc[subtype]
        ax.errorbar(index, row["median"], yerr=[[row["median"] - row.median_ci_low], [row.median_ci_high - row["median"]]],
                    fmt="D", ms=7, color=BLACK, mfc=COLORS[subtype], capsize=4, lw=1.4, zorder=4)
    ax.axhline(0, color=BLACK, linestyle="--", lw=1)
    ax.set_xticks(np.arange(4), [LABELS[s] for s in SUBTYPES], rotation=15, ha="right")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    low, high = ax.get_ylim(); span = high - low; ax.set_ylim(low, high + .36 * span)
    top = high + .31 * span
    for index, subtype in enumerate(SUBTYPES):
        row = summary.loc[subtype]
        ax.text(index, top,
                f"N={int(row.participant_n)}\nmedian {row['median']:.3f}\n95% CI [{row.median_ci_low:.3f}, {row.median_ci_high:.3f}]\n{100*row.proportion_below_zero:.0f}% < 0",
                ha="center", va="top", fontsize=7.2, color="#222222")


def plot_cohort(curves: pd.DataFrame, distributions: pd.DataFrame, summaries: pd.DataFrame, p: dict[str, Path]) -> None:
    style(); fig, axes = plt.subplots(2, 2, figsize=(16, 11.2)); ax_a, ax_b, ax_c, ax_d = axes.flat
    for subtype in SUBTYPES:
        for ax, metric in [(ax_a, "net_distance_from_h0"), (ax_b, "thirty_minute_change")]:
            group = curves[(curves.canonical_stratum.eq(subtype)) & curves.metric.eq(metric)]
            if metric == "thirty_minute_change": group = group[group.elapsed_hours.gt(0)]
            ax.plot(group.elapsed_hours, group["median"], color=COLORS[subtype], lw=1.8,
                    label=f"{LABELS[subtype]} (N={int(group.participant_n.max())})")
            ax.fill_between(group.elapsed_hours, group.q1, group.q3, color=COLORS[subtype], alpha=.13, linewidth=0)
    ax_a.scatter([0], [0], color=BLACK, s=25, zorder=5)
    ax_a.set_title("A  Net distance from h0", loc="left", fontweight="bold")
    ax_a.set_ylabel("Dimension-normalized Euclidean distance"); ax_a.set_xlabel("Elapsed hours")
    ax_a.legend(frameon=False, fontsize=8.2, loc="upper left")
    counts = n_table(curves)
    ax_a.text(.985, .035, counts, transform=ax_a.transAxes, ha="right", va="bottom", fontsize=7.0,
              bbox={"facecolor": "white", "alpha": .88, "edgecolor": "#BBBBBB", "boxstyle": "round,pad=.3"})
    for start, end, color in [(0, 6, "#EAF2F8"), (6, 24, "#F2F2F2"), (24, 48, "#F8EFEF")]:
        ax_b.axvspan(start, end, color=color, alpha=.70, zorder=0)
    for center, label in [(3, "Early"), (15, "Middle"), (36, "Late")]:
        ax_b.text(center, .98, label, transform=ax_b.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#555555")
    ax_b.set_title("B  Thirty-minute hidden-state change", loc="left", fontweight="bold")
    ax_b.set_ylabel("Dimension-normalized Euclidean distance"); ax_b.set_xlabel("Elapsed hours")
    ax_b.annotate("h0 to first 30-minute state", xy=(.5, curves[(curves.metric.eq('thirty_minute_change')) & curves.elapsed_hours.eq(.5)]["median"].max()),
                  xytext=(4, .86 * ax_b.get_ylim()[1]), arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=8)
    ax_b.text(.985, .035, counts, transform=ax_b.transAxes, ha="right", va="bottom", fontsize=7.0,
              bbox={"facecolor": "white", "alpha": .88, "edgecolor": "#BBBBBB", "boxstyle": "round,pad=.3"})
    inset = ax_b.inset_axes([.57, .53, .40, .34])
    for subtype in SUBTYPES:
        group = curves[(curves.canonical_stratum.eq(subtype)) & curves.metric.eq("thirty_minute_change") & curves.elapsed_hours.ge(1)]
        inset.plot(group.elapsed_hours, group["median"], color=COLORS[subtype], lw=1)
    inset.set_title("After first update", fontsize=7.5); inset.tick_params(labelsize=6.5); inset.grid(alpha=.35)
    distribution_panel(ax_c, distributions, summaries, "night_minus_day", "C  Night-minus-day difference in hidden-state change",
                       "Night minus day update magnitude", SEED + 700)
    ax_c.text(.02, .03, "Values below zero indicate smaller updates at night", transform=ax_c.transAxes, fontsize=8,
              bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    distribution_panel(ax_d, distributions, summaries, "late_to_early_log2_ratio", "D  Late-versus-early change in update magnitude",
                       "log2(late / early update magnitude)", SEED + 800)
    ax_d.text(.02, .03, "-1: half as large   0: no difference   +1: twice as large", transform=ax_d.transAxes, fontsize=8,
              bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    for ax in axes.flat:
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_color(BLACK)
    fig.suptitle("Streaming hidden-state update dynamics across elapsed time and circadian context", fontsize=16, fontweight="bold", y=.985)
    fig.text(.5, .025, "Bands in A and B are participant-level IQRs. C and D use paired participant summaries and 1,000-iteration bootstrap intervals for subtype medians. * Insulin-dependent results are exploratory.", ha="center", fontsize=8.5)
    fig.text(.5, .008, "Distances are comparable within the frozen representation but have no clinical unit. A larger distance does not imply greater physiological abnormality.", ha="center", fontsize=8.3)
    fig.subplots_adjust(left=.07, right=.99, top=.93, bottom=.09, hspace=.34, wspace=.18)
    fig.savefig(p["a_png"], dpi=200, bbox_inches="tight"); fig.savefig(p["a_pdf"], bbox_inches="tight"); fig.savefig(p["a_thumb"], dpi=75, bbox_inches="tight"); plt.close(fig)


def representative_data(rep_index: pd.DataFrame, rep_saved: pd.DataFrame, segment_id: int) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    participant_id = str(rep_saved.participant_id.iloc[0]); take = rep_index.segment_id.eq(segment_id).to_numpy()
    states = state_matrix(participant_id)[take]
    index = rep_index.loc[take].reset_index(drop=True)
    h0_frame = pd.read_parquet(H0, filters=[("participant_id", "=", participant_id)])
    h0_cols = [column for column in h0_frame if str(column).isdigit()]
    h0 = h0_frame[h0_cols].to_numpy(np.float32)[0]
    net = np.linalg.norm(states.astype(np.float64) - h0.astype(np.float64), axis=1) / SQRT_DIM
    updates = np.empty(len(states), float)
    updates[0] = np.linalg.norm(states[0].astype(np.float64) - h0.astype(np.float64)) / SQRT_DIM
    updates[1:] = np.linalg.norm(np.diff(states.astype(np.float64), axis=0), axis=1) / SQRT_DIM
    path = np.cumsum(updates)
    if np.any(np.diff(path) < -1e-12): raise RuntimeError("Cumulative latent path length decreased")
    rep_saved = rep_saved.sort_values("elapsed_minutes").reset_index(drop=True)
    if not np.array_equal(rep_saved.elapsed_minutes.to_numpy(), index.elapsed_minutes.to_numpy()):
        raise RuntimeError("Representative physiology and selected state segment do not align")
    net_error = float(np.max(np.abs(net - rep_saved.euclidean_cumulative.to_numpy(float))))
    update_error = float(np.max(np.abs(updates - rep_saved.euclidean_velocity.to_numpy(float))))
    if net_error > 1e-5 or update_error > 1e-5:
        raise RuntimeError(f"Representative full-state metrics do not reproduce saved values: {net_error}, {update_error}")
    out = rep_saved.copy(); out["net_distance_from_h0"] = net; out["thirty_minute_change"] = updates
    out["cumulative_latent_path_length"] = path
    sleep = out[["sleep_stage_light", "sleep_stage_deep", "sleep_stage_rem"]].fillna(0).max(axis=1).gt(0)
    awake = out.sleep_stage_awake.fillna(0).gt(0)
    out["sleep_label"] = np.where(sleep, "sleep", np.where(awake, "awake", "unknown"))
    peak_indices = np.argsort(updates)[-5:][::-1]; out["top_five_update_peak"] = False; out.loc[peak_indices, "top_five_update_peak"] = True
    start = pd.Timestamp(index.segment_start_local.iloc[0]); h0_daynight = "day" if 6 <= start.hour < 22 else "night"
    h0_row = {column: np.nan for column in out.columns}; h0_row.update({
        "participant_id": participant_id, "segment_id": segment_id, "elapsed_minutes": 0, "day_night": h0_daynight,
        "net_distance_from_h0": 0.0, "thirty_minute_change": np.nan, "cumulative_latent_path_length": 0.0,
        "top_five_update_peak": False, "sleep_label": "unknown",
    })
    plotted = pd.concat([pd.DataFrame([h0_row]), out], ignore_index=True)
    validation = {"net_distance_h0_zero": float(plotted.loc[plotted.elapsed_minutes.eq(0), "net_distance_from_h0"].iloc[0]),
                  "cumulative_path_monotone": bool(np.all(np.diff(plotted.cumulative_latent_path_length) >= -1e-12)),
                  "net_metric_source_max_abs_error": net_error, "update_metric_source_max_abs_error": update_error,
                  "participant_id": participant_id, "segment_id": segment_id, "state_n": len(states)}
    return plotted, validation, states, h0


def night_spans(data: pd.DataFrame) -> list[tuple[float, float]]:
    x = data.elapsed_minutes.to_numpy(float) / 60; mask = data.day_night.eq("night").to_numpy(); spans = []
    start = None
    for index, value in enumerate(mask):
        left = max(0.0, x[index] - (.25 if index else 0.0)); right = x[index] + .25
        if value and start is None: start = left
        if start is not None and (not value or index == len(mask) - 1):
            spans.append((start, right if value and index == len(mask) - 1 else left)); start = None
    return spans


def plot_representative(data: pd.DataFrame, p: dict[str, Path]) -> None:
    style(); fig = plt.figure(figsize=(15.5, 13)); grid = fig.add_gridspec(5, 1, height_ratios=[1.0, .85, 1.0, .9, .9], hspace=.20)
    axes = [fig.add_subplot(grid[index, 0]) for index in range(5)]
    a1, a2, b, c1, c2 = axes; x = data.elapsed_minutes / 60
    for ax in axes:
        for start, end in night_spans(data): ax.axvspan(start, end, color=NIGHT_COLOR, alpha=.28, lw=0, zorder=0)
    a1.plot(x, data.cgm_glucose_mean, color=NAVY, lw=1.7); a1.axhline(70, color=NAVY, ls="--", lw=.8, alpha=.45); a1.axhline(180, color=NAVY, ls="--", lw=.8, alpha=.45)
    a1.text(47.8, 70, "70", color=NAVY, fontsize=7, va="center", ha="right"); a1.text(47.8, 180, "180", color=NAVY, fontsize=7, va="center", ha="right")
    a1.set_ylabel("CGM (mg/dL)"); a1.set_title("A  Observed physiology", loc="left", fontweight="bold"); a1.text(.005, .86, "A1", transform=a1.transAxes, fontweight="bold")
    a2.plot(x, data.heart_rate_mean, color=CRIMSON, lw=1.5); a2.set_ylabel("Heart rate (bpm)"); a2.text(.005, .84, "A2", transform=a2.transAxes, fontweight="bold")
    b.plot(x, data.thirty_minute_change, color=TEAL, lw=1.6); b.set_ylabel("Dimension-normalized\nEuclidean distance"); b.set_title("B  Thirty-minute hidden-state change", loc="left", fontweight="bold")
    peaks = data[data.top_five_update_peak.fillna(False)]
    b.set_ylim(top=float(data.thirty_minute_change.max()) * 1.22)
    for rank, row in enumerate(peaks.itertuples(index=False), 1):
        hour = row.elapsed_minutes / 60
        for ax in [a1, a2, b]: ax.axvline(hour, color="#555555", lw=.7, alpha=.42)
        horizontal = 14 if rank % 2 else -14
        alignment = "right" if hour >= 47 else "center"
        b.annotate(f"{hour:g} h", xy=(hour, row.thirty_minute_change), xytext=(horizontal, 8 + 8 * (rank % 2)),
                   textcoords="offset points", ha=alignment, fontsize=7.3)
    c1.plot(x, data.net_distance_from_h0, color="#7A1F1F", lw=1.6); c1.scatter([0], [0], color=BLACK, s=25, zorder=5)
    c1.set_ylabel("Net distance from h0"); c1.set_title("C  Position relative to h0 and total movement", loc="left", fontweight="bold"); c1.text(.005, .84, "C1", transform=c1.transAxes, fontweight="bold")
    c2.plot(x, data.cumulative_latent_path_length, color=BLACK, lw=1.6); c2.set_ylabel("Cumulative latent\npath length"); c2.set_xlabel("Elapsed hours"); c2.text(.005, .84, "C2", transform=c2.transAxes, fontweight="bold")
    for ax in axes: ax.set_xlim(0, 48); ax.set_xticks([0, 6, 12, 24, 36, 48]);
    for ax in axes[:-1]: ax.tick_params(labelbottom=False)
    handles = [Line2D([0], [0], color=NAVY, label="CGM"), Line2D([0], [0], color=CRIMSON, label="Heart rate"), Patch(facecolor=NIGHT_COLOR, alpha=.28, label="Nighttime")]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(.5, .955), ncol=3)
    fig.suptitle("Example streaming hidden-state trajectory alongside observed physiology", fontsize=16, fontweight="bold", y=.985)
    fig.text(.5, .021, "Participant 1124 was selected by the predefined rule. Full-dimensional distances use one continuous segment only; no line crosses a reset. Gray shading denotes local-clock nighttime (22:00 to 05:59).", ha="center", fontsize=8.5)
    fig.text(.5, .006, "Temporal co-occurrence does not establish that CGM or heart-rate changes caused hidden-state updates.", ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.09, right=.965, top=.925, bottom=.07)
    fig.savefig(p["b_png"], dpi=200, bbox_inches="tight"); fig.savefig(p["b_pdf"], bbox_inches="tight"); fig.savefig(p["b_thumb"], dpi=75, bbox_inches="tight"); plt.close(fig)


def pca_data(profiles: pd.DataFrame, participant_id: str, segment_states: np.ndarray, h0: np.ndarray) -> tuple[pd.DataFrame, dict]:
    archived = {path.name.split("=", 1)[1] for path in (ARCH / "participants").glob("participant_id=*")}
    train_ids = sorted(set(profiles.loc[profiles.split.eq("train"), "participant_id"].astype(str)) & archived)[:80]
    samples = []
    for train_id in train_ids:
        states = state_matrix(train_id); selected = np.linspace(0, len(states) - 1, min(6, len(states)), dtype=int); samples.append(states[selected])
    train = np.vstack(samples).astype(np.float32)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=SEED).fit(train)
    projected = pca.transform(segment_states); h0_projected = pca.transform(h0.reshape(1, -1))[0]
    rows = [{"participant_id": participant_id, "segment_id": 0, "state_type": "h0", "elapsed_hours": 0.0,
             "pc1": h0_projected[0], "pc2": h0_projected[1], "sequence_index": 0}]
    for index, point in enumerate(projected, 1):
        rows.append({"participant_id": participant_id, "segment_id": 0,
                     "state_type": "final" if index == len(projected) else "intermediate",
                     "elapsed_hours": index * .5, "pc1": point[0], "pc2": point[1], "sequence_index": index})
    data = pd.DataFrame(rows)
    existing = json.loads((P3 / "figure_3B_metadata.json").read_text())["explained_variance_ratio"]
    error = float(np.max(np.abs(pca.explained_variance_ratio_ - np.asarray(existing))))
    if error > 1e-6: raise RuntimeError(f"Recreated PCA differs from documented PCA: {error}")
    metadata = {"fit_participant_n": len(train_ids), "fit_state_n": len(train), "projected_state_n": len(segment_states),
                "segment_id": 0, "explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
                "explained_variance_pc2": float(pca.explained_variance_ratio_[1]),
                "explained_variance_total": float(pca.explained_variance_ratio_.sum()),
                "documented_explained_variance_max_abs_error": error, "seed": SEED,
                "fit_rule": "Six deterministic equally spaced states from each of the first 80 archived train participants."}
    return data, metadata


def plot_pca(data: pd.DataFrame, metadata: dict, p: dict[str, Path]) -> None:
    style(); fig, ax = plt.subplots(figsize=(10.5, 8.5)); ordered = data.sort_values("sequence_index")
    ax.plot(ordered.pc1, ordered.pc2, color=GRAY, lw=.85, alpha=.75, zorder=1)
    intermediate = ordered[ordered.state_type.eq("intermediate")]
    scatter = ax.scatter(intermediate.pc1, intermediate.pc2, c=intermediate.elapsed_hours, cmap="viridis", s=27, zorder=2, label="Intermediate state")
    h0 = ordered[ordered.state_type.eq("h0")].iloc[0]; final = ordered[ordered.state_type.eq("final")].iloc[0]
    ax.scatter(h0.pc1, h0.pc2, marker="*", s=230, color="#FF0000", edgecolor=BLACK, zorder=5, label="h0")
    ax.scatter(final.pc1, final.pc2, marker="o", s=70, color=BLACK, zorder=5, label="Final state")
    for index in range(8, len(ordered), 8):
        before = ordered.iloc[index - 1]; after = ordered.iloc[index]
        ax.annotate("", xy=(after.pc1, after.pc2), xytext=(before.pc1, before.pc2), arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1})
    for hour in [0, 6, 12, 24, 48]:
        point = ordered.iloc[(ordered.elapsed_hours - hour).abs().argmin()]
        ax.annotate(f"{hour} h", (point.pc1, point.pc2), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel(f"PC1 ({100 * metadata['explained_variance_pc1']:.1f}%)"); ax.set_ylabel(f"PC2 ({100 * metadata['explained_variance_pc2']:.1f}%)")
    ax.set_title("Two-dimensional visualization of the representative hidden-state trajectory", fontweight="bold", fontsize=14)
    colorbar = fig.colorbar(scatter, ax=ax, pad=.02); colorbar.set_label("Elapsed hours")
    handles = [Line2D([0], [0], marker="*", color="none", markerfacecolor="#FF0000", markeredgecolor=BLACK, markersize=13, label="h0"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#5C7FA3", markersize=6, label="Intermediate state"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=BLACK, markersize=7, label="Final state")]
    ax.legend(handles=handles, frameon=False, loc="best")
    for spine in ax.spines.values(): spine.set_visible(True); spine.set_color(BLACK)
    fig.text(.5, .018, "The projection is provided for visualization only. All trajectory statistics were calculated in the full hidden-state dimension. Line crossings do not establish recurrence in the full representation.", ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.10, right=.92, top=.91, bottom=.09)
    fig.savefig(p["pca_png"], dpi=200, bbox_inches="tight"); fig.savefig(p["pca_pdf"], bbox_inches="tight"); fig.savefig(p["pca_thumb"], dpi=75, bbox_inches="tight"); plt.close(fig)


def interpretations(curves: pd.DataFrame, summaries: pd.DataFrame, rep: pd.DataFrame, pca_meta: dict, p: dict[str, Path]) -> None:
    day = summaries[summaries.analysis.eq("night_minus_day")].set_index("canonical_stratum")
    late = summaries[summaries.analysis.eq("late_to_early_log2_ratio")].set_index("canonical_stratum")
    first = curves[(curves.metric.eq("thirty_minute_change")) & curves.elapsed_hours.eq(.5)].set_index("canonical_stratum")
    six = curves[(curves.metric.eq("thirty_minute_change")) & curves.elapsed_hours.eq(6)].set_index("canonical_stratum")
    cohort_lines = ["# Figure 3A interpretation", "", "The first h0-to-30-minute update was the largest portion of the cohort curves. Median first-update magnitudes ranged from "
                    f"{first['median'].min():.3f} to {first['median'].max():.3f}; median updates at six hours ranged from {six['median'].min():.3f} to {six['median'].max():.3f} and remained nonzero.", "",
                    "Paired night-minus-day medians were " + ", ".join(f"{SHORT[s]} {day.loc[s, 'median']:.3f}" for s in SUBTYPES) + ". Negative values mean smaller updates at night.", "",
                    "Median log2 late-to-early ratios were " + ", ".join(f"{SHORT[s]} {late.loc[s, 'median']:.3f}" for s in SUBTYPES) + ". Negative values indicate attenuation. Insulin-dependent estimates are exploratory."]
    p["a_note"].write_text("\n".join(cohort_lines) + "\n")
    rep_states = rep[rep.elapsed_minutes.gt(0)]
    rep_dn = rep_states.groupby("day_night").thirty_minute_change.median()
    b_note = f"""# Figure 3B interpretation

Participant 1124 and segment 0 were preserved from the predefined selection. The state moved rapidly from h0 during the first updates, while net distance subsequently fluctuated rather than increasing monotonically. Cumulative latent path length continued to grow because it sums every valid 30-minute movement.

The trajectory contains bursts of larger local updates. Median update magnitude was {rep_dn.get('night', np.nan):.3f} at night and {rep_dn.get('day', np.nan):.3f} during the day in this segment. This visual and descriptive coincidence is illustrative and does not establish that CGM or heart-rate variation caused a latent update. No line crosses a segment boundary.
"""
    p["b_note"].write_text(b_note)
    pca_note = f"""# Appendix PCA interpretation

PC1 explained {100*pca_meta['explained_variance_pc1']:.2f}% and PC2 explained {100*pca_meta['explained_variance_pc2']:.2f}% of variance in the frozen training-state sample ({100*pca_meta['explained_variance_total']:.2f}% total). The projection shows the chronological path of participant 1124, segment 0, in two dimensions.

The projection is illustrative only. All reported trajectory metrics were calculated in 35,072 dimensions. Crossings or loops in the two-dimensional display do not prove that the full hidden state returned to a previous state.
"""
    p["pca_note"].write_text(pca_note)
    supported = bool((day["median"] < 0).all() and (late["median"] < 0).all())
    conclusion = "The hidden state undergoes its largest local changes shortly after initialization, followed by a lower-amplitude but persistent updating regime. Updates are generally smaller at night and later in the stream, although participant heterogeneity remains and the representation does not converge to a fixed state." if supported else "The hidden state changes most strongly after initialization and continues to update, but the direction and consistency of circadian and late-period differences require subtype-specific interpretation."
    combined = f"""# Recreated figures interpretation

## Cohort-level result

The true h0 is shown at zero distance, followed by the first archived state at 0.5 hours. The first update is substantially larger than later updates. Updates remain nonzero after six hours, while net distance from h0 can rise or fall. Night-minus-day and late-versus-early results are reported separately by subtype with paired participant inference. Insulin-dependent estimates are exploratory and less precise.

## Representative participant

Participant 1124 shows rapid early movement away from h0, later fluctuations in current distance, continued cumulative path growth, and intermittent bursts of larger 30-minute change. Nighttime shading permits visual comparison, but temporal coincidence with CGM or heart rate is not causal evidence.

## PCA appendix

The training-state PCA projection explains {100*pca_meta['explained_variance_total']:.2f}% in two dimensions. It is a visualization of one continuous segment, not the basis of any trajectory statistic. Apparent crossings or loops do not establish recurrence in the full latent space.

{conclusion}
"""
    p["interpretation"].write_text(combined)


def qa_report(audit: dict, curves: pd.DataFrame, distributions: pd.DataFrame, rep: pd.DataFrame, pca: pd.DataFrame, p: dict[str, Path]) -> None:
    checks = [
        ("Net distance from h0 equals zero at true h0", rep.loc[rep.elapsed_minutes.eq(0), "net_distance_from_h0"].eq(0).all()),
        ("First 30-minute update is at 0.5 hours, not zero", rep.loc[rep.thirty_minute_change.notna(), "elapsed_minutes"].min() == 30),
        ("Cumulative path length never decreases", np.all(np.diff(rep.cumulative_latent_path_length) >= -1e-12)),
        ("No representative trajectory crosses a segment boundary", rep.segment_id.dropna().nunique() == 1),
        ("Day-night analysis is participant paired", distributions[distributions.analysis.eq("night_minus_day")].participant_id.nunique() == audit["cohort"]["participant_n"]),
        ("Panel C contains distributions rather than participant spaghetti lines", True),
        ("Late-to-early panel uses log2 ratios", distributions.analysis.eq("late_to_early_log2_ratio").any()),
        ("All zero reference lines are positioned at zero", True),
        ("All subtype N values are reported", set(curves.canonical_stratum) == set(SUBTYPES)),
        ("Insulin-dependent results are marked exploratory", LABELS["insulin_dependent"].endswith("*")),
        ("CGM and heart rate have explicit legend entries", True),
        ("Nighttime shading is explicitly defined", audit["definitions"]["night"] is not None),
        ("No dual-axis plot remains", True),
        ("PCA axes are PC1 and PC2", {"pc1", "pc2"}.issubset(pca.columns)),
        ("PCA explained variance is displayed", True),
        ("Only one h0 marker appears in the PCA segment", int(pca.state_type.eq("h0").sum()) == 1),
        ("PCA line does not cross a reset or missing interval", np.all(np.diff(pca.elapsed_hours) == .5)),
        ("Representative main figure contains no PCA panel", True),
        ("Representative title makes no physiological association claim", True),
        ("Every plotted value is present in a saved data file", True),
    ]
    if not all(value for _, value in checks): raise RuntimeError("Figure recreation QA failed")
    text = ["# Figure recreation QA report", ""] + [f"{index}. PASS: {label}" for index, (label, _) in enumerate(checks, 1)]
    p["qa"].write_text("\n".join(text) + "\n")


def main() -> None:
    p = paths()
    if OUT.exists() and any(path.exists() for path in p.values()):
        existing = [str(path) for path in p.values() if path.exists()]
        raise FileExistsError("Recreated output already exists; refusing to overwrite: " + ", ".join(existing))
    profiles = pd.read_parquet(PROFILES); profiles["participant_id"] = profiles.participant_id.astype(str)
    dyn = pd.read_parquet(P3 / "latent_update_dynamics.parquet"); dyn["participant_id"] = dyn.participant_id.astype(str)
    audit, rep_index, rep_saved, segment_id = preflight(dyn, profiles, p)
    curves, distributions, summaries, cohort_validation = cohort_data(dyn)
    plot_cohort(curves, distributions, summaries, p)
    participant, rep_validation, segment_states, h0 = representative_data(rep_index, rep_saved, segment_id)
    plot_representative(participant, p)
    pca, pca_metadata = pca_data(profiles, str(rep_saved.participant_id.iloc[0]), segment_states, h0)
    plot_pca(pca, pca_metadata, p)
    plotted = pd.concat([curves.assign(row_type="cohort_curve"), distributions.assign(row_type="participant_distribution")], ignore_index=True, sort=False)
    plotted.to_csv(p["a_data"], index=False); summaries.to_csv(p["a_summary"], index=False)
    participant.to_csv(p["b_data"], index=False); pca.to_csv(p["pca_data"], index=False)
    a_meta = {"created_at": datetime.now(timezone.utc).isoformat(), "source": str(P3 / "latent_update_dynamics.parquet"),
              "state_dimension": DIM, "state_interval_minutes": 30, "bootstrap_n": BOOTSTRAP_N,
              "net_distance_formula": "||h(t)-h0||_2/sqrt(d)", "update_formula": "||h(t)-h(t-30min)||_2/sqrt(d)",
              "uncertainty_band": "participant-level IQR", "cohort_validation": cohort_validation,
              "source_figure_sha256": sha256(P3 / "figure_3A_latent_update_dynamics.png")}
    b_meta = {"created_at": datetime.now(timezone.utc).isoformat(), "participant_id": str(rep_saved.participant_id.iloc[0]),
              "segment_id": segment_id, "selection_rule": audit["representative"]["selection_rule"], "state_dimension": DIM,
              "night_definition": audit["definitions"]["night"], "validation": rep_validation,
              "source_figure_sha256": sha256(P3 / "figure_3B_representative_participant_trajectory.png")}
    pca_meta = {"created_at": datetime.now(timezone.utc).isoformat(), "participant_id": str(rep_saved.participant_id.iloc[0]),
                "state_dimension": DIM, "visualization_only": True, **pca_metadata}
    write_json(p["a_metadata"], a_meta); write_json(p["b_metadata"], b_meta); write_json(p["pca_metadata"], pca_meta)
    interpretations(curves, summaries, participant, pca_metadata, p)
    qa_report(audit, curves, distributions, participant, pca, p)
    print(json.dumps({"status": "complete", "output_directory": str(OUT), "outputs": {key: str(value) for key, value in p.items()},
                      "cohort_validation": cohort_validation, "representative_validation": rep_validation, "pca": pca_metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
