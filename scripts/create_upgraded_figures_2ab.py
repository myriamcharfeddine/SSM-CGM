"""Create non-overwriting upgraded circadian-neighborhood Figures 2A and 2B."""
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
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_extended_circadian_dynamics as phase23  # noqa: E402

EXT = ROOT / "outputs/static_phenotype_trajectory_stratified_v2/extended_clinical_latent_dynamics_v1"
OUT = EXT / "02_circadian_matched_reorganization"
PROFILES = EXT / "01_cluster_metabolic_profiles/participant_frozen_cluster_profiles.parquet"
CACHE = EXT / "cache"
OLD_UNMATCHED = ROOT / "outputs/static_phenotype_trajectory_stratified_v2/tables/phase3_three_space_preservation.csv"

SEED = 42
BOOTSTRAP_N = 1000
PERMUTATION_N = 1000
HOURS = [6, 12, 24, 48]
SUBTYPES = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
SUBTYPE_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes": "Pre-diabetes",
    "t2d_oral_non_insulin": "T2D oral\nnon-insulin",
    "insulin_dependent": "Insulin-dependent*",
}
METRICS = ["clinical_to_h0_jaccard", "clinical_to_ht_jaccard", "h0_to_ht_jaccard"]
METRIC_LABELS = {
    "clinical_to_h0_jaccard": "Clinical to h0",
    "clinical_to_ht_jaccard": "Clinical to ht",
    "h0_to_ht_jaccard": "h0 to ht",
}
COLORS = {
    "clinical_to_h0_jaccard": "#17365D",
    "clinical_to_ht_jaccard": "#159D9B",
    "h0_to_ht_jaccard": "#7C8794",
}
SPACE_LABELS = {"clinical": "Clinical", "h0": "h0", "ht": "ht"}
SPACE_COLORS = {"clinical": "#17365D", "h0": "#159D9B", "ht": "#7C8794"}
EQUIVALENCE_MARGIN_PER10 = 0.5


def output_paths() -> dict[str, Path]:
    stem_a = OUT / "figure_2A_upgraded_matched_unmatched_reorganization"
    stem_b = OUT / "figure_2B_upgraded_day_night_reorganization"
    return {
        "a_png": stem_a.with_suffix(".png"),
        "a_pdf": stem_a.with_suffix(".pdf"),
        "a_thumb": OUT / f"{stem_a.name}_thumbnail.png",
        "b_png": stem_b.with_suffix(".png"),
        "b_pdf": stem_b.with_suffix(".pdf"),
        "b_thumb": OUT / f"{stem_b.name}_thumbnail.png",
        "participant_overlap": OUT / "figure_2AB_upgraded_participant_overlap.parquet",
        "a_data": OUT / "figure_2A_upgraded_plotted_data.csv",
        "b_data": OUT / "figure_2B_upgraded_plotted_data.csv",
        "coverage": OUT / "figure_2AB_upgraded_coverage.csv",
        "metadata": OUT / "figure_2AB_upgraded_metadata.json",
        "note": OUT / "figure_2AB_upgraded_metric_note.md",
    }


def refuse_overwrite(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Upgraded Figure 2 outputs already exist; refusing to overwrite: " + ", ".join(existing))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_fraction_from_jaccard(jaccard: float) -> float:
    """For two equal-size neighbor sets, convert Jaccard to intersection/k."""
    return 2.0 * jaccard / (1.0 + jaccard) if np.isfinite(jaccard) else np.nan


def permutation_expected_retention(candidate_n: int, k: int, seed: list[int]) -> float:
    """Permutation null for overlap with one size-k set held fixed."""
    if candidate_n < 1 or k < 1 or k > candidate_n:
        return np.nan
    rng = np.random.default_rng(np.random.SeedSequence([SEED, *seed]))
    intersections = rng.hypergeometric(k, candidate_n - k, k, size=PERMUTATION_N)
    return float(intersections.mean() / k)


def enrich_overlap_rows(data: pd.DataFrame, condition: str, seed_code: int) -> pd.DataFrame:
    rows = []
    for row_index, row in enumerate(data.itertuples(index=False)):
        for metric_index, metric in enumerate(METRICS):
            observed = retained_fraction_from_jaccard(float(getattr(row, metric)))
            expected = permutation_expected_retention(
                int(row.candidate_pool_n),
                int(row.effective_k),
                [seed_code, int(row.hour), metric_index, row_index],
            )
            adjusted = (observed - expected) / (1.0 - expected) if expected < 1.0 else np.nan
            rows.append({
                "condition": condition,
                "canonical_stratum": row.canonical_stratum,
                "hour": int(row.hour),
                "clock_bin": str(row.clock_bin),
                "participant_id": str(row.participant_id),
                "metric": metric,
                "candidate_pool_n": int(row.candidate_pool_n),
                "effective_k": int(row.effective_k),
                "jaccard": float(getattr(row, metric)),
                "retained_fraction": observed,
                "retained_per10": 10.0 * observed,
                "permutation_expected_retention": expected,
                "adjusted_overlap": adjusted,
                "permutation_n": PERMUTATION_N,
            })
    return pd.DataFrame(rows)


def reconstruct_unmatched(profiles: pd.DataFrame) -> pd.DataFrame:
    """All-clock graphs at each elapsed hour, using the same cached states and test participants."""
    clinical, h0_matrix, h0_map = phase23.clinical_and_h0(profiles)
    k_map = phase23.original_k(profiles)
    profile_index = profiles.set_index("participant_id")
    rows = []
    for hour in HOURS:
        archive = np.load(CACHE / f"clock_states_hour{hour:02d}.npz", allow_pickle=False)
        states = archive["state"]
        participant_ids = archive["participant_id"].astype(str)
        split = archive["split"].astype(str)
        for subtype in SUBTYPES:
            allowed = np.array([
                participant_id in profile_index.index
                and profile_index.at[participant_id, "canonical_stratum"] == subtype
                and split[index] == "test"
                for index, participant_id in enumerate(participant_ids)
            ])
            selected = np.flatnonzero(allowed)
            grouped: dict[str, list[int]] = {}
            for index in selected:
                grouped.setdefault(participant_ids[index], []).append(index)
            ids = sorted(grouped)
            if len(ids) < 2:
                continue
            latent_t = np.stack([states[grouped[pid]].mean(axis=0, dtype=np.float32) for pid in ids])
            clinical_x = np.stack([clinical[subtype][pid] for pid in ids])
            latent_h0 = np.stack([h0_matrix[h0_map[pid]] for pid in ids])
            effective_k = min(k_map[subtype], len(ids) - 1)
            _, nc = phase23.nearest(clinical_x, effective_k, "euclidean")
            _, n0 = phase23.nearest(latent_h0, effective_k, "cosine")
            _, nt = phase23.nearest(latent_t, effective_k, "cosine")
            comparisons = {
                "clinical_to_h0_jaccard": (nc, n0),
                "clinical_to_ht_jaccard": (nc, nt),
                "h0_to_ht_jaccard": (n0, nt),
            }
            for anchor, participant_id in enumerate(ids):
                base = {
                    "canonical_stratum": subtype,
                    "hour": hour,
                    "clock_bin": "all_clock",
                    "participant_id": participant_id,
                    "candidate_pool_n": len(ids) - 1,
                    "effective_k": effective_k,
                }
                for metric, (first, second) in comparisons.items():
                    overlap = len(set(first[anchor]) & set(second[anchor]))
                    union = len(set(first[anchor]) | set(second[anchor]))
                    base[metric] = overlap / union if union else np.nan
                rows.append(base)
        print(f"Unmatched all-clock graphs complete through hour {hour}", flush=True)
        del archive, states
    return pd.DataFrame(rows)


def validate_unmatched_reconstruction(unmatched: pd.DataFrame, matched_metrics: pd.DataFrame) -> dict[str, object]:
    """Validate same-hour all-clock graphs without conflating them with the old 0–6 h endpoint."""
    expected_n = matched_metrics.groupby("canonical_stratum").participant_id.nunique()
    errors = []
    for subtype in SUBTYPES:
        group = unmatched[unmatched.canonical_stratum.eq(subtype)]
        for hour in HOURS:
            hour_group = group[group.hour.eq(hour)]
            if hour_group.participant_id.nunique() != int(expected_n[subtype]):
                errors.append(f"{subtype} h{hour}: participant count mismatch")
            if not (hour_group.candidate_pool_n == int(expected_n[subtype]) - 1).all():
                errors.append(f"{subtype} h{hour}: candidate-pool mismatch")
        clinical_h0 = group.pivot(index="participant_id", columns="hour", values="clinical_to_h0_jaccard")
        if float((clinical_h0.max(axis=1) - clinical_h0.min(axis=1)).max()) > 1e-12:
            errors.append(f"{subtype}: clinical-to-h0 graph changed across elapsed hours")
    values = unmatched[METRICS].to_numpy(float)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        errors.append("Jaccard values are nonfinite or outside [0, 1]")
    if errors:
        raise RuntimeError("All-clock reconstruction validation failed: " + "; ".join(errors))
    return {
        "status": "passed",
        "validated_rows": int(len(unmatched)),
        "validated_hours": HOURS,
        "participant_counts": {subtype: int(expected_n[subtype]) for subtype in SUBTYPES},
        "candidate_pool_equals_subtype_test_n_minus_one": True,
        "clinical_to_h0_invariant_across_hours": True,
        "jaccard_bounds_valid": True,
        "old_frozen_summary_not_used_for_validation": (
            "Its ht is an overnight endpoint summary using 0–6 h anchors, not a same-hour h6/h12/h24/h48 state."
        ),
    }

def paired_summary(data: pd.DataFrame, value: str, conditions: list[str], seed_code: int) -> pd.DataFrame:
    participant = data.groupby(
        ["canonical_stratum", "metric", "participant_id", "condition"], as_index=False
    )[value].mean()
    rows = []
    for subtype_index, subtype in enumerate(SUBTYPES):
        for metric_index, metric in enumerate(METRICS):
            group = participant[(participant.canonical_stratum.eq(subtype)) & participant.metric.eq(metric)]
            wide = group.pivot(index="participant_id", columns="condition", values=value).dropna(subset=conditions)
            values = wide[conditions].to_numpy(float)
            if not len(values):
                continue
            rng = np.random.default_rng(np.random.SeedSequence([SEED, seed_code, subtype_index, metric_index]))
            indices = rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
            boot = values[indices].mean(axis=1)
            for condition_index, condition in enumerate(conditions):
                rows.append({
                    "measure": value,
                    "canonical_stratum": subtype,
                    "metric": metric,
                    "condition": condition,
                    "estimate": float(values[:, condition_index].mean()),
                    "ci_low": float(np.percentile(boot[:, condition_index], 2.5)),
                    "ci_high": float(np.percentile(boot[:, condition_index], 97.5)),
                    "paired_participant_n": int(len(values)),
                    "bootstrap_n": BOOTSTRAP_N,
                })
            difference = values[:, 1] - values[:, 0]
            boot_difference = boot[:, 1] - boot[:, 0]
            rows.append({
                "measure": value,
                "canonical_stratum": subtype,
                "metric": metric,
                "condition": f"{conditions[1]}_minus_{conditions[0]}",
                "estimate": float(difference.mean()),
                "ci_low": float(np.percentile(boot_difference, 2.5)),
                "ci_high": float(np.percentile(boot_difference, 97.5)),
                "paired_participant_n": int(len(values)),
                "bootstrap_n": BOOTSTRAP_N,
            })
    return pd.DataFrame(rows)


def purity_summary(purity: pd.DataFrame) -> pd.DataFrame:
    participant = purity.groupby(["canonical_stratum", "space", "participant_id"], as_index=False).agg(
        observed=("observed_purity", "mean"),
        expected=("permutation_expected_purity", "mean"),
        adjusted=("adjusted_purity", "mean"),
    )
    rows = []
    for subtype_index, subtype in enumerate(SUBTYPES):
        for space_index, space in enumerate(["clinical", "h0", "ht"]):
            group = participant[(participant.canonical_stratum.eq(subtype)) & participant.space.eq(space)].dropna()
            values = group[["observed", "expected", "adjusted"]].to_numpy(float)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 500, subtype_index, space_index]))
            indices = rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
            boot = values[indices].mean(axis=1)
            for measure_index, measure in enumerate(["observed", "expected", "adjusted"]):
                rows.append({
                    "measure": f"purity_{measure}",
                    "canonical_stratum": subtype,
                    "metric": space,
                    "condition": measure,
                    "estimate": float(values[:, measure_index].mean()),
                    "ci_low": float(np.percentile(boot[:, measure_index], 2.5)),
                    "ci_high": float(np.percentile(boot[:, measure_index], 97.5)),
                    "paired_participant_n": int(len(values)),
                    "bootstrap_n": BOOTSTRAP_N,
                })
    return pd.DataFrame(rows)


def coverage_table(metrics: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[metrics.scenario.eq("primary_test_2h")]
    daynight = metrics[metrics.scenario.eq("primary_test_day_night")]
    total = profiles[profiles.split.eq("test")].groupby("canonical_stratum").participant_id.nunique()
    rows = []
    for subtype in SUBTYPES:
        matched = primary[primary.canonical_stratum.eq(subtype)]
        base = {"canonical_stratum": subtype, "test_participant_n": int(total[subtype])}
        base.update({
            "matched_participant_n": int(matched.participant_id.nunique()),
            "matched_anchor_n": int(len(matched)),
            "matched_pool_median": float(matched.candidate_pool_n.median()),
            "matched_pool_min": int(matched.candidate_pool_n.min()),
            "matched_pool_max": int(matched.candidate_pool_n.max()),
        })
        for period in ["day", "night"]:
            group = daynight[(daynight.canonical_stratum.eq(subtype)) & daynight.clock_bin.eq(period)]
            base[f"{period}_participant_n"] = int(group.participant_id.nunique())
            base[f"{period}_anchor_n"] = int(len(group))
            base[f"{period}_pool_median"] = float(group.candidate_pool_n.median())
            base[f"{period}_pool_min"] = int(group.candidate_pool_n.min())
            base[f"{period}_pool_max"] = int(group.candidate_pool_n.max())
        paired = daynight.groupby(["canonical_stratum", "participant_id"]).clock_bin.nunique()
        base["day_night_paired_n"] = int((paired.loc[subtype] == 2).sum()) if subtype in paired.index.get_level_values(0) else 0
        rows.append(base)
    return pd.DataFrame(rows)


def classify_day_night(row: pd.Series) -> str:
    if row.ci_low >= -EQUIVALENCE_MARGIN_PER10 and row.ci_high <= EQUIVALENCE_MARGIN_PER10:
        return "no meaningful difference"
    if row.ci_low > EQUIVALENCE_MARGIN_PER10:
        return "reliably higher at night"
    if row.ci_high < -EQUIVALENCE_MARGIN_PER10:
        return "reliably lower at night"
    return "uncertain"


def style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "sans-serif", "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black", "axes.spines.top": True, "axes.spines.right": True,
        "grid.color": "#D9D9D9", "pdf.fonttype": 42,
    })


def condition_panel(ax, data: pd.DataFrame, measure: str, conditions: list[str], ylabel: str,
                    title: str, ylim: tuple[float, float] | None = None) -> None:
    subset = data[data.measure.eq(measure)]
    x = np.arange(len(SUBTYPES))
    offsets = [-0.22, 0.0, 0.22]
    for metric, offset in zip(METRICS, offsets):
        group = subset[subset.metric.eq(metric)]
        estimates = []
        for condition_index, condition in enumerate(conditions):
            points = group[group.condition.eq(condition)].set_index("canonical_stratum").reindex(SUBTYPES)
            position = x + offset + (-0.035 if condition_index == 0 else 0.035)
            face = "white" if condition_index == 0 else COLORS[metric]
            ax.errorbar(position, points.estimate,
                        yerr=[points.estimate - points.ci_low, points.ci_high - points.estimate],
                        fmt="o", ms=6.5, mfc=face, mec=COLORS[metric], mew=1.4,
                        ecolor=COLORS[metric], elinewidth=1.1, capsize=2.5, zorder=3)
            estimates.append(points.estimate.to_numpy(float))
        for position, first, second in zip(x + offset, estimates[0], estimates[1]):
            ax.plot([position - 0.035, position + 0.035], [first, second], color=COLORS[metric], lw=1.2, zorder=2)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [SUBTYPE_LABELS[subtype] for subtype in SUBTYPES])
    if ylim:
        ax.set_ylim(*ylim)


def add_metric_legend(ax, first_label: str, second_label: str) -> None:
    metric_handles = [Line2D([0], [0], marker="o", color=COLORS[m], lw=1.5,
                             label=METRIC_LABELS[m], markerfacecolor=COLORS[m]) for m in METRICS]
    condition_handles = [
        Line2D([0], [0], marker="o", color="#222222", lw=0, markerfacecolor="white", label=first_label),
        Line2D([0], [0], marker="o", color="#222222", lw=0, markerfacecolor="#222222", label=second_label),
    ]
    first = ax.legend(handles=metric_handles, frameon=False, fontsize=8.3, loc="upper left")
    ax.add_artist(first)
    ax.legend(handles=condition_handles, frameon=False, fontsize=8.3, loc="upper right")


def coverage_lines(coverage: pd.DataFrame, daynight: bool) -> list[str]:
    lines = []
    for row in coverage.itertuples(index=False):
        label = SUBTYPE_LABELS[row.canonical_stratum].replace("\n", " ")
        if daynight:
            lines.append(
                f"{label}: test N={row.test_participant_n}; day n={row.day_participant_n}, anchors={row.day_anchor_n}; "
                f"night n={row.night_participant_n}, anchors={row.night_anchor_n}; paired n={row.day_night_paired_n}; "
                f"pool median [range] day {row.day_pool_median:.0f} [{row.day_pool_min}-{row.day_pool_max}], "
                f"night {row.night_pool_median:.0f} [{row.night_pool_min}-{row.night_pool_max}]"
            )
        else:
            lines.append(
                f"{label}: paired n={row.matched_participant_n}; circadian anchors={row.matched_anchor_n}; "
                f"matched pool median [range]={row.matched_pool_median:.0f} [{row.matched_pool_min}-{row.matched_pool_max}]"
            )
    return lines


def plot_figure_a(summary: pd.DataFrame, purity: pd.DataFrame, coverage: pd.DataFrame,
                  paths: dict[str, Path]) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.6))
    condition_panel(axes[0, 0], summary, "retained_per10", ["unmatched", "circadian_matched"],
                    "Equivalent shared neighbors per 10", "A  Same participants: all-clock versus circadian-matched", (0, 10))
    add_metric_legend(axes[0, 0], "All-clock", "Circadian-matched")
    condition_panel(axes[0, 1], summary, "adjusted_overlap", ["unmatched", "circadian_matched"],
                    "Permutation-adjusted overlap", "B  Preservation above the random-overlap null")
    axes[0, 1].axhline(0, color="black", linestyle="--", lw=0.9)

    x = np.arange(len(SUBTYPES)); offsets = [-0.22, 0.0, 0.22]
    for space, offset in zip(["clinical", "h0", "ht"], offsets):
        group = purity[purity.metric.eq(space)]
        expected = group[group.condition.eq("expected")].set_index("canonical_stratum").reindex(SUBTYPES)
        observed = group[group.condition.eq("observed")].set_index("canonical_stratum").reindex(SUBTYPES)
        for index, subtype_x in enumerate(x + offset):
            axes[1, 0].plot([subtype_x - 0.035, subtype_x + 0.035],
                            [expected.estimate.iloc[index], observed.estimate.iloc[index]],
                            color=SPACE_COLORS[space], lw=1.2)
        for condition, points, shift, marker, face in [
            ("expected", expected, -0.035, "D", "white"), ("observed", observed, 0.035, "o", SPACE_COLORS[space])
        ]:
            axes[1, 0].errorbar(x + offset + shift, points.estimate,
                                yerr=[points.estimate - points.ci_low, points.ci_high - points.estimate],
                                fmt=marker, ms=6.2, mfc=face, mec=SPACE_COLORS[space], mew=1.3,
                                ecolor=SPACE_COLORS[space], capsize=2.5, lw=1.0)
    axes[1, 0].set_title("C  Observed and permutation-expected fixed-label purity", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Neighbor purity")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xticks(x, [SUBTYPE_LABELS[s] for s in SUBTYPES])
    space_handles = [Line2D([0], [0], marker="o", color=SPACE_COLORS[s], label=SPACE_LABELS[s]) for s in ["clinical", "h0", "ht"]]
    status_handles = [Line2D([0], [0], marker="D", color="#222", lw=0, mfc="white", label="Permutation expected"),
                      Line2D([0], [0], marker="o", color="#222", lw=0, mfc="#222", label="Observed")]
    first = axes[1, 0].legend(handles=space_handles, frameon=False, fontsize=8.3, loc="upper left")
    axes[1, 0].add_artist(first); axes[1, 0].legend(handles=status_handles, frameon=False, fontsize=8.3, loc="upper right")

    adjusted = purity[purity.condition.eq("adjusted")]
    for space, offset in zip(["clinical", "h0", "ht"], offsets):
        points = adjusted[adjusted.metric.eq(space)].set_index("canonical_stratum").reindex(SUBTYPES)
        axes[1, 1].errorbar(x + offset, points.estimate,
                            yerr=[points.estimate - points.ci_low, points.ci_high - points.estimate],
                            fmt="o", ms=7, mfc=SPACE_COLORS[space], mec=SPACE_COLORS[space],
                            ecolor=SPACE_COLORS[space], capsize=3, lw=1.2, label=SPACE_LABELS[space])
    axes[1, 1].axhline(0, color="black", linestyle="--", lw=0.9)
    axes[1, 1].set_title("D  Purity enrichment above chance", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("Adjusted purity")
    axes[1, 1].set_xticks(x, [SUBTYPE_LABELS[s] for s in SUBTYPES])
    axes[1, 1].legend(frameon=False, fontsize=8.5)
    for ax in axes.flat:
        ax.tick_params(axis="x", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("black")
    fig.suptitle("Circadian matching: intuitive preservation, random-overlap control, and adjusted purity",
                 fontsize=15.5, fontweight="bold", y=0.985)
    sample_lines = coverage_lines(coverage, daynight=False)
    fig.text(0.5, 0.075, "Coverage — " + "  |  ".join(sample_lines[:2]), ha="center", fontsize=8.1)
    fig.text(0.5, 0.055, "  |  ".join(sample_lines[2:]), ha="center", fontsize=8.1)
    fig.text(0.5, 0.030,
             r"Shared per 10 $=10|N_A\cap N_B|/k=20J/(1+J)$. Adjusted overlap $=(R_{obs}-R_{perm})/(1-R_{perm})$. "
             r"Adjusted purity $=(P_{obs}-P_{perm})/(1-P_{perm})$.", ha="center", fontsize=8.5)
    fig.text(0.5, 0.011,
             "Nulls use 1,000 permutations within the exact candidate pool; purity-label permutations preserve within-pool cluster prevalence. Error bars: paired participant bootstrap 95% CI.",
             ha="center", fontsize=8.2)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.14, hspace=0.32, wspace=0.18)
    fig.savefig(paths["a_png"], dpi=220, bbox_inches="tight")
    fig.savefig(paths["a_pdf"], bbox_inches="tight")
    fig.savefig(paths["a_thumb"], dpi=80, bbox_inches="tight")
    plt.close(fig)


def plot_figure_b(day_summary: pd.DataFrame, coverage: pd.DataFrame, paths: dict[str, Path]) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.9))
    x = np.arange(len(SUBTYPES))
    for ax, metric, letter in zip(axes.flat[:3], METRICS, ["A", "B", "C"]):
        group = day_summary[(day_summary.measure.eq("retained_per10")) & day_summary.metric.eq(metric)]
        day = group[group.condition.eq("day")].set_index("canonical_stratum").reindex(SUBTYPES)
        night = group[group.condition.eq("night")].set_index("canonical_stratum").reindex(SUBTYPES)
        for index, xpos in enumerate(x):
            ax.plot([xpos - 0.05, xpos + 0.05], [day.estimate.iloc[index], night.estimate.iloc[index]],
                    color=COLORS[metric], lw=1.4)
        for points, shift, face, label in [(day, -0.05, "white", "Day"), (night, 0.05, COLORS[metric], "Night")]:
            ax.errorbar(x + shift, points.estimate,
                        yerr=[points.estimate - points.ci_low, points.ci_high - points.estimate],
                        fmt="o", ms=7, mfc=face, mec=COLORS[metric], mew=1.4,
                        ecolor=COLORS[metric], capsize=3, lw=1.1, label=label)
        ax.set_title(f"{letter}  {METRIC_LABELS[metric]}", loc="left", fontweight="bold", color=COLORS[metric])
        ax.set_ylabel("Equivalent shared neighbors per 10")
        ax.set_ylim(0, 6.5)
        ax.set_xticks(x, [SUBTYPE_LABELS[s] for s in SUBTYPES])
        ax.legend(frameon=False, fontsize=8.5)

    ax = axes.flat[3]
    difference = day_summary[(day_summary.measure.eq("retained_per10")) & day_summary.condition.eq("night_minus_day")].copy()
    difference["classification"] = difference.apply(classify_day_night, axis=1)
    markers = {"reliably higher at night": "^", "no meaningful difference": "s", "uncertain": "o", "reliably lower at night": "v"}
    offsets = [-0.20, 0.0, 0.20]
    ax.axhspan(-EQUIVALENCE_MARGIN_PER10, EQUIVALENCE_MARGIN_PER10, color="#E8E8E8", zorder=0)
    ax.axhline(0, color="black", linestyle="--", lw=0.9)
    for metric, offset in zip(METRICS, offsets):
        points = difference[difference.metric.eq(metric)].set_index("canonical_stratum").reindex(SUBTYPES)
        for index, subtype in enumerate(SUBTYPES):
            row = points.loc[subtype]
            ax.errorbar(x[index] + offset, row.estimate,
                        yerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                        fmt=markers[row.classification], ms=7, mfc=COLORS[metric], mec=COLORS[metric],
                        ecolor=COLORS[metric], capsize=3, lw=1.1)
    ax.set_title("D  Main contrast: paired night minus day", loc="left", fontweight="bold")
    ax.set_ylabel("Δ shared neighbors per 10")
    ax.set_xticks(x, [SUBTYPE_LABELS[s] for s in SUBTYPES])
    metric_handles = [Line2D([0], [0], marker="o", color=COLORS[m], label=METRIC_LABELS[m]) for m in METRICS]
    present = list(dict.fromkeys(difference.classification.tolist()))
    class_handles = [Line2D([0], [0], marker=markers[c], color="#222", lw=0, label=c) for c in present]
    first = ax.legend(handles=metric_handles, frameon=False, fontsize=7.9, loc="upper left")
    ax.add_artist(first); ax.legend(handles=class_handles, frameon=False, fontsize=7.9, loc="lower right")
    for panel in axes.flat:
        panel.tick_params(axis="x", labelsize=9)
        for spine in panel.spines.values():
            spine.set_visible(True); spine.set_color("black")
    fig.suptitle("Day–night neighborhood preservation with paired participant contrasts",
                 fontsize=15.5, fontweight="bold", y=0.985)
    sample_lines = coverage_lines(coverage, daynight=True)
    fig.text(0.5, 0.071, sample_lines[0] + "  |  " + sample_lines[1], ha="center", fontsize=7.8)
    fig.text(0.5, 0.050, sample_lines[2] + "  |  " + sample_lines[3], ha="center", fontsize=7.8)
    fig.text(0.5, 0.026,
             r"Shared per 10 $=10|N_A\cap N_B|/k$. Panel D uses paired participant bootstraps; separate error bars must not be used to infer the day–night contrast.",
             ha="center", fontsize=8.3)
    fig.text(0.5, 0.008,
             "Gray band is the prespecified practical-equivalence region (±0.5 shared neighbors per 10). Reliable requires the full 95% CI beyond this band; otherwise results are uncertain unless the CI lies wholly inside it.",
             ha="center", fontsize=8.0)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.15, hspace=0.34, wspace=0.18)
    fig.savefig(paths["b_png"], dpi=220, bbox_inches="tight")
    fig.savefig(paths["b_pdf"], bbox_inches="tight")
    fig.savefig(paths["b_thumb"], dpi=80, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paths = output_paths()
    refuse_overwrite(paths)
    profiles = pd.read_parquet(PROFILES)
    profiles["participant_id"] = profiles.participant_id.astype(str)
    original_metrics = pd.read_parquet(OUT / "circadian_participant_metrics.parquet")
    original_metrics["participant_id"] = original_metrics.participant_id.astype(str)

    matched_raw = original_metrics[original_metrics.scenario.eq("primary_test_2h")].copy()
    matched = enrich_overlap_rows(matched_raw, "circadian_matched", 100)
    unmatched_raw = reconstruct_unmatched(profiles)
    validation = validate_unmatched_reconstruction(unmatched_raw, matched_raw)
    unmatched = enrich_overlap_rows(unmatched_raw, "unmatched", 200)
    paired_overlap = pd.concat([unmatched, matched], ignore_index=True)

    a_retained = paired_summary(paired_overlap, "retained_per10", ["unmatched", "circadian_matched"], 300)
    a_adjusted = paired_summary(paired_overlap, "adjusted_overlap", ["unmatched", "circadian_matched"], 301)
    purity_raw = pd.read_parquet(OUT / "figure_2C_participant_adjusted_purity.parquet")
    purity = purity_summary(purity_raw)
    figure_a_data = pd.concat([a_retained, a_adjusted, purity], ignore_index=True)

    daynight_raw = original_metrics[original_metrics.scenario.eq("primary_test_day_night")].copy()
    daynight = pd.concat([
        enrich_overlap_rows(daynight_raw[daynight_raw.clock_bin.eq(period)], period, 400 + index)
        for index, period in enumerate(["day", "night"])
    ], ignore_index=True)
    b_retained = paired_summary(daynight, "retained_per10", ["day", "night"], 600)
    b_adjusted = paired_summary(daynight, "adjusted_overlap", ["day", "night"], 601)
    figure_b_data = pd.concat([b_retained, b_adjusted], ignore_index=True)
    mask = figure_b_data.condition.eq("night_minus_day") & figure_b_data.measure.eq("retained_per10")
    figure_b_data.loc[mask, "classification"] = figure_b_data.loc[mask].apply(classify_day_night, axis=1)

    coverage = coverage_table(original_metrics, profiles)
    plot_figure_a(figure_a_data, purity, coverage, paths)
    plot_figure_b(figure_b_data, coverage, paths)
    paired_overlap.to_parquet(paths["participant_overlap"], index=False)
    figure_a_data.to_csv(paths["a_data"], index=False)
    figure_b_data.to_csv(paths["b_data"], index=False)
    coverage.to_csv(paths["coverage"], index=False)

    note = """# Upgraded Figures 2A and 2B: metric definitions

The primary intuitive overlap metric is **equivalent shared neighbors per 10**:

`10 * |N_A intersection N_B| / k = 20 * Jaccard / (1 + Jaccard)`.

The word *equivalent* is essential because the frozen subtype-specific neighborhood sizes are not all 10 (healthy 11; pre-diabetes 8; T2D oral non-insulin 9; insulin-dependent 8), and small circadian pools can reduce effective k. The transformation reports the retained fraction on a common 0–10 scale without changing the frozen graph definition.

For overlap, one size-k neighbor set is held fixed and the other is permuted within the exact candidate pool 1,000 times. Adjusted overlap is `(observed retention - expected retention) / (1 - expected retention)`. Cluster labels are not part of an overlap statistic, so cluster prevalence does not affect this null.

For purity, frozen cluster labels are permuted 1,000 times within the exact subtype, elapsed-hour, and two-hour circadian pool, preserving pool-specific label prevalence. Adjusted purity is `(observed purity - expected purity) / (1 - expected purity)`.

All-clock and circadian-matched estimates use the same cached states, elapsed hours, test participants, clinical representation, h0 representation, ht representation, distances, and frozen subtype-specific k. Only candidate-pool restriction differs. The older frozen unmatched summary is not used in the direct comparison because its ht is an overnight endpoint summary using 0–6-hour anchors rather than a same-hour state.

Day–night differences use paired participant bootstraps. The practical-equivalence region was set transparently at ±0.5 equivalent shared neighbors per 10. A result is called reliably higher at night only when its full 95% interval exceeds +0.5; no meaningful difference requires the full interval inside ±0.5; remaining results are uncertain.
"""
    paths["note"].write_text(note)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "existing_figures_overwritten": False,
        "source_figure_2A_sha256": sha256(OUT / "figure_2A_circadian_matched_reorganization.png"),
        "source_figure_2B_sha256": sha256(OUT / "figure_2B_day_night_reorganization.png"),
        "bootstrap_n": BOOTSTRAP_N,
        "permutation_n": PERMUTATION_N,
        "shared_per10_formula": "10 * intersection / effective_k = 20 * Jaccard / (1 + Jaccard)",
        "adjusted_overlap_formula": "(observed retention - permutation expected retention) / (1 - permutation expected retention)",
        "adjusted_purity_formula": "(observed purity - permutation expected purity) / (1 - permutation expected purity)",
        "overlap_null": "One neighbor set fixed; the other permuted within the exact candidate pool.",
        "purity_null": "Frozen labels permuted within subtype x elapsed hour x exact two-hour candidate pool.",
        "day_night_equivalence_margin_shared_per10": EQUIVALENCE_MARGIN_PER10,
        "same_hour_unmatched_validation": validation,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "complete", "validation": validation,
                      "outputs": {key: str(path) for key, path in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
