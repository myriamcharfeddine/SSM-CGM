"""Create a non-overwriting circadian Jaccard and adjusted-purity figure."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
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
SEED = 42
PERMUTATION_N = 1000
BOOTSTRAP_N = 1000
HOURS = [6, 12, 24, 48]
SUBTYPES = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
SUBTYPE_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes": "Pre-diabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent*",
}
COLORS = {
    "healthy": "#003366",
    "pre_diabetes": "#2F7F7F",
    "t2d_oral_non_insulin": "#BA2828",
    "insulin_dependent": "#7A8798",
}
SPACE_LABELS = {"clinical": "Clinical", "h0": "h0", "ht": "ht"}


def paths() -> dict[str, Path]:
    return {
        "png": OUT / "figure_2C_circadian_jaccard_adjusted_purity.png",
        "pdf": OUT / "figure_2C_circadian_jaccard_adjusted_purity.pdf",
        "thumbnail": OUT / "figure_2C_circadian_jaccard_adjusted_purity_thumbnail.png",
        "data": OUT / "figure_2C_plotted_data.csv",
        "participant_data": OUT / "figure_2C_participant_adjusted_purity.parquet",
        "metadata": OUT / "figure_2C_metadata.json",
        "note": OUT / "figure_2C_metric_note.md",
    }


def refuse_overwrite(targets: dict[str, Path]) -> None:
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("Figure 2C outputs already exist; refusing to overwrite: " + ", ".join(existing))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def permutation_expected_purity(
    labels: np.ndarray,
    neighbor_sets: dict[str, np.ndarray],
    seed_components: list[int],
) -> dict[str, np.ndarray]:
    """Mean fixed-graph purity after labels are permuted within one candidate pool."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, *seed_components]))
    totals = {space: np.zeros(len(labels), dtype=np.float64) for space in neighbor_sets}
    for _ in range(PERMUTATION_N):
        permuted = rng.permutation(labels)
        for space, neighbors in neighbor_sets.items():
            totals[space] += (permuted[neighbors] == permuted[:, None]).mean(axis=1)
    return {space: values / PERMUTATION_N for space, values in totals.items()}


def recompute_adjusted_purity(profiles: pd.DataFrame) -> pd.DataFrame:
    clinical, h0_matrix, h0_map = phase23.clinical_and_h0(profiles)
    k_map = phase23.original_k(profiles)
    profile_index = profiles.set_index("participant_id")
    rows = []
    for hour in HOURS:
        cache_path = CACHE / f"clock_states_hour{hour:02d}.npz"
        archive = np.load(cache_path, allow_pickle=False)
        states = archive["state"]
        participant_ids = archive["participant_id"].astype(str)
        split = archive["split"].astype(str)
        clock_bins = archive["clock2"].astype(int)
        for subtype_index, subtype in enumerate(SUBTYPES):
            allowed = np.array([
                participant_id in profile_index.index
                and profile_index.at[participant_id, "canonical_stratum"] == subtype
                and split[index] == "test"
                for index, participant_id in enumerate(participant_ids)
            ])
            for clock_bin in sorted(set(clock_bins[allowed].tolist())):
                selected = np.flatnonzero(allowed & (clock_bins == clock_bin))
                grouped: dict[str, list[int]] = {}
                for index in selected:
                    grouped.setdefault(participant_ids[index], []).append(index)
                ids = sorted(grouped)
                if len(ids) < 2:
                    continue
                latent_t = np.stack([
                    states[grouped[participant_id]].mean(axis=0, dtype=np.float32)
                    for participant_id in ids
                ])
                clinical_x = np.stack([clinical[subtype][participant_id] for participant_id in ids])
                latent_h0 = np.stack([h0_matrix[h0_map[participant_id]] for participant_id in ids])
                labels = np.array([
                    int(profile_index.at[participant_id, "display_cluster"])
                    for participant_id in ids
                ])
                effective_k = min(k_map[subtype], len(ids) - 1)
                _, clinical_neighbors = phase23.nearest(clinical_x, effective_k, "euclidean")
                _, h0_neighbors = phase23.nearest(latent_h0, effective_k, "cosine")
                _, ht_neighbors = phase23.nearest(latent_t, effective_k, "cosine")
                neighbor_sets = {
                    "clinical": clinical_neighbors,
                    "h0": h0_neighbors,
                    "ht": ht_neighbors,
                }
                expected = permutation_expected_purity(
                    labels,
                    neighbor_sets,
                    [hour, subtype_index, int(clock_bin)],
                )
                for anchor_index, participant_id in enumerate(ids):
                    for space, neighbors in neighbor_sets.items():
                        observed = float(np.mean(labels[neighbors[anchor_index]] == labels[anchor_index]))
                        chance = float(expected[space][anchor_index])
                        adjusted = (observed - chance) / (1.0 - chance) if chance < 1 else np.nan
                        rows.append({
                            "scenario": "primary_test_2h",
                            "canonical_stratum": subtype,
                            "hour": hour,
                            "clock_bin": str(int(clock_bin)),
                            "participant_id": participant_id,
                            "space": space,
                            "candidate_pool_n": len(ids) - 1,
                            "knn_k": k_map[subtype],
                            "effective_k": effective_k,
                            "underpowered": len(ids) - 1 < k_map[subtype] + 5,
                            "observed_purity": observed,
                            "permutation_expected_purity": chance,
                            "adjusted_purity": adjusted,
                            "permutation_n": PERMUTATION_N,
                        })
        print(f"Adjusted-purity pools complete through hour {hour}", flush=True)
        del archive, states
    return pd.DataFrame(rows)


def validate_observed_purity(recomputed: pd.DataFrame) -> dict[str, float]:
    original = pd.read_parquet(OUT / "circadian_participant_metrics.parquet")
    original = original[original.scenario.eq("primary_test_2h")].copy()
    original["clock_bin"] = original.clock_bin.astype(str)
    original_long = original.melt(
        id_vars=["canonical_stratum", "hour", "clock_bin", "participant_id"],
        value_vars=["clinical_purity", "h0_purity", "ht_purity"],
        var_name="space",
        value_name="original_observed_purity",
    )
    original_long["space"] = original_long.space.str.replace("_purity", "", regex=False)
    merged = recomputed.merge(
        original_long,
        on=["canonical_stratum", "hour", "clock_bin", "participant_id", "space"],
        how="left",
        validate="one_to_one",
    )
    if merged.original_observed_purity.isna().any():
        raise RuntimeError("Failed to align recomputed purity with original Figure 2A metrics")
    differences = (merged.observed_purity - merged.original_observed_purity).abs()
    maximum = float(differences.max())
    if maximum > 1e-12:
        raise RuntimeError(f"Recomputed observed purity differs from Figure 2A: max difference {maximum}")
    return {"maximum_absolute_purity_reproduction_error": maximum, "validated_rows": int(len(merged))}


def bootstrap_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subtype_index, subtype in enumerate(SUBTYPES):
        for space_index, space in enumerate(["clinical", "h0", "ht"]):
            group = data[(data.canonical_stratum.eq(subtype)) & (data.space.eq(space))]
            participant = group.groupby("participant_id").agg(
                adjusted_purity=("adjusted_purity", "mean"),
                observed_purity=("observed_purity", "mean"),
                permutation_expected_purity=("permutation_expected_purity", "mean"),
            ).dropna()
            values = participant.adjusted_purity.to_numpy(float)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 700, subtype_index, space_index]))
            bootstraps = np.array([
                rng.choice(values, len(values), replace=True).mean()
                for _ in range(BOOTSTRAP_N)
            ])
            rows.append({
                "panel": "adjusted_purity",
                "canonical_stratum": subtype,
                "metric": space,
                "estimate": float(values.mean()),
                "ci_low": float(np.percentile(bootstraps, 2.5)),
                "ci_high": float(np.percentile(bootstraps, 97.5)),
                "participant_n": int(len(values)),
                "observed_purity_mean": float(participant.observed_purity.mean()),
                "permutation_expected_purity_mean": float(participant.permutation_expected_purity.mean()),
                "participant_bootstrap_n": BOOTSTRAP_N,
            })
    return pd.DataFrame(rows)


def jaccard_summary() -> pd.DataFrame:
    source = pd.read_csv(OUT / "figure_2A_plotted_data.csv")
    metrics = ["clinical_to_h0_jaccard", "clinical_to_ht_jaccard", "h0_to_ht_jaccard"]
    result = source[(source.scenario.eq("primary_test_2h")) & source.metric.isin(metrics)].copy()
    result["panel"] = "jaccard_similarity"
    result["observed_purity_mean"] = np.nan
    result["permutation_expected_purity_mean"] = np.nan
    return result


def plot(jaccard: pd.DataFrame, adjusted: pd.DataFrame, target_paths: dict[str, Path]) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.color": "#D9D9D9",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    x = np.arange(len(SUBTYPES))
    width = 0.19
    alpha = [0.45, 0.68, 0.92]
    left_metrics = ["clinical_to_h0_jaccard", "clinical_to_ht_jaccard", "h0_to_ht_jaccard"]
    left_labels = ["Clinical vs h0", "Clinical vs ht", "h0 vs ht"]
    for metric_index, (metric, label) in enumerate(zip(left_metrics, left_labels)):
        group = jaccard[jaccard.metric.eq(metric)].set_index("canonical_stratum").reindex(SUBTYPES)
        positions = x + (metric_index - 1) * width
        axes[0].bar(
            positions,
            group.estimate,
            width,
            color=[COLORS[subtype] for subtype in SUBTYPES],
            alpha=alpha[metric_index],
            edgecolor="black",
            linewidth=0.8,
            label=label,
        )
        axes[0].errorbar(
            positions,
            group.estimate,
            yerr=[group.estimate - group.ci_low, group.ci_high - group.estimate],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
    axes[0].set_title("A  Circadian-matched neighborhood Jaccard similarity", loc="left", fontweight="bold")
    axes[0].set_ylabel("Mean participant-level Jaccard similarity")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    for space_index, space in enumerate(["clinical", "h0", "ht"]):
        group = adjusted[adjusted.metric.eq(space)].set_index("canonical_stratum").reindex(SUBTYPES)
        positions = x + (space_index - 1) * width
        axes[1].bar(
            positions,
            group.estimate,
            width,
            color=[COLORS[subtype] for subtype in SUBTYPES],
            alpha=alpha[space_index],
            edgecolor="black",
            linewidth=0.8,
            label=SPACE_LABELS[space],
        )
        axes[1].errorbar(
            positions,
            group.estimate,
            yerr=[group.estimate - group.ci_low, group.ci_high - group.estimate],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_title("B  Fixed-label purity enrichment above chance", loc="left", fontweight="bold")
    axes[1].set_ylabel("Adjusted neighbor purity")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    lower = min(0.0, float(adjusted.ci_low.min()))
    upper = max(0.0, float(adjusted.ci_high.max()))
    margin = max(0.035, 0.12 * (upper - lower))
    axes[1].set_ylim(lower - margin, upper + margin)
    for ax in axes:
        ax.set_xticks(x, [SUBTYPE_LABELS[subtype] for subtype in SUBTYPES], rotation=18, ha="right")
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
    fig.suptitle(
        "Circadian-matched neighborhood similarity and purity enrichment above chance",
        fontsize=15.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.025,
        r"Jaccard similarity: $J(N_A,N_B)=|N_A\cap N_B|/|N_A\cup N_B|$. "
        r"Adjusted purity: $(P_{obs}-P_{perm})/(1-P_{perm})$, where $P_{obs}=k^{-1}\sum_{j\in N_i}\mathbf{1}(y_j=y_i)$.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.006,
        "Expected purity uses 1,000 label permutations within each subtype, elapsed-hour, and two-hour circadian candidate pool. Error bars are 95% participant-bootstrap intervals.",
        ha="center",
        fontsize=8.8,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.20, wspace=0.19)
    fig.savefig(target_paths["png"], dpi=220, bbox_inches="tight")
    fig.savefig(target_paths["pdf"], bbox_inches="tight")
    fig.savefig(target_paths["thumbnail"], dpi=80, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    target_paths = paths()
    refuse_overwrite(target_paths)
    profiles = pd.read_parquet(PROFILES)
    profiles["participant_id"] = profiles.participant_id.astype(str)
    participant_adjusted = recompute_adjusted_purity(profiles)
    validation = validate_observed_purity(participant_adjusted)
    adjusted = bootstrap_summary(participant_adjusted)
    jaccard = jaccard_summary()
    combined = pd.concat([jaccard, adjusted], ignore_index=True, sort=False)
    plot(jaccard, adjusted, target_paths)
    participant_adjusted.to_parquet(target_paths["participant_data"], index=False)
    combined.to_csv(target_paths["data"], index=False)
    metric_note = """# Figure 2C metric definitions

## Panel A

The displayed overlap is **Jaccard similarity**, not neighbor retention or the overlap coefficient:

`J(N_A, N_B) = |N_A intersection N_B| / |N_A union N_B|`.

Each set contains the effective k nearest neighbors for the same anchor and circadian candidate pool. For equal-size sets, Jaccard 0 means no shared neighbors and 1 means identical neighbor sets.

## Panel B

Observed fixed-label purity is:

`P_obs(i) = (1/k) sum[j in N_i] I(label_j = label_i)`.

Expected purity, `P_perm`, is the mean purity after 1,000 permutations of the frozen cluster labels within the exact subtype, elapsed-hour, and two-hour local-clock candidate pool, while holding the neighbor graph fixed. Adjusted purity is:

`Adjusted purity = (P_obs - P_perm) / (1 - P_perm)`.

Zero means chance-level purity under the within-pool label distribution, positive values indicate enrichment above chance, and negative values indicate less same-label neighborhood structure than expected by chance. Frozen labels and graph construction were not changed.
"""
    target_paths["note"].write_text(metric_note)
    original_figure = OUT / "figure_2A_circadian_matched_reorganization.png"
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_figure": str(original_figure),
        "source_figure_sha256": sha256(original_figure),
        "source_metrics": str(OUT / "circadian_participant_metrics.parquet"),
        "primary_scenario": "primary_test_2h",
        "left_metric": "Participant-level k-nearest-neighbor Jaccard similarity",
        "left_formula": "|N_A intersection N_B| / |N_A union N_B|",
        "right_metric": "Purity enrichment above chance",
        "right_formula": "(observed purity - permutation expected purity) / (1 - permutation expected purity)",
        "permutation_n": PERMUTATION_N,
        "permutation_scope": "Frozen cluster labels permuted within subtype x elapsed hour x two-hour circadian candidate pool; neighbor graphs held fixed.",
        "participant_bootstrap_n": BOOTSTRAP_N,
        "latent_distance": "cosine",
        "clinical_distance": "euclidean",
        "frozen_labels_changed": False,
        "neighbor_graph_definition_changed": False,
        "existing_figure_overwritten": False,
        "purity_reproduction_validation": validation,
        "adjusted_purity_summary": adjusted.to_dict(orient="records"),
    }
    target_paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "status": "complete",
        "validation": validation,
        "outputs": {name: str(path) for name, path in target_paths.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
