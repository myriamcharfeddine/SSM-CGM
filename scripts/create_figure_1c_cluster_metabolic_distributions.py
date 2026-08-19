"""Create Figure 1C from frozen participant clusters in raw clinical units."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / (
    "outputs/static_phenotype_trajectory_stratified_v2/"
    "extended_clinical_latent_dynamics_v1/01_cluster_metabolic_profiles"
)
SOURCE = OUT / "participant_frozen_cluster_profiles.parquet"
SEED = 42

SUBTYPES = [
    "healthy",
    "pre_diabetes",
    "t2d_oral_non_insulin",
    "insulin_dependent",
]
SUBTYPE_LABELS = {
    "healthy": "Healthy",
    "pre_diabetes": "Pre-diabetes",
    "t2d_oral_non_insulin": "T2D oral non-insulin",
    "insulin_dependent": "Insulin-dependent*",
}
PALETTES = {
    "healthy": ["#003366", "#5B7FA3"],
    "pre_diabetes": ["#2F7F7F", "#7BC4C4"],
    "t2d_oral_non_insulin": ["#7A1F1F", "#BA2828", "#E57373"],
    "insulin_dependent": ["#4A5568", "#7A8798", "#B0BAC6"],
}
VARIABLES = [
    ("cgm_mean", "Mean CGM (mg/dL)", 1.0, "linear", None),
    ("cgm_cv", "CGM CV (%)", 100.0, "linear", None),
    ("cgm_time_in_range", "Time in range (%)", 100.0, "linear", (0, 100)),
    ("cgm_time_above_180", "Time above 180 (%)", 100.0, "linear", (0, 100)),
    ("cgm_time_below_70", "Time below 70 (%)", 100.0, "asinh", [0, 0.1, 0.5, 1, 5, 10, 40]),
    ("cgm_masd", "Mean absolute successive\ndifference (mg/dL)", 1.0, "linear", None),
    ("hba1c_percent_baseline", "HbA1c (%)", 1.0, "linear", None),
    ("c_peptide_ngml_baseline", "C-peptide (ng/mL)", 1.0, "linear", None),
    ("tg_hdl_ratio", "TG/HDL (ratio)", 1.0, "asinh", [0, 0.5, 1, 2, 5, 10, 20, 50]),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths() -> dict[str, Path]:
    return {
        "png": OUT / "figure_1C_cluster_metabolic_distributions.png",
        "pdf": OUT / "figure_1C_cluster_metabolic_distributions.pdf",
        "data": OUT / "figure_1C_plotted_data.csv",
        "metadata": OUT / "figure_1C_metadata.json",
        "note": OUT / "figure_1C_interpretation_note.md",
    }


def refuse_overwrite(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Figure 1C outputs already exist; refusing to overwrite: " + ", ".join(existing))


def long_data(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, label, multiplier, scale, _ in VARIABLES:
        part = frame[["participant_id", "split", "canonical_stratum", "display_cluster", source]].copy()
        part["value_raw"] = pd.to_numeric(part[source], errors="coerce") * multiplier
        part["variable"] = source
        part["variable_label"] = label.replace("\n", " ")
        part["display_multiplier"] = multiplier
        part["axis_scale"] = scale
        rows.append(part.drop(columns=source).dropna(subset=["value_raw"]))
    return pd.concat(rows, ignore_index=True)


def separation_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subtype, variable), group in data.groupby(["canonical_stratum", "variable"], sort=False):
        medians = group.groupby("display_cluster").value_raw.median().sort_index()
        overall_iqr = group.value_raw.quantile(0.75) - group.value_raw.quantile(0.25)
        score = (medians.max() - medians.min()) / overall_iqr if overall_iqr > 0 else np.nan
        rows.append({
            "canonical_stratum": subtype,
            "variable": variable,
            "separation_score_median_range_over_iqr": score,
            "cluster_medians": "; ".join(f"C{int(k)}={v:.3g}" for k, v in medians.items()),
        })
    return pd.DataFrame(rows)


def interpretation_note(data: pd.DataFrame) -> str:
    sep = separation_table(data)
    labels = {source: label.replace("\n", " ") for source, label, *_ in VARIABLES}
    paragraphs = [
        "# Figure 1C interpretation note",
        "",
        "This figure displays participant-level values in raw clinical units for the frozen within-subtype clusters. "
        "The separation ranking below uses the range of cluster medians divided by the subtype-wide interquartile range only as a concise visual guide; it is not a new clustering analysis or an inferential test.",
        "",
    ]
    mean_scores = {}
    for subtype in SUBTYPES:
        ranked = sep[sep.canonical_stratum.eq(subtype)].sort_values(
            "separation_score_median_range_over_iqr", ascending=False
        )
        top = ranked.head(3)
        weak = ranked.tail(2)
        mean_scores[subtype] = float(ranked.separation_score_median_range_over_iqr.mean())
        top_text = ", ".join(
            f"{labels[row.variable]} ({row.cluster_medians})" for row in top.itertuples()
        )
        weak_text = ", ".join(labels[row.variable] for row in weak.itertuples())
        paragraphs.extend([
            f"## {SUBTYPE_LABELS[subtype]}",
            "",
            f"The clearest median separation is seen for {top_text}. The heaviest overlap is seen for {weak_text}, "
            "so those variables have comparatively weak visual discriminatory value. Directional combinations of mean CGM, time in range, time above 180, HbA1c, C-peptide, and TG/HDL provide the most direct support for the prior literature-aligned interpretation; overlapping boxes and points prevent treating that interpretation as a formal subtype assignment.",
            "",
        ])
    three_cluster = np.mean([mean_scores["t2d_oral_non_insulin"], mean_scores["insulin_dependent"]])
    two_cluster = np.mean([mean_scores["healthy"], mean_scores["pre_diabetes"]])
    comparison = "clearer" if three_cluster > two_cluster else "not consistently clearer"
    paragraphs.extend([
        "## Cross-subtype comparison",
        "",
        f"Across the nine displayed variables, the three-cluster T2D oral non-insulin and insulin-dependent strata show {comparison} median differentiation than the two-cluster healthy and pre-diabetes strata. "
        "This comparison is descriptive because cluster counts, sample sizes, and within-subtype ranges differ. Insulin-dependent findings remain exploratory.",
        "",
        "Baseline insulin is not displayed because the audit found no sufficiently complete validated participant-level laboratory insulin measurement. The treatment indicator `med_insulin` was not substituted.",
    ])
    return "\n".join(paragraphs) + "\n"


def draw(frame: pd.DataFrame, data: pd.DataFrame, paths: dict[str, Path]) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(len(VARIABLES), len(SUBTYPES), figsize=(18, 30), sharey="row")
    rng = np.random.default_rng(SEED)
    counts = frame.groupby(["canonical_stratum", "display_cluster"]).participant_id.nunique().to_dict()
    for row_index, (variable, ylabel, _, scale, scale_detail) in enumerate(VARIABLES):
        for col_index, subtype in enumerate(SUBTYPES):
            ax = axes[row_index, col_index]
            panel = data[(data.variable.eq(variable)) & (data.canonical_stratum.eq(subtype))]
            clusters = sorted(panel.display_cluster.astype(int).unique())
            values = [panel.loc[panel.display_cluster.eq(cluster), "value_raw"].to_numpy(float) for cluster in clusters]
            boxes = ax.boxplot(
                values,
                positions=np.arange(len(clusters)),
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.25},
                whiskerprops={"color": "black", "linewidth": 0.9},
                capprops={"color": "black", "linewidth": 0.9},
            )
            for patch, color in zip(boxes["boxes"], PALETTES[subtype]):
                patch.set_facecolor(color)
                patch.set_alpha(0.72)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.9)
            for x_position, cluster_values, color in zip(range(len(clusters)), values, PALETTES[subtype]):
                jitter = rng.uniform(-0.17, 0.17, len(cluster_values))
                ax.scatter(
                    x_position + jitter,
                    cluster_values,
                    s=8,
                    alpha=0.24,
                    color=color,
                    edgecolors="none",
                    rasterized=True,
                    zorder=1,
                )
            ax.set_xticks(
                np.arange(len(clusters)),
                [f"C{cluster}\nN={counts[(subtype, cluster)]:,}" for cluster in clusters],
            )
            ax.tick_params(axis="x", labelsize=8, pad=3)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", visible=True)
            if col_index == 0:
                ax.set_ylabel(ylabel, fontsize=9.5, labelpad=7)
            else:
                ax.set_ylabel("")
            if row_index == 0:
                ax.set_title(SUBTYPE_LABELS[subtype], fontsize=12, fontweight="normal", color=PALETTES[subtype][0], pad=9)
            if scale == "asinh":
                linear_width = 0.5 if variable == "cgm_time_below_70" else 1.0
                ax.set_yscale(
                    "function",
                    functions=(
                        lambda values, width=linear_width: np.arcsinh(values / width),
                        lambda values, width=linear_width: np.sinh(values) * width,
                    ),
                )
                ax.set_yticks(scale_detail)
                ax.set_yticklabels([f"{tick:g}" for tick in scale_detail])
            elif scale_detail is not None:
                ax.set_ylim(*scale_detail)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.8)
    fig.suptitle(
        "Frozen clinical clusters show distinct glucose and metabolic-expression distributions",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.018,
        "Values are shown in raw units; clusters are frozen separately within subtype. Age is omitted because the figure focuses on glucose and metabolic expression.\n"
        "Time below 70 and TG/HDL use asinh axis spacing with ticks labeled in raw units so zero values and extreme outliers remain visible. "
        "Baseline insulin was unavailable as a validated laboratory measure. Insulin-dependent results remain exploratory.",
        ha="center",
        va="bottom",
        fontsize=9,
        linespacing=1.35,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.965, bottom=0.055, hspace=0.48, wspace=0.16)
    fig.savefig(paths["png"], dpi=220, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paths = output_paths()
    refuse_overwrite(paths)
    frame = pd.read_parquet(SOURCE)
    frame["participant_id"] = frame.participant_id.astype(str)
    expected_k = {"healthy": 2, "pre_diabetes": 2, "t2d_oral_non_insulin": 3, "insulin_dependent": 3}
    observed_k = frame.groupby("canonical_stratum").display_cluster.nunique().astype(int).to_dict()
    if observed_k != expected_k:
        raise RuntimeError(f"Frozen cluster count changed: {observed_k}")
    data = long_data(frame)
    draw(frame, data, paths)
    data.to_csv(paths["data"], index=False)
    paths["note"].write_text(interpretation_note(data))
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "participant_count": int(frame.participant_id.nunique()),
        "subtype_order": SUBTYPES,
        "cluster_counts": {
            subtype: {
                f"C{int(cluster)}": int(count)
                for cluster, count in frame[frame.canonical_stratum.eq(subtype)]
                .groupby("display_cluster").participant_id.nunique().items()
            }
            for subtype in SUBTYPES
        },
        "variables": [
            {
                "source": source,
                "label": label.replace("\n", " "),
                "display_multiplier": multiplier,
                "axis_scale": scale,
                "raw_values_not_standardized": True,
            }
            for source, label, multiplier, scale, _ in VARIABLES
        ],
        "baseline_insulin_included": False,
        "baseline_insulin_audit": "No sufficiently complete validated laboratory measurement; med_insulin is treatment status and was not substituted.",
        "cluster_refit": False,
        "frozen_labels_changed": False,
        "existing_heatmap_modified": False,
        "boxplot_fliers": "All individual raw observations are overlaid; boxplot duplicate fliers are suppressed.",
        "seed": SEED,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "complete", "outputs": {key: str(value) for key, value in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
