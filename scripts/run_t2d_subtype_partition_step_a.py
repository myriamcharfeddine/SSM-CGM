#!/usr/bin/env python3
"""Step A coverage audit for the external T2D clinical partition."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


STATIC_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/enriched_multimodal/"
    "participant_static_features.parquet"
)
SPLIT_PATH = Path(
    "/home/myriamcharfeddine/CGM/Data/"
    "experiment_c_split_adapt6h_seed42/split_participants.csv"
)
OUTPUT_DIRECTORY = Path(
    "/home/myriamcharfeddine/CGM/SSM-CGM/subtype_partition"
)
AUDIT_PATH = OUTPUT_DIRECTORY / "t2d_coverage_audit.csv"
FIGURE_PATH = OUTPUT_DIRECTORY / "fig1_t2d_coverage.png"

PARTICIPANT_COLUMN = "participant_id"
STUDY_GROUP_COLUMN = "participants_study_group"
AGE_COLUMN = "participants_age"
BMI_COLUMN = "bmi_baseline"
C_PEPTIDE_COLUMN = "c_peptide_ngml_baseline"
TRIGLYCERIDES_COLUMN = "triglycerides_mgdl_baseline"
HDL_COLUMN = "hdl_cholesterol_mgdl_baseline"
T2D_STUDY_GROUPS = (
    "insulin_dependent",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled",
)
SPLIT_LABEL_MAP = {"val": "validation", "test": "test"}
ANALYSIS_SPLITS = ("validation", "test")
PRACTICAL_TEST_FLOOR = 80
PRIMARY_K = 3
SENSITIVITY_K = 4
STRATUM_COLORS = ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]
NONFASTING_CAVEAT = (
    "C-peptide and triglycerides were not confirmed fasting measurements; "
    "their values, TG/HDL ratio, coverage, and any later subtype profiles "
    "must be interpreted with this non-fasting caveat."
)
AGE_CAVEAT = (
    "Direct participant age from participants_age is available in the final "
    "enriched multimodal dataset and is identically copied to the static "
    "table. It matches participants.tsv exactly and is not age at diabetes "
    "diagnosis. Exact recalculation from cached OMOP birth fields is not "
    "possible because birth month and day are suppressed and birth_datetime "
    "is a placeholder."
)
COMPLETE_CASE_DEFINITION = (
    "Non-missing BMI, C-peptide, triglycerides, HDL cholesterol, and direct "
    "participant age, with HDL greater than zero so TG/HDL is defined."
)
NO_EM_DASH = "\u2014"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def coverage_row(
    split: str,
    metric: str,
    count: int,
    denominator: int,
    source_column: str,
    viability_note: str,
) -> dict[str, object]:
    return {
        "split": split,
        "metric": metric,
        "count": int(count),
        "denominator": int(denominator),
        "coverage_fraction": (
            float(count / denominator) if denominator > 0 else np.nan
        ),
        "source_column": source_column,
        "t2d_definition_field": STUDY_GROUP_COLUMN,
        "t2d_definition_values": "|".join(T2D_STUDY_GROUPS),
        "age_definition": AGE_CAVEAT,
        "complete_case_definition": COMPLETE_CASE_DEFINITION,
        "nonfasting_caveat": NONFASTING_CAVEAT,
        "practical_test_floor": PRACTICAL_TEST_FLOOR,
        "primary_k": PRIMARY_K,
        "sensitivity_k": SENSITIVITY_K,
        "viability_note": viability_note,
        "static_source_sha256": sha256_file(STATIC_PATH),
        "split_source_sha256": sha256_file(SPLIT_PATH),
    }


def add_value_labels(axis: plt.Axes) -> None:
    for container in axis.containers:
        axis.bar_label(container, fmt="%.0f", padding=3, fontsize=9)


def main() -> None:
    required_columns = [
        PARTICIPANT_COLUMN,
        STUDY_GROUP_COLUMN,
        AGE_COLUMN,
        BMI_COLUMN,
        C_PEPTIDE_COLUMN,
        TRIGLYCERIDES_COLUMN,
        HDL_COLUMN,
    ]
    static = pd.read_parquet(STATIC_PATH, columns=required_columns)
    split = pd.read_csv(SPLIT_PATH, dtype={PARTICIPANT_COLUMN: str})
    static[PARTICIPANT_COLUMN] = static[PARTICIPANT_COLUMN].astype(str)
    if static[PARTICIPANT_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate participant in static table")
    if split[PARTICIPANT_COLUMN].duplicated().any():
        raise RuntimeError("Duplicate participant in split file")
    split["analysis_split"] = split["split"].map(SPLIT_LABEL_MAP)
    selected_split = split[split["analysis_split"].isin(ANALYSIS_SPLITS)]
    frame = selected_split.merge(
        static, on=PARTICIPANT_COLUMN, how="left", validate="one_to_one"
    )
    if frame[required_columns[1:]].isna().all(axis=1).any():
        raise RuntimeError("Validation or test participant missing static row")
    frame["is_t2d"] = frame[STUDY_GROUP_COLUMN].isin(T2D_STUDY_GROUPS)
    frame["tg_hdl"] = (
        frame[TRIGLYCERIDES_COLUMN] / frame[HDL_COLUMN]
    )
    frame.loc[
        frame[HDL_COLUMN].isna() | (frame[HDL_COLUMN] <= 0), "tg_hdl"
    ] = np.nan
    marker_rules = {
        "bmi_nonmissing": frame[BMI_COLUMN].notna(),
        "c_peptide_nonmissing": frame[C_PEPTIDE_COLUMN].notna(),
        "triglycerides_nonmissing": frame[TRIGLYCERIDES_COLUMN].notna(),
        "hdl_cholesterol_nonmissing": frame[HDL_COLUMN].notna(),
        "direct_age_nonmissing": frame[AGE_COLUMN].notna(),
        "tg_hdl_defined": frame["tg_hdl"].notna(),
    }
    frame["complete_case"] = (
        marker_rules["bmi_nonmissing"]
        & marker_rules["c_peptide_nonmissing"]
        & marker_rules["tg_hdl_defined"]
        & marker_rules["direct_age_nonmissing"]
    )

    split_summaries: dict[str, dict[str, int]] = {}
    for analysis_split in ANALYSIS_SPLITS:
        current = frame[frame["analysis_split"] == analysis_split]
        t2d = current[current["is_t2d"]]
        split_summaries[analysis_split] = {
            "full_cohort": len(current),
            "t2d": len(t2d),
            "non_t2d": int((~current["is_t2d"]).sum()),
            "complete_case": int(t2d["complete_case"].sum()),
        }
    test_complete = split_summaries["test"]["complete_case"]
    if test_complete < PRACTICAL_TEST_FLOOR:
        viability_note = (
            f"T2D test complete-case n={test_complete} is below the practical "
            f"floor of about {PRACTICAL_TEST_FLOOR}. k=4 stability is doubtful "
            "and k=3 may also be marginal. Do not cluster before review."
        )
    else:
        viability_note = (
            f"T2D test complete-case n={test_complete} is at or above the "
            f"practical floor of about {PRACTICAL_TEST_FLOOR}. k=4 remains a "
            "sensitivity analysis and still requires validation stability review."
        )

    rows: list[dict[str, object]] = []
    metric_sources = {
        "bmi_nonmissing": BMI_COLUMN,
        "c_peptide_nonmissing": C_PEPTIDE_COLUMN,
        "triglycerides_nonmissing": TRIGLYCERIDES_COLUMN,
        "hdl_cholesterol_nonmissing": HDL_COLUMN,
        "direct_age_nonmissing": AGE_COLUMN,
        "tg_hdl_defined": f"{TRIGLYCERIDES_COLUMN}/{HDL_COLUMN}",
    }
    for analysis_split in ANALYSIS_SPLITS:
        current = frame[frame["analysis_split"] == analysis_split]
        t2d = current[current["is_t2d"]]
        full_count = len(current)
        t2d_count = len(t2d)
        rows.append(
            coverage_row(
                analysis_split,
                "full_cohort",
                full_count,
                full_count,
                "split_participants.csv",
                viability_note,
            )
        )
        rows.append(
            coverage_row(
                analysis_split,
                "t2d",
                t2d_count,
                full_count,
                STUDY_GROUP_COLUMN,
                viability_note,
            )
        )
        rows.append(
            coverage_row(
                analysis_split,
                "non_t2d",
                int((~current["is_t2d"]).sum()),
                full_count,
                STUDY_GROUP_COLUMN,
                viability_note,
            )
        )
        for metric, mask in marker_rules.items():
            rows.append(
                coverage_row(
                    analysis_split,
                    metric,
                    int(mask.loc[t2d.index].sum()),
                    t2d_count,
                    metric_sources[metric],
                    viability_note,
                )
            )
        rows.append(
            coverage_row(
                analysis_split,
                "complete_case_four_marker",
                int(t2d["complete_case"].sum()),
                t2d_count,
                (
                    f"{BMI_COLUMN}|{C_PEPTIDE_COLUMN}|"
                    f"{TRIGLYCERIDES_COLUMN}|{HDL_COLUMN}|{AGE_COLUMN}"
                ),
                viability_note,
            )
        )
    audit = pd.DataFrame(rows)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    atomic_csv(audit, AUDIT_PATH)

    population_rows: list[dict[str, object]] = []
    marker_rows: list[dict[str, object]] = []
    marker_labels = {
        "bmi_nonmissing": "BMI",
        "c_peptide_nonmissing": "C-peptide",
        "triglycerides_nonmissing": "Triglycerides",
        "hdl_cholesterol_nonmissing": "HDL cholesterol",
        "direct_age_nonmissing": "Direct age",
        "tg_hdl_defined": "TG/HDL",
        "complete_case_four_marker": "Complete case",
    }
    for analysis_split in ANALYSIS_SPLITS:
        summary = split_summaries[analysis_split]
        population_rows.extend(
            [
                {
                    "split": analysis_split.title(),
                    "group": "T2D",
                    "count": summary["t2d"],
                },
                {
                    "split": analysis_split.title(),
                    "group": "Non-T2D",
                    "count": summary["non_t2d"],
                },
            ]
        )
        selected = audit[
            (audit["split"] == analysis_split)
            & audit["metric"].isin(marker_labels)
        ]
        for row in selected.itertuples(index=False):
            marker_rows.append(
                {
                    "split": analysis_split.title(),
                    "marker": marker_labels[row.metric],
                    "count": row.count,
                }
            )
    population_plot = pd.DataFrame(population_rows)
    marker_plot = pd.DataFrame(marker_rows)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(
        data=population_plot,
        x="group",
        y="count",
        hue="split",
        palette=STRATUM_COLORS[:2],
        ax=axes[0],
    )
    axes[0].set_title("T2D and non-T2D participants")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Participants")
    add_value_labels(axes[0])
    sns.barplot(
        data=marker_plot,
        x="marker",
        y="count",
        hue="split",
        palette=STRATUM_COLORS[:2],
        ax=axes[1],
    )
    axes[1].set_title("Marker coverage within confirmed T2D")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("T2D participants with data")
    axes[1].tick_params(axis="x", rotation=25)
    add_value_labels(axes[1])
    figure.suptitle(
        "T2D clinical subtype coverage audit\n"
        "C-peptide and triglycerides are not confirmed fasting; "
        "direct age is available in the final enriched dataset and is not age at diagnosis.",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        viability_note,
        ha="center",
        va="bottom",
        fontsize=9,
        color=STRATUM_COLORS[0]
        if test_complete < PRACTICAL_TEST_FLOOR
        else STRATUM_COLORS[1],
    )
    figure.tight_layout(rect=[0, 0.07, 1, 0.93])
    figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    if NO_EM_DASH in AUDIT_PATH.read_text():
        raise RuntimeError("Em dash found in CSV output")
    metadata = {
        "audit_path": str(AUDIT_PATH),
        "figure_path": str(FIGURE_PATH),
        "counts": split_summaries,
        "viability_note": viability_note,
        "nonfasting_caveat": NONFASTING_CAVEAT,
        "age_caveat": AGE_CAVEAT,
        "clustering_executed": False,
        "pause_required": True,
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
