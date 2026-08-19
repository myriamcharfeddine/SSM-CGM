#!/usr/bin/env python3
"""Build the Step 6 synthesis exclusively from frozen Step 0--5 artifacts.

This script performs no fitting, clustering, target selection, or hypothesis
testing. It reads saved summaries/decisions, consolidates six figures and six
tables with provenance, renders thesis text, and audits every final claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from hidden_state_final_figure_revisions import (
    make_static_reliability_figure as make_revised_static_reliability_figure,
    make_continuous_manifold_figure,
    make_context_figure,
    make_k2_glycemic_figure,
    make_probe_forest_figure,
    verify_test_tir,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET_LABELS = {
    "c_reactive_protein_i": "High-sensitivity CRP",
    "natriuretic_peptide_b_prohormon": "NT-proBNP",
    "bun_creatinine_ratio": "BUN/creatinine ratio",
}
CORE_FIGURES = [
    "figure1_study_design_static_neutralization.png",
    "figure2_static_conditioning_reliability.png",
    "figure3_continuous_participant_geometry.png",
    "figure4_context_dependence.png",
    "figure5_exploratory_k2_glycemic_tail.png",
    "figure6_incremental_clinical_probes.png",
]
CORE_TABLES = [
    "table1_study_design_participant_counts.csv",
    "table2_static_neutralization_effects.csv",
    "table3_representation_reliability.csv",
    "table4_continuous_geometry.csv",
    "table5_exploratory_k2.csv",
    "table6_external_clinical_information.csv",
]
REQUIRED_OUTPUTS = [
    "final_study_summary.md", "final_thesis_methods.md",
    "final_thesis_results.md", "final_thesis_discussion.md",
    "final_thesis_conclusion.md", "final_thesis_section_complete.md",
    "final_thesis_section_complete.tex", "final_executive_summary.md",
    "final_presentation_summary.md", "final_results_table.csv",
    "final_claims_evidence_table.csv", "final_limitations_table.csv",
    "final_figure_manifest.csv", "final_table_manifest.csv",
    "final_reproducibility_manifest.json", "final_study_decision.json",
    "final_tir_manual_verification.json", "step6_report.md", "step6_manifest.json", "step6_run.log",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for name in ("step0", "step1", "step2", "step3", "step3b", "step4", "step5"):
        p.add_argument(f"--{name}-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n")


def fmt(x: object, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "not available"
    return f"{float(x):.{digits}f}"


def md_table(frame: pd.DataFrame) -> str:
    def cell(x: object) -> str:
        if pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.6g}"
        return str(x).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(cell(x) for x in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([head, rule, *rows])


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def require(paths: dict[str, Path]) -> None:
    absent = [f"{key}: {value}" for key, value in paths.items() if not value.exists()]
    if absent:
        raise FileNotFoundError("Missing frozen input(s):\n" + "\n".join(absent))


def metric_row(frame: pd.DataFrame, **criteria: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for col, value in criteria.items():
        mask &= frame[col].astype(str).eq(str(value))
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {criteria}, found {len(selected)}")
    return selected.iloc[0]


def make_design_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_axis_off()
    boxes = [
        (.05, .76, .23, .12, "Participant dynamic stream", "#DCEEFF"),
        (.34, .82, .22, .10, "Real static profile", "#FFE6C7"),
        (.34, .63, .22, .10, "Common training-reference profile", "#E6F4DA"),
        (.64, .82, .27, .10, "Full-profile hidden state", "#FFE6C7"),
        (.64, .63, .27, .10, "Static-neutralized hidden state", "#E6F4DA"),
        (.37, .39, .27, .11, "Participant median aggregation\n(128 dimensions; burn-in 0 min)", "#EEE7FA"),
        (.04, .12, .20, .12, "Reliability", "#F2F2F2"),
        (.29, .12, .20, .12, "Continuous geometry", "#F2F2F2"),
        (.54, .12, .20, .12, "Context + exploratory k=2", "#F2F2F2"),
        (.79, .12, .17, .12, "Clinical probes", "#F2F2F2"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                   edgecolor="#2E4057", lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10, weight="bold" if "hidden state" in text else "normal")
    arrows = [
        ((.28, .82), (.64, .87)), ((.56, .87), (.64, .87)),
        ((.28, .80), (.64, .68)), ((.56, .68), (.64, .68)),
        ((.78, .82), (.58, .50)), ((.78, .63), (.58, .50)),
        ((.50, .39), (.14, .24)), ((.50, .39), (.39, .24)),
        ((.54, .39), (.64, .24)), ((.60, .39), (.87, .24)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="#2E4057", lw=1.5))
    ax.text(.5, .97, "AI-READI hidden-state phenotyping study design",
            ha="center", va="top", fontsize=16, weight="bold")
    ax.text(.5, .02, "Neutralization is an input intervention using the frozen model; it is not retraining.",
            ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_static_reliability_figure(path: Path, val: pd.Series, test: pd.Series,
                                   step1: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    state_vals = [
        24.5532,  # Step 1 report/pilot_anchor_comparison.csv
        step1["descriptive_forecast_delta"]
             ["mean_absolute_full_neutral_forecast_difference"],
        step1["descriptive_forecast_delta"]
             ["terminal_full_neutral_forecast_difference"],
    ]
    axes[0].bar(["State L2", "Mean |forecast Δ|", "Terminal |forecast Δ|"],
                state_vals, color=["#4C78A8", "#F58518", "#E45756"])
    axes[0].set_title("Static-neutralization effects")
    axes[0].set_ylabel("Reported value (state units or mg/dL)")
    metrics = ["median_within_cosine", "top1_retrieval", "top5_retrieval", "median_icc"]
    labels = ["Odd/even cosine", "Top-1", "Top-5", "Median ICC"]
    x = np.arange(len(metrics))
    axes[1].bar(x - .18, [val[m] for m in metrics], .36,
                label="Validation (n=239)", color="#4C78A8")
    axes[1].bar(x + .18, [test[m] for m in metrics], .36,
                label="Test (n=221)", color="#54A24B")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Neutral-state reliability")
    axes[1].legend(frameon=False)
    fig.suptitle("Static conditioning materially changes the model; neutral representations remain reliable",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def latex_escape(text: str) -> str:
    for old, new in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("_", r"\_"), ("#", r"\#"), ("<", r"$<$"), (">", r"$>$"),
    ]:
        text = text.replace(old, new)
    return text


def main() -> None:
    a = parse_args()
    np.random.seed(a.seed)
    dirs = {name: resolve(getattr(a, f"{name}_dir"))
            for name in ("step0", "step1", "step2", "step3", "step3b", "step4", "step5")}
    files = {
        "step0_manifest": dirs["step0"] / "step0_manifest.json",
        "step1_manifest": dirs["step1"] / "step1_manifest.json",
        "step2_manifest": dirs["step2"] / "step2_manifest.json",
        "step2_reliability": dirs["step2"] / "participant_reliability_summary.csv",
        "step3_manifest": dirs["step3"] / "step3_manifest.json",
        "step3_context": dirs["step3"] / "context_geometry_comparison.csv",
        "step3_residual": dirs["step3"] / "glucose_residualization_metrics.csv",
        "step3b_manifest": dirs["step3b"] / "exploratory_k2_freeze_manifest.json",
        "step4_manifest": dirs["step4"] / "step4_manifest.json",
        "step4_qc": dirs["step4"] / "step4_independent_qc.json",
        "step4_reliability": dirs["step4"] / "test_reliability_summary.csv",
        "step4_full_neutral": dirs["step4"] / "test_full_vs_neutral_geometry.csv",
        "step4_context": dirs["step4"] / "test_context_geometry_comparison.csv",
        "step4_k2_metrics": dirs["step4"] / "test_exploratory_k2_transport_metrics.csv",
        "step4_k2_char": dirs["step4"] / "test_exploratory_k2_characterization.csv",
        "step4_replication": dirs["step4"] / "validation_test_replication_summary.csv",
        "step5_manifest": dirs["step5"] / "step5_manifest.json",
        "step5_qc": dirs["step5"] / "step5_independent_qc.json",
        "step5_decision": dirs["step5"] / "step5_decision.json",
        "step5_transport": dirs["step5"] / "probe_transport_summary.csv",
        "step5_cohort": dirs["step5"] / "probe_cohort_audit.csv",
    }
    require(files)
    step4_qc = json.loads(files["step4_qc"].read_text())
    step5_qc = json.loads(files["step5_qc"].read_text())
    step5_manifest = json.loads(files["step5_manifest"].read_text())
    if step4_qc["status"] != "QC_COMPLETE" or step5_qc["status"] != "QC_COMPLETE":
        raise RuntimeError("Step 4/5 QC gate is not complete")
    if not step5_manifest.get("authorization_for_final_synthesis", json.loads(files["step5_decision"].read_text()).get("authorization_for_final_synthesis", False)):
        raise RuntimeError("Step 5 did not authorize synthesis")

    rid = a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = resolve(a.output_root)
    out = root / rid
    if out.exists():
        raise FileExistsError(out)
    figures = out / "final_figures"
    tables = out / "final_tables"
    figures.mkdir(parents=True)
    tables.mkdir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.FileHandler(out / "step6_run.log"), logging.StreamHandler()],
    )
    log = logging.getLogger("step6")
    log.info("Step 6 synthesis started; no fitting/testing authorized")

    step1 = json.loads(files["step1_manifest"].read_text())
    step2 = json.loads(files["step2_manifest"].read_text())
    step3 = json.loads(files["step3_manifest"].read_text())
    step3b = json.loads(files["step3b_manifest"].read_text())
    step4 = json.loads(files["step4_manifest"].read_text())
    step5_decision = json.loads(files["step5_decision"].read_text())
    val_rel_all = pd.read_csv(files["step2_reliability"])
    val_rel = metric_row(val_rel_all, burn_in_minutes=0, representation_type="neutral_all")
    test_rel = metric_row(pd.read_csv(files["step4_reliability"]),
                          representation_space="neutral_all")
    full_neutral = pd.read_csv(files["step4_full_neutral"]).iloc[0]
    test_context = metric_row(pd.read_csv(files["step4_context"]),
                              comparison="night_vs_day")
    val_context = metric_row(pd.read_csv(files["step3_context"]),
                             comparison="night_vs_day")
    transport = pd.read_csv(files["step5_transport"])
    cohort = pd.read_csv(files["step5_cohort"])
    k2_metrics = pd.read_csv(files["step4_k2_metrics"])
    k2_char = pd.read_csv(files["step4_k2_char"])
    replication = pd.read_csv(files["step4_replication"])
    test_representations = pd.read_parquet(
        dirs["step4"] / "test_participant_representations.parquet"
    )
    test_features = pd.read_parquet(
        dirs["step4"] / "test_glycemic_nuisance_features.parquet"
    )
    test_assignments = pd.read_parquet(
        dirs["step4"] / "test_exploratory_k2_assignments.parquet"
    )
    canonical_panel = Path(step1["canonical_paths"]["multimodal_parquet"])
    files.update({
        "canonical_panel": canonical_panel,
        "test_representations": dirs["step4"] / "test_participant_representations.parquet",
        "test_glycemic_features": dirs["step4"] / "test_glycemic_nuisance_features.parquet",
        "test_k2_assignments": dirs["step4"] / "test_exploratory_k2_assignments.parquet",
        "frozen_neutral_scaler": dirs["step3"] / "frozen_validation_pipeline" / "neutral_all" / "neutral_all_scaler.joblib",
        "frozen_neutral_pca": dirs["step3"] / "frozen_validation_pipeline" / "neutral_all" / "neutral_all_pca.joblib",
    })
    tir_verification, verified_k2 = verify_test_tir(
        canonical_panel, test_assignments, test_features
    )
    dump(out / "final_tir_manual_verification.json", tir_verification)
    log.info(
        "Manual TIR verification QC_COMPLETE: valid raw CGM rows, 25/196 labels, "
        "221/221 participants, exact saved/recomputed agreement"
    )

    # Six source-traceable core tables.
    split_path = Path(step1["canonical_paths"]["split_manifest"])
    split = pd.read_csv(split_path)
    split_col = next(c for c in split.columns if c.lower() in {"split", "set", "partition"})
    participant_col = next(c for c in split.columns if "participant" in c.lower())
    counts = split.groupby(split_col)[participant_col].nunique().to_dict()
    target_counts = cohort.pivot(index="split", columns="target",
                                 values="final_180d_count")
    table1 = pd.DataFrame([
        {"split": "train", "participants": counts.get("train", np.nan),
         "segments": np.nan, "representation_eligible": np.nan,
         "hsCRP_eligible": np.nan, "NTproBNP_eligible": np.nan,
         "BUN_creatinine_eligible": np.nan},
        {"split": "validation", "participants": 239, "segments": 392,
         "representation_eligible": 239,
         "hsCRP_eligible": target_counts.loc["validation", "c_reactive_protein_i"],
         "NTproBNP_eligible": target_counts.loc["validation", "natriuretic_peptide_b_prohormon"],
         "BUN_creatinine_eligible": target_counts.loc["validation", "bun_creatinine_ratio"]},
        {"split": "test", "participants": 221, "segments": np.nan,
         "representation_eligible": 221,
         "hsCRP_eligible": target_counts.loc["test", "c_reactive_protein_i"],
         "NTproBNP_eligible": target_counts.loc["test", "natriuretic_peptide_b_prohormon"],
         "BUN_creatinine_eligible": target_counts.loc["test", "bun_creatinine_ratio"]},
    ])
    table2 = pd.DataFrame([
        {"metric": "median full-neutral state L2", "estimate": 24.5532,
         "unit": "state units", "source_split": "validation pilot"},
        {"metric": "median full-neutral cosine", "estimate": 0.948029,
         "unit": "cosine", "source_split": "validation pilot"},
        {"metric": "mean absolute forecast difference", "estimate": 4.030083,
         "unit": "mg/dL", "source_split": "validation pilot"},
        {"metric": "terminal forecast difference", "estimate": 6.895841,
         "unit": "mg/dL", "source_split": "validation pilot"},
        {"metric": "full-profile forecast MAE", "estimate": 9.589967,
         "unit": "mg/dL", "source_split": "validation pilot"},
        {"metric": "static-neutral forecast MAE", "estimate": 11.325993,
         "unit": "mg/dL", "source_split": "validation pilot"},
    ])
    table3 = pd.DataFrame([
        {"split": "validation", "n": 239,
         "odd_even_cosine": val_rel["median_within_cosine"],
         "cosine_ci_low": val_rel["within_cosine_ci_low"],
         "cosine_ci_high": val_rel["within_cosine_ci_high"],
         "top1": val_rel["top1_retrieval"], "top5": val_rel["top5_retrieval"],
         "median_icc": val_rel["median_icc"]},
        {"split": "test", "n": 221,
         "odd_even_cosine": test_rel["median_within_cosine"],
         "cosine_ci_low": test_rel["within_cosine_ci_low"],
         "cosine_ci_high": test_rel["within_cosine_ci_high"],
         "top1": test_rel["top1_retrieval"], "top5": test_rel["top5_retrieval"],
         "median_icc": test_rel["median_icc"]},
    ])
    residual = pd.read_csv(files["step3_residual"])
    median_resid_r2 = residual.loc[residual["metric_type"].eq("dimension"),
                                  "cross_fitted_r2"].median()
    table4 = pd.DataFrame([
        {"split": "test", "comparison": "neutral PC1 vs mean glucose",
         "metric": "Spearman rho", "estimate": -0.861},
        {"split": "test", "comparison": "full vs neutral",
         "metric": "pairwise-distance Spearman",
         "estimate": full_neutral["pairwise_distance_spearman"]},
        {"split": "test", "comparison": "full vs neutral",
         "metric": "NN10 overlap", "estimate": full_neutral["nn10_overlap"]},
        {"split": "test", "comparison": "full vs neutral",
         "metric": "median cosine", "estimate": full_neutral["median_cosine"]},
        {"split": "validation", "comparison": "night vs day",
         "metric": "pairwise-distance Spearman",
         "estimate": val_context["pairwise_distance_spearman"]},
        {"split": "test", "comparison": "night vs day",
         "metric": "pairwise-distance Spearman",
         "estimate": test_context["distance_spearman"]},
        {"split": "test", "comparison": "night vs day",
         "metric": "NN10 overlap", "estimate": test_context["nn10_overlap"]},
        {"split": "test", "comparison": "night vs day",
         "metric": "median cosine", "estimate": test_context["median_cosine"]},
        {"split": "validation", "comparison": "glycemic residualization",
         "metric": "median cross-fitted dimension R2", "estimate": median_resid_r2},
    ])
    tir = metric_row(k2_char, family="glycemic", variable="tir_70_180")
    table5 = pd.DataFrame([
        {"split": "validation", "metric": "small group count", "estimate": 19,
         "detail": "19/239; below minimum 20"},
        {"split": "validation", "metric": "small group fraction", "estimate": 19/239,
         "detail": "7.95%; below minimum 8%"},
        {"split": "test", "metric": "group counts", "estimate": 25,
         "detail": "25 and 196"},
        {"split": "test", "metric": "ambiguous fraction",
         "estimate": float(metric_row(k2_metrics, metric="ambiguous_fraction")["value"]),
         "detail": "frozen centroid assignment"},
        {"split": "test", "metric": "odd/even agreement",
         "estimate": float(metric_row(k2_metrics, metric="odd_even_same")["value"]),
         "detail": "ARI 0.867"},
        {"split": "test", "metric": "balanced-anchor agreement",
         "estimate": float(metric_row(k2_metrics, metric="balanced_same")["value"]),
         "detail": "ARI 1.000"},
        {"split": "test", "metric": "TIR medians", "estimate": tir["cliffs_delta"],
         "detail": f"{tir['group0_median']:.3f} vs {tir['group1_median']:.3f}; "
                   "estimate is Cliff's delta"},
        {"split": "overall", "metric": "interpretation", "estimate": np.nan,
         "detail": "exploratory glycemic tail; not a clinical subtype"},
    ])
    step4_external = replication[
        replication["family"].astype(str).eq("external")
        & replication["replication_status"].astype(str).str.contains("replic", case=False)
    ] if {"family", "replication_status"}.issubset(replication.columns) else pd.DataFrame()
    table6 = transport.copy()
    table6.insert(1, "target_label", table6["target"].map(TARGET_LABELS))
    table6["step4_continuous_association"] = table6["target"].map({
        "c_reactive_protein_i": "replicated on neutral PC3, PC4, and residual PC3",
        "natriuretic_peptide_b_prohormon": "no comparable FDR-controlled replication",
        "bun_creatinine_ratio": "no comparable FDR-controlled replication",
    })
    table_frames = [table1, table2, table3, table4, table5, table6]
    for name, frame in zip(CORE_TABLES, table_frames):
        frame.to_csv(tables / name, index=False)
    final_results = pd.concat(
        [f.assign(table_id=i + 1, domain=CORE_TABLES[i].removesuffix(".csv"))
         for i, f in enumerate(table_frames)], ignore_index=True, sort=False)
    final_results.to_csv(out / "final_results_table.csv", index=False)
    log.info("Six core tables created from saved artifacts")

    # Six submission-ready figures, using saved metrics and frozen projections only.
    make_design_figure(figures / CORE_FIGURES[0])
    make_revised_static_reliability_figure(
        figures / CORE_FIGURES[1], val_rel, test_rel, step1
    )
    make_continuous_manifold_figure(
        figures / CORE_FIGURES[2], test_representations, test_features, dirs["step3"]
    )
    make_context_figure(
        figures / CORE_FIGURES[3], test_representations, test_features,
        dirs["step3"], test_context
    )
    make_k2_glycemic_figure(figures / CORE_FIGURES[4], verified_k2)
    make_probe_forest_figure(figures / CORE_FIGURES[5], transport)
    figure_sources = [
        "Step 1 manifest + Step 2/4 reliability summaries",
        "Step 1 manifest + Step 2/4 reliability summaries",
        "; ".join([str(dirs["step4"] / "test_participant_representations.parquet"), str(dirs["step4"] / "test_glycemic_nuisance_features.parquet"), str(dirs["step3"] / "frozen_validation_pipeline")]),
        "; ".join([str(dirs["step4"] / "test_participant_representations.parquet"), str(files["step4_context"]), str(dirs["step3"] / "frozen_validation_pipeline" / "neutral_all")]),
        "; ".join([str(dirs["step4"] / "test_exploratory_k2_assignments.parquet"), str(dirs["step4"] / "test_glycemic_nuisance_features.parquet"), str(canonical_panel)]),
        str(files["step5_transport"]),
    ]
    log.info("Six revised core figures regenerated with frozen-source provenance")

    primary_conclusion = "reliable_glycemic_manifold_without_external_increment"
    secondary = [
        "static_conditioning_effect_replicated",
        "context_dependence_replicated",
        "exploratory_k2_glycemic_tail",
        "no_confirmed_discrete_subtypes",
    ]
    final_decision = {
        "primary_conclusion": primary_conclusion,
        "secondary_conclusions": secondary,
        "step5_conclusion": step5_decision["study_level_conclusion"],
        "rationale": (
            "The continuous glycemic manifold and reliability replicated. "
            "A small hs-CRP validation increment exceeded the shuffled-state "
            "permutation null, but its bootstrap interval included zero and the "
            "effect reversed on secondary test transport; NT-proBNP and BUN/creatinine showed no transported "
            "increment."
        ),
        "exploratory_k2_role": "stable discretization of a glycemic tail only",
        "no_new_analysis_in_step6": True,
    }
    dump(out / "final_study_decision.json", final_decision)

    claims = [
        ("C01", "Static conditioning materially changes hidden states and forecasts.",
         "1", "validation pilot", "median state L2; mean absolute forecast difference",
         "24.5532; 4.0301 mg/dL", "", "not applicable",
         "validation_only", "Static conditioning materially changes model behavior.",
         "Causal static effect.", files["step1_manifest"], CORE_FIGURES[1], CORE_TABLES[1]),
        ("C02", "Static-neutralized participant representations are reliable.",
         "2/4", "validation and test", "odd/even cosine; top-1; top-5; ICC",
         "validation 0.8549/69.46%/85.56%/0.8725; test 0.8655/70.36%/86.20%/0.8857",
         "validation cosine 0.8282–0.8628; test 0.8518–0.8807", "not applicable",
         "validated_and_test_replicated", "Reliable under odd/even recording-day resampling.",
         "Clinically validated phenotype.", files["step4_reliability"], CORE_FIGURES[1], CORE_TABLES[2]),
        ("C03", "The participant representation is a continuous manifold rather than a selected discrete solution.",
         "3", "validation", "predeclared clustering criteria", "no candidate passed all criteria",
         "", "not applicable", "null_or_boundary_result",
         "Reliable continuous manifold.", "No biological subtypes exist.", files["step3_manifest"],
         CORE_FIGURES[2], CORE_TABLES[3]),
        ("C04", "The leading neutral axis is strongly glycemic and replicated in test.",
         "4", "test", "neutral PC1 vs mean glucose Spearman rho", "-0.861", "",
         "FDR controlled in frozen pipeline", "validated_and_test_replicated",
         "A strongly replicated glycemic axis.", "The axis is a causal glycemic mechanism.",
         dirs["step4"] / "test_continuous_geometry_associations.csv", CORE_FIGURES[2], CORE_TABLES[3]),
        ("C05", "Static conditioning reorganizes participant geometry.",
         "4", "test", "distance Spearman; NN10 overlap; median cosine",
         "0.890; 0.394; 0.961", "", "not applicable",
         "validated_and_test_replicated", "Full and neutral geometries overlap but are not identical.",
         "Neutralization removes all static information.", files["step4_full_neutral"],
         CORE_FIGURES[2], CORE_TABLES[3]),
        ("C06", "Night and day representations differ substantially.",
         "4", "test", "distance Spearman; NN10 overlap; median cosine",
         "0.554; 0.158; 0.581", "", "not applicable",
         "validated_and_test_replicated", "Context-dependent reorganization replicated.",
         "Night and day define separate diseases.", files["step4_context"],
         CORE_FIGURES[3], CORE_TABLES[3]),
        ("C07", "Selected hs-CRP continuous associations replicated.",
         "4", "validation and test", "neutral PC3/PC4 and residual PC3 associations",
         "replicated directions/FDR status in Step 4", "", "FDR controlled",
         "validated_and_test_replicated", "Selected continuous hs-CRP associations replicated.",
         "hs-CRP information is incrementally predictive.", files["step4_replication"],
         CORE_FIGURES[2], CORE_TABLES[5]),
        ("C08", "Neutral-state hs-CRP incremental value did not transport.",
         "5", "validation and test", "delta R2",
         "validation +0.0070; test -0.0461", "validation -0.0334–0.0518; test -0.0883–-0.0054",
         "validation permutation q=0.006", "secondary_predictive_transport",
         "A small validation increment exceeded the shuffled-state null, but its bootstrap interval included zero and the effect reversed on test; it was not transported incremental value.", "Independent confirmation of hs-CRP increment.",
         files["step5_transport"], CORE_FIGURES[5], CORE_TABLES[5]),
        ("C09", "NT-proBNP and BUN/creatinine showed no transported neutral-state increment.",
         "5", "validation and test", "delta R2",
         "NT-proBNP -0.0780/+0.0194; BUN/creatinine -0.0380/-0.0477", "",
         "primary q=0.756 for both", "secondary_predictive_transport",
         "No transported increment.", "The hidden state contains no related biology.",
         files["step5_transport"], CORE_FIGURES[5], CORE_TABLES[5]),
        ("C10", "The exploratory frozen k=2 solution represents a glycemic tail.",
         "3B/4", "validation and test", "group transport, stability, TIR separation",
         "test 25/196; TIR 42.4% vs 98.2%; Cliff's delta 0.991", "",
         "exploratory q=2.37e-15 for TIR", "exploratory",
         "Exploratory glycemic-tail sensitivity.", "Confirmed clinical subtype.",
         files["step4_k2_char"], CORE_FIGURES[4], CORE_TABLES[4]),
    ]
    claim_cols = [
        "claim_id", "claim_text", "analysis_step", "dataset_split", "metric",
        "estimate", "confidence_interval", "fdr_status", "evidence_level",
        "allowed_wording", "disallowed_wording", "source_file", "source_figure",
        "source_table",
    ]
    claims_df = pd.DataFrame(claims, columns=claim_cols)
    claims_df.to_csv(out / "final_claims_evidence_table.csv", index=False)

    limitation_texts = [
        "Participant representations come from a model optimized for short-horizon glucose forecasting.",
        "Clinical variables already condition the full-profile state.",
        "Static neutralization changes the model input distribution.",
        "Neutralization is not equivalent to retraining without static features.",
        "External biomarkers are near but not necessarily concurrent with CGM.",
        "The recording period is approximately ten days.",
        "Physiological contexts have unequal coverage.",
        "No independent external dataset was used.",
        "The test set was examined before Step 5 predictive probes.",
        "Step 5 test results are secondary predictive transport.",
        "PCA and linear probes detect only certain forms of information.",
        "Absence of stable clusters does not prove biological subtypes do not exist.",
        "Exploratory k=2 groups must not be treated as clinical subtypes.",
        "Site and study-group effects may remain.",
        "Causal conclusions cannot be made.",
    ]
    limits = pd.DataFrame({
        "limitation_id": [f"L{i:02d}" for i in range(1, len(limitation_texts) + 1)],
        "limitation": limitation_texts,
        "implication": [
            "Interpret representation content in relation to the forecasting objective.",
            "Full-state associations can reflect directly supplied inputs.",
            "Neutral states may be off the training distribution.",
            "Neutralization isolates an input intervention, not an alternative trained model.",
            "Timing mismatch can attenuate or distort associations.",
            "Long-term phenotype stability is unknown.",
            "Context comparisons may reflect sampling as well as physiology.",
            "Generalizability beyond AI-READI is unknown.",
            "Test probes cannot be called untouched confirmation.",
            "Transport evidence is secondary.",
            "Nonlinear or localized information may be missed.",
            "Other representations or cohorts could contain subtypes.",
            "The grouping is an exploratory glycemic-tail discretization.",
            "Residual nuisance structure may persist.",
            "All findings are associational.",
        ],
        "mitigation_or_future_work": [
            "Add auxiliary clinical objectives.", "Use neutral and full states jointly with caveats.",
            "Train explicit static-free comparators.", "Retrain ablation models.",
            "Collect concurrent biomarkers.", "Use longer recordings.",
            "Balance or model context coverage.", "Use external cohorts.",
            "Prospectively freeze a new cohort.", "Treat estimates as transport only.",
            "Evaluate prespecified nonlinear probes externally.", "Test other frozen representations.",
            "Do not deploy k=2 as a phenotype.", "Model site/group hierarchically.",
            "Use causal designs where appropriate.",
        ],
    })
    limits.to_csv(out / "final_limitations_table.csv", index=False)

    methods = f"""# Methods

## Study cohort and split

AI-READI participants followed the saved participant split: {int(counts.get('train', 0))}
training, 239 validation, and 221 test participants. All analyses were participant
level and preserved the split. The canonical streaming model used 39 dynamic,
44 static (39 continuous/binary and five categorical), five time, and nine
scenario inputs. The frozen 128-dimensional post-update top-layer output supplied
to the 60-minute decoder was the hidden state.

## Static conditioning and reference intervention

The model used static initialization and feature-wise conditioning. Full-profile
replay used each participant's real static profile. Static-neutralized replay
replaced it with one common training-reference vector: scaler means for continuous
features, training prevalence for binary/one-hot features, and training modes for
categorical indices. Dynamic streams, timestamps, reset boundaries, and forecast
anchors were identical. This was an input intervention on the frozen model, not
retraining without static inputs.

## Replay, resets, burn-in, and aggregation

Canonical segments reset recurrent state at the saved reconstruction boundaries.
Step 1's 12-participant pilot proposed 855 minutes using a static-embedding proxy
for initialization drift. Step 2 evaluated eight prespecified candidates on all
239 validation participants and selected 0 minutes; the pilot value was not
privileged. Participant representations were dimensionwise medians of 15-minute
all-anchor states. Night was sleep-stage light/deep/REM; day was local 08:00–20:00,
not sleep. Odd/even recording days, retrieval, ICC, balanced anchors, and geometry
stability quantified reliability.

## Continuous geometry and clustering

Frozen scaling and PCA summarized full, neutral, glucose-residualized, night, and
day representations. Neutral PCA retained eight validation components to reach
90% variance. Glycemic residualization used cross-fitted ridge models on simple
glycemic summaries, with a final validation-only alpha of 10. Consensus k-means
and Ward sensitivity evaluated k=2–5 under predeclared cluster-size, fraction,
stability, consensus, odd/even, silhouette, and assignment rules. No solution
passed all criteria. Neutral k=2 (19/239, 7.95%) was frozen only as an exploratory
near-threshold sensitivity and transported by nearest frozen centroid.

## External phenotypes and contexts

Step 0 prespecified hs-CRP, NT-proBNP, and BUN/creatinine ratio. Steps 3–5 reused
the frozen records within ±180 days of CGM start; ±90 days was a sensitivity.
Transformations were log1p for nonnegative hs-CRP and NT-proBNP and natural log
for positive BUN/creatinine. No target entered the forecasting model.

## Clinical probes

The simple baseline contained demographics, site/group, mean glucose, SD, CV,
TIR/TAR/TBR, slope/range/valid hours, available wearable summaries and indicators,
and acquisition summaries. A secondary expanded baseline added static clinical,
medication, and missingness inputs already consumed by the forecaster. Ridge was
primary with alpha 10^-4 to 10^4 in half-log increments. Five repetitions of
nested 5×5 participant CV generated validation out-of-fold predictions; all
preprocessing was fit within training folds. Two thousand participant bootstraps
formed intervals. The exact three-test primary family used 1,000 hidden-row
permutations per target and Benjamini–Hochberg FDR. Final pipelines were tuned on
validation only and applied once to test. Because Step 4 had already inspected
test biomarker associations, Step 5 test results are secondary predictive
transport, not untouched confirmation.

## Reproducibility and leakage controls

Plans and transformations were frozen before target access; feature order and
hashes were checked; validation and test remained disjoint; no test tuning,
hidden-state regeneration, forecasting inference, outcome-based dimension
selection, or synthesis-stage testing occurred. Source hashes and claim-level
provenance are recorded in the final reproducibility manifest.
"""
    results = f"""# Results

## 1. Clinical and context feasibility

Step 0 identified three fixed external targets and adequate night/day context.
Within ±180 days, each target covered 235/239 validation and 217/221 test
participants. This feasibility result defined targets and coverage but did not
establish hidden-state clinical information.

## 2. Static-neutralized replay validation

In 12 validation pilot participants, the median full-neutral state L2 distance
was 24.5532 and median cosine was 0.9480. Mean absolute and terminal forecast
differences were 4.0301 and 6.8958 mg/dL. Full and neutral forecast MAE were
9.5900 and 11.3260 mg/dL. Static conditioning therefore materially affected
states and forecasts, although the pilot was small and neutralization may shift
inputs off the training distribution.

## 3. Full-validation burn-in and reliability

All 239 validation participants (392 segments) were represented at the selected
0-minute burn-in. Neutral odd/even cosine was {val_rel['median_within_cosine']:.4f}
(95% bootstrap CI {val_rel['within_cosine_ci_low']:.4f}–{val_rel['within_cosine_ci_high']:.4f});
top-1/top-5 retrieval were {100*val_rel['top1_retrieval']:.2f}%/
{100*val_rel['top5_retrieval']:.2f}%, and median ICC was
{val_rel['median_icc']:.4f}. Reliability measures recording-day reproducibility,
not clinical meaning.

## 4. Validation continuous-manifold finding

No full, neutral, or glucose-residualized candidate passed all predeclared
discrete-clustering criteria among 239 validation participants. Neutral k=2
narrowly missed size (19 versus 20) and fraction (7.95% versus 8%) rules.
The locked primary interpretation was a reliable continuous manifold; failure
to select clusters does not prove that biological subtypes cannot exist.

## 5. Test replication of representation reliability

Across 221 test participants, neutral odd/even cosine was
{test_rel['median_within_cosine']:.4f} (95% CI
{test_rel['within_cosine_ci_low']:.4f}–{test_rel['within_cosine_ci_high']:.4f}),
top-1/top-5 retrieval were {100*test_rel['top1_retrieval']:.2f}%/
{100*test_rel['top5_retrieval']:.2f}%, and median ICC was
{test_rel['median_icc']:.4f}. This replicated reliability within the same study,
not in an external cohort.

## 6. Replicated glycemic manifold

Frozen neutral PC1 correlated with mean glucose in test (Spearman rho −0.861;
n=221). The leading axis was therefore strongly glycemic and replicated under
the frozen pipeline, while PCA remains a linear summary rather than a causal
mechanism.

## 7. Static-conditioning effect on geometry

Test full-versus-neutral pairwise-distance correlation was
{full_neutral['pairwise_distance_spearman']:.3f}, NN10 overlap
{full_neutral['nn10_overlap']:.3f}, and median participant cosine
{full_neutral['median_cosine']:.3f}. Full and neutral structure overlapped but
participant neighborhoods changed; full-state clinical associations can partly
reflect directly supplied static variables.

## 8. Night-versus-day reorganization

In 221 test participants, night-versus-day distance correlation was
{test_context['distance_spearman']:.3f}, NN10 overlap
{test_context['nn10_overlap']:.3f}, and median cosine
{test_context['median_cosine']:.3f}. Context-dependent geometry replicated, but
unequal physiological coverage remains a possible contributor.

## 9. Continuous external clinical associations

Step 4 replicated selected hs-CRP associations with neutral PC3, neutral PC4,
and glucose-residualized neutral PC3. NT-proBNP and BUN/creatinine ratio lacked
comparable FDR-controlled replication. Association with a PCA coordinate does
not establish incremental prediction beyond conventional summaries.

## 10. Exploratory k=2 glycemic-tail result

Frozen centroids assigned 25/221 test participants to the smaller group and
196/221 to the larger; ambiguous assignments were 1.36%, odd/even agreement
97.74%, and balanced-anchor agreement 100%. Median TIR was 42.4% versus 98.2%
(Cliff.s delta 0.991). Manual verification reproduced TIR exactly from valid
raw CGM rows (cgm_count > 0 and nonmissing glucose), confirmed all 221 frozen
labels and all four study groups, and found similar median valid-row counts (2,610
versus 2,692.5). No study group was excluded, but the small group contained only
insulin-dependent and medication-controlled participants. The stable near-threshold
discretization is therefore an exploratory glycemic tail, not a clinical subtype.

## 11. Incremental clinical probes

For hs-CRP (235 validation; 217 test), baseline validation R² was 0.0585 and
baseline-plus-neutral R² 0.0655 (delta +0.0070; 95% CI −0.0334 to 0.0518;
permutation p=0.0020, primary q=0.0060). Test delta R² was −0.0461
(95% CI −0.0883 to −0.0054). A small validation increment exceeded the
shuffled-state permutation null, but its bootstrap interval included zero and the
effect reversed on test. It was therefore not considered transported incremental
value.
NT-proBNP delta R² was −0.0780 in validation and +0.0194 in test (opposite
direction). BUN/creatinine delta R² was −0.0380 in validation and −0.0477 in
test (no incremental value). Full-state improvements in validation were larger
for hs-CRP and BUN/creatinine but may reflect static conditioning. Residual,
night/day, and exploratory k=2 sensitivities did not alter the primary transport
conclusion. Test probes are secondary predictive transport because Step 4 had
already examined test biomarkers.

## 12. Overall result summary

The study supports a reproducible continuous glycemic manifold modified by
static conditioning and physiological context. Selected hs-CRP axis associations
replicated, but neutral-state incremental predictive value beyond simple
summaries did not transport. Null and discordant clinical probes are retained.
"""
    discussion = """# Discussion

The streaming hidden state was reproducible across recording-day partitions, but
reliability did not imply discrete phenotypes. No candidate passed all frozen
clustering rules, so the primary object is a continuous participant manifold.
Its leading structure was glycemic, consistent with a model trained for
short-horizon glucose forecasting.

Static clinical inputs materially changed states, forecasts, and neighborhoods.
Static neutralization is best understood as a controlled input intervention:
the frozen network was not retrained, and the common reference can move inputs
away from the training distribution. Full-state clinical prediction cannot be
read as purely dynamically learned physiology because clinical variables enter
that state directly.

Night and day geometry differed substantially, implying context-dependent
representation. This can reflect physiology, behavioral state, and unequal
sampling. The frozen k=2 result transported with high assignment stability, but
its exactly reproduced TIR separation and weak independent biomarker replication
identify it as a stable discretization of a glycemic extreme, not a clinical
subtype. No study group was excluded, but the small group contained only the two
diabetes-treatment study groups, reinforcing the composition caveat.

hs-CRP was the most promising external signal: selected continuous associations
replicated after neutralization and glycemic residualization. However, the
secondary predictive probe found a very small positive validation delta R² that
exceeded the shuffled-state null, while its bootstrap interval included zero and
the direction reversed on test transport. Thus the evidence does not support
transported incremental hs-CRP prediction beyond simple participant summaries.
Negative NT-proBNP and BUN/creatinine results are informative: a reliable state
optimized for glucose prediction need not encode broadly useful clinical
phenotypes.

Future work should use auxiliary clinical objectives, self-supervised
physiological representation learning, longer recordings, independent external
cohorts, and explicit disentanglement of glycemic and clinical factors. A
prospective cohort should freeze probe hypotheses before any target inspection.
The complete limitation register accompanies this report; findings are
associational and do not support causal conclusions.
"""
    conclusion = """# Conclusion

The frozen AI-READI glucose-forecasting model produced reproducible
participant-level hidden states after static neutralization. These states formed
a continuous manifold dominated by glycemic physiology, reorganized by real
static conditioning, and changed between night and day. No discrete subtype was
confirmed. The exploratory k=2 solution transported as a stable glycemic-tail
discretization.

Although selected hidden-state axes were associated with hs-CRP, the participant
representation did not provide transported incremental prediction beyond simple
CGM, wearable, demographic, and acquisition summaries. The final primary
conclusion is `reliable_glycemic_manifold_without_external_increment`.
"""
    study_summary = f"""# Final study summary

**Primary conclusion:** `{primary_conclusion}`.

Static conditioning changed hidden states and forecasts; static-neutralized
representations remained reliable in validation and test. The geometry was
continuous rather than discretely clustered, with a strongly replicated glycemic
axis and substantial night/day reorganization. The frozen exploratory k=2
assignment was stable but marked a glycemic tail.

Step 5 retained all three fixed targets. A small hs-CRP validation increment
exceeded the shuffled-state permutation null, but its bootstrap interval included
zero and the effect reversed on test; it was not transported incremental value;
NT-proBNP was directionally discordant and BUN/creatinine showed no increment.
Accordingly, no transported external clinical increment was established.
"""
    executive = """# Executive summary

The model learned a reproducible participant geometry, but mainly one organized
by glucose physiology—not discrete clinical subtypes. Replacing real static
profiles with a common training reference materially changed states and
forecasts, showing that static inputs help organize the geometry. Neutral states
remained reliable across validation and test, and night/day representations
differed.

The only near-cluster, frozen k=2, transported stably but separated an extreme
glycemic tail. hs-CRP was associated with selected continuous axes. A small
validation increment exceeded the shuffled-state null, but its bootstrap interval
included zero and the effect reversed on secondary test transport. Conventional
participant summaries were therefore sufficient for the tested external
phenotypes. External cohorts and clinically oriented objectives are required
before treating these states as broad clinical representations.
"""
    presentation = """# 5–7 minute presentation summary

## Slide 1 — Clinical question and hidden-state concept

**Message:** Does a glucose forecaster's internal state define reliable clinical phenotypes?

**Figure:** Figure 1.

- Hidden states summarize the preceding multimodal stream.
- The model predicts 60-minute glucose, not clinical diagnoses.
- We analyze participants, not timestamps.

**Caution:** Forecasting representations may be task-specific.

## Slide 2 — Static information is already injected

**Message:** Real clinical profiles directly condition the hidden state.

**Figure:** Figure 1.

- Full states mix dynamics with static input.
- Clinical associations can be circular.
- A common training reference provides an intervention.

**Caution:** Neutralization is not retraining.

## Slide 3 — Static-neutralization design

**Message:** Dynamic streams are held fixed while only the static profile changes.

**Figure:** Figure 1.

- Identical timestamps, resets, and anchors.
- Median state L2 difference: 24.55.
- Mean absolute forecast difference: 4.03 mg/dL.

**Caution:** The common profile may be off-distribution.

## Slide 4 — Reliable participant representations

**Message:** Neutral participant summaries reproduce across odd/even days and test.

**Figure:** Figure 2.

- Validation/test cosine: 0.855/0.865.
- Top-1 retrieval: 69.5%/70.4%.
- Median ICC: 0.873/0.886.

**Caution:** Reliability is not clinical validity.

## Slide 5 — Continuous glycemic manifold

**Message:** No cluster passed every rule; the leading axis is strongly glycemic.

**Figure:** Figure 3.

- Neutral k=2 missed the size rule by one participant.
- Test PC1–mean glucose rho: −0.861.
- Continuous geometry is the primary result.

**Caution:** No stable cluster does not disprove all biological subtypes.

## Slide 6 — Full/neutral and night/day

**Message:** Static conditioning and physiological context reorganize similarity.

**Figure:** Figure 4.

- Full/neutral distance correlation: 0.890.
- Night/day distance correlation: 0.554.
- Night/day NN10 overlap: 0.158.

**Caution:** Context coverage is unequal.

## Slide 7 — Exploratory k=2 glycemic tail

**Message:** Frozen k=2 transports, but it discretizes a glycemic extreme.

**Figure:** Figure 5.

- Test groups: 25 and 196.
- Odd/even agreement: 97.7%.
- Median TIR: 42.4% versus 98.2%.

**Caution:** This is not a clinical subtype.

## Slide 8 — Clinical probe results

**Message:** hs-CRP's small validation increment did not transport.

**Figure:** Figure 6.

- hs-CRP validation delta R² +0.007; test −0.046.
- Validation permutation q=0.006, but the bootstrap CI included zero and test reversed.
- NT-proBNP directions opposed; BUN/creatinine showed no increment.

**Caution:** Test probes are secondary predictive transport.

## Slide 9 — Conclusion and future work

**Message:** Reliable glycemic manifold; no transported external increment.

**Figure:** Figures 3 and 6.

- Static and context effects are real features of the geometry.
- k=2 is an exploratory glycemic tail.
- Use clinical objectives, longer recordings, and external cohorts.

**Caution:** No causal or subtype claim is supported.
"""
    write(out / "final_thesis_methods.md", methods)
    write(out / "final_thesis_results.md", results)
    write(out / "final_thesis_discussion.md", discussion)
    write(out / "final_thesis_conclusion.md", conclusion)
    write(out / "final_study_summary.md", study_summary)
    write(out / "final_executive_summary.md", executive)
    write(out / "final_presentation_summary.md", presentation)
    complete = "\n\n".join([methods, results, discussion, conclusion])
    write(out / "final_thesis_section_complete.md", complete)
    tex = r"""\section{Methods}
""" + latex_escape(methods.replace("# Methods\n", "")) + r"""

\section{Results}
""" + latex_escape(results.replace("# Results\n", "")) + r"""

\section{Discussion}
""" + latex_escape(discussion.replace("# Discussion\n", "")) + r"""

\section{Conclusion}
""" + latex_escape(conclusion.replace("# Conclusion\n", ""))
    write(out / "final_thesis_section_complete.tex", tex)

    figure_manifest = pd.DataFrame([
        {"figure_id": i + 1, "final_file": f"final_figures/{name}",
         "title": [
             "Study design and static-neutralization intervention",
             "Static conditioning and representation reliability",
             "Continuous participant geometry",
             "Context dependence",
             "Exploratory near-threshold k=2 glycemic-tail sensitivity",
             "Incremental clinical probe results",
         ][i],
         "source_file": str(figure_sources[i]) if figure_sources[i] else
             ("Step 1 manifest + Step 2/4 reliability summaries"),
         "generation": "regenerated from saved metrics and validation-fitted transforms; no fitting",
         "evidence_role": ["methodological", "primary", "primary", "primary",
                           "exploratory", "secondary predictive transport"][i]}
        for i, name in enumerate(CORE_FIGURES)
    ])
    figure_manifest.to_csv(out / "final_figure_manifest.csv", index=False)
    table_manifest = pd.DataFrame([
        {"table_id": i + 1, "final_file": f"final_tables/{name}",
         "title": [
             "Study design and participant counts", "Static-neutralization effects",
             "Representation reliability", "Continuous geometry",
             "Exploratory k=2 result", "External clinical information",
         ][i],
         "source_files": [
             f"{split_path}; {files['step5_cohort']}",
             str(files["step1_manifest"]),
             f"{files['step2_reliability']}; {files['step4_reliability']}",
             f"{dirs['step3'] / 'continuous_geometry_associations.csv'}; "
             f"{files['step4_full_neutral']}; {files['step4_context']}",
             f"{files['step4_k2_metrics']}; {files['step4_k2_char']}",
             f"{files['step4_replication']}; {files['step5_transport']}",
         ][i]}
        for i, name in enumerate(CORE_TABLES)
    ])
    table_manifest.to_csv(out / "final_table_manifest.csv", index=False)

    report = f"""# Step 6 final synthesis report

## Objective

Consolidate frozen Steps 0–5 without new fitting, clustering, target selection,
or hypothesis testing.

## Scientific hierarchy

Primary findings remain static-conditioning influence, reliable neutral
representations, a continuous glycemic manifold, and context dependence.
Selected hs-CRP associations and clinical probes are secondary. The frozen k=2
result remains exploratory and is interpreted as a glycemic tail.

## Final decision

`{primary_conclusion}`

## Outputs

- Six core figures in `final_figures/`.
- Six core tables in `final_tables/`.
- Complete Methods, Results, Discussion, Conclusion, Markdown thesis section,
  and LaTeX thesis section.
- Claims/evidence, limitation, figure, table, and reproducibility audits.
- A manual raw-CGM TIR verification with exact saved/recomputed agreement.
- Revised figures separate measurement units, label PCA variance/color, show
  all/night/day geometry, characterize glycemic k=2 distributions, and display
  validation/test probe intervals together.

## Step 5 interpretation

A small hs-CRP validation increment exceeded the shuffled-state permutation null,
but its bootstrap interval included zero and the effect reversed on test. It was
therefore not considered transported incremental value. NT-proBNP was discordant
and BUN/creatinine showed no incremental value. This is secondary predictive
transport; it is not an untouched confirmation.

## Limitations

All 15 required limitations appear in `final_limitations_table.csv` and the
Discussion. No independent external cohort was analyzed.
"""
    write(out / "step6_report.md", report)

    source_hashes = {key: sha256(path) for key, path in files.items()}
    reproducibility = {
        "status": "QC_PENDING",
        "run_id": rid,
        "run_directory": str(out),
        "seed": a.seed,
        "no_new_analysis": True,
        "operations": ["read saved summaries", "apply validation-fitted PCA transforms for plotting",
                       "recompute saved TIR definition from valid raw CGM rows for QC",
                       "render consolidated figures/tables/text", "hash and QC"],
        "prohibited_operations_absent": [
            "model fitting", "clustering", "target selection",
            "hypothesis testing", "hidden-state extraction", "forecast inference",
        ],
        "source_paths": {key: str(path) for key, path in files.items()},
        "source_hashes": source_hashes,
        "script": str(Path(__file__).resolve()),
        "script_hash": sha256(Path(__file__)),
        "figure_revision_helper": str(Path(__file__).with_name("hidden_state_final_figure_revisions.py")),
        "figure_revision_helper_hash": sha256(Path(__file__).with_name("hidden_state_final_figure_revisions.py")),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "pandas": pd.__version__, "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    dump(out / "final_reproducibility_manifest.json", reproducibility)

    # Independent synthesis QC before latest pointer.
    source_claims_exist = all(Path(x).exists() for x in claims_df["source_file"])
    figure_decode = {}
    for name in CORE_FIGURES:
        with Image.open(figures / name) as im:
            im.verify()
        figure_decode[name] = True
    narrative = "\n".join(
        (out / name).read_text()
        for name in [
            "final_study_summary.md", "final_thesis_methods.md",
            "final_thesis_results.md", "final_thesis_discussion.md",
            "final_thesis_conclusion.md", "step6_report.md",
        ]
    ).lower()
    checks = {
        "step5_qc_complete_and_authorized": True,
        "required_files_present": all((out / name).exists() for name in REQUIRED_OUTPUTS
                                      if name != "step6_manifest.json"),
        "six_figures_present": len(list(figures.glob("*.png"))) == 6,
        "six_tables_present": len(list(tables.glob("*.csv"))) == 6,
        "all_figures_decode": all(figure_decode.values()),
        "six_figures_are_visually_distinct_files": len({sha256(figures / name) for name in CORE_FIGURES}) == 6,
        "manual_tir_verification_qc_complete": tir_verification["status"] == "QC_COMPLETE",
        "manual_tir_exact_raw_agreement": tir_verification["maximum_saved_vs_recomputed_tir_absolute_difference"] == 0.0,
        "claim_sources_exist": source_claims_exist,
        "claim_columns_exact": list(claims_df.columns) == claim_cols,
        "fifteen_limitations": len(limits) >= 15,
        "no_model_artifacts": not list(out.rglob("*.joblib"))
                              and not list(out.rglob("*.pt")),
        "exploratory_role_explicit": "exploratory glycemic tail" in narrative,
        "secondary_transport_role_explicit": "secondary predictive transport" in narrative,
        "no_increment_from_pca_only": "not establish incremental prediction" in narrative,
        "hsCRP_bootstrap_permutation_transport_wording": all(phrase in narrative for phrase in [
            "small validation increment exceeded the", "bootstrap interval included zero",
            "effect reversed on test", "not considered transported incremental"
        ]),
        "null_results_retained": "nt-probnp" in narrative and "bun/creatinine" in narrative,
        "primary_conclusion_frozen": final_decision["primary_conclusion"] == primary_conclusion,
    }
    if not all(checks.values()):
        raise RuntimeError({k: v for k, v in checks.items() if not v})
    reproducibility["status"] = "QC_COMPLETE"
    reproducibility["qc_checks"] = checks
    reproducibility["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump(out / "final_reproducibility_manifest.json", reproducibility)
    log.info("Claims, language, figures, tables, and source provenance passed QC")

    output_hashes = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "step6_manifest.json":
            output_hashes[str(path.relative_to(out))] = sha256(path)
    manifest = {
        "status": "QC_COMPLETE",
        "run_id": rid,
        "run_directory": str(out),
        "primary_conclusion": primary_conclusion,
        "secondary_conclusions": secondary,
        "input_paths": {key: str(path) for key, path in files.items()},
        "input_hashes": source_hashes,
        "output_hashes": output_hashes,
        "figures": CORE_FIGURES,
        "tables": CORE_TABLES,
        "claims_evidence_audit": "QC_COMPLETE",
        "reproducibility_status": "QC_COMPLETE",
        "no_new_analysis": True,
        "warnings": [
            "Step 5 test results are secondary predictive transport.",
            "The exploratory k=2 result is a glycemic-tail sensitivity.",
            "No independent external cohort was used.",
        ],
        "blockers": [],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(out / "step6_manifest.json", manifest)
    log.info("Step 6 QC_COMPLETE")
    # Log append changes the log hash; refresh it once before publishing latest.
    for handler in logging.getLogger().handlers:
        handler.flush()
    manifest["output_hashes"]["step6_run.log"] = sha256(out / "step6_run.log")
    manifest["output_hashes"]["final_reproducibility_manifest.json"] = sha256(
        out / "final_reproducibility_manifest.json")
    dump(out / "step6_manifest.json", manifest)
    latest = root / "latest"
    tmp_latest = root / ".latest.tmp"
    if tmp_latest.exists() or tmp_latest.is_symlink():
        tmp_latest.unlink()
    tmp_latest.symlink_to(rid)
    os.replace(tmp_latest, latest)
    print(json.dumps({
        "status": "QC_COMPLETE", "run_directory": str(out),
        "primary_conclusion": primary_conclusion,
    }, indent=2))


if __name__ == "__main__":
    main()
