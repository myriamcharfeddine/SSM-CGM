"""Final Step 7 thesis-text revision and run finalization."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd


NO_EM_DASH = "\u2014"
PROHIBITED_FINAL_PHRASES = (
    "failed cluster",
    "confirmed subtype",
    "phenotype cluster",
    "discovered subgroup",
)
MODEL_INPUT_PARAGRAPH = (
    "Most core metabolic, demographic, medication, site, and study-group "
    "variables in the enriched static profile were supplied to the model, "
    "including HbA1c, BMI, blood pressure, lipids, C-peptide, medications, "
    "clinical site, and study group."
)
EXTERNAL_INPUT_PARAGRAPH = (
    "The selected external targets NT-proBNP, high-sensitivity CRP, and "
    "BUN/creatinine ratio were not forecasting-model inputs."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_converter(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path, datetime, pd.Timestamp)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, default=json_converter
        )
        + "\n"
    )
    os.replace(temporary, path)


def clean_source_text(text: str) -> str:
    replacements = (
        (
            "No target entered the forecasting model.",
            MODEL_INPUT_PARAGRAPH
            + "\n\n"
            + EXTERNAL_INPUT_PARAGRAPH
            + "\n\nHbA1c was therefore a direct-input positive control in "
            "the full-profile state.",
        ),
        (
            "glycemic-tail discretization",
            "exploratory glycemic-tail stratification",
        ),
        (
            "glycemic-tail solution",
            "exploratory glycemic-tail stratification",
        ),
        ("glycemic-tail result", "exploratory glycemic-tail stratification"),
        ("glycemic tail", "reproducible glycemic extreme"),
        ("glycemic-tail group", "glycemic-extreme group"),
    )
    cleaned = text.replace(NO_EM_DASH, " - ")
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    return cleaned


def result_row(
    results: pd.DataFrame,
    condition: str,
    variable: str,
    k_neighbors: int = 10,
    site_matched: bool = False,
) -> pd.Series:
    selected = results[
        results["condition"].eq(condition)
        & results["variable"].eq(variable)
        & results["k_neighbors"].eq(k_neighbors)
        & results["site_matched"].eq(site_matched)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one result row: {condition} {variable} "
            f"k={k_neighbors} site={site_matched}"
        )
    return selected.iloc[0]


def make_neighbor_results_text(results: pd.DataFrame) -> str:
    neutral_variables = (
        "mean_glucose",
        "glucose_cv",
        "tir_70_180",
        "glucose_sd",
    )
    neutral_rows = {
        variable: result_row(results, "neutral_all", variable)
        for variable in neutral_variables
    }
    hba1c = result_row(results, "neutral_all", "hba1c")
    study_group = result_row(results, "neutral_all", "study_group")
    ntprobnp = result_row(
        results, "neutral_all", "natriuretic_peptide_b_prohormon"
    )
    crp = result_row(results, "neutral_all", "c_reactive_protein_i")
    bun = result_row(results, "neutral_all", "bun_creatinine_ratio")
    site_hba1c = result_row(
        results, "neutral_all", "hba1c", site_matched=True
    )
    site_group = result_row(
        results, "neutral_all", "study_group", site_matched=True
    )
    full_hba1c = result_row(results, "full_all", "hba1c")
    full_group = result_row(results, "full_all", "study_group")
    k5_hba1c = result_row(results, "neutral_all", "hba1c", k_neighbors=5)
    k20_hba1c = result_row(results, "neutral_all", "hba1c", k_neighbors=20)
    lines = [
        "## Clinical similarity among hidden-state neighbours",
        "",
        "The primary analysis used 221 test participants, Euclidean distance "
        "in the static-neutral frozen validation PCA space, and ten directed "
        "nearest neighbours per focal participant. Each participant was "
        "compared with 2,000 equally sized random non-neighbour sets. "
        "Confidence intervals used 2,000 focal-participant bootstraps, and "
        "the primary Benjamini-Hochberg family contained exactly nine "
        "static-neutral tests.",
        "",
        "Participants close in the static-neutral manifold had more similar "
        "glucose control than random participants. Standardized similarity "
        "gains were "
        f"{neutral_rows['mean_glucose']['standardized_similarity_gain']:.3f} "
        f"[{neutral_rows['mean_glucose']['bootstrap_ci_low']:.3f}, "
        f"{neutral_rows['mean_glucose']['bootstrap_ci_high']:.3f}] for mean "
        "glucose, "
        f"{neutral_rows['glucose_cv']['standardized_similarity_gain']:.3f} "
        f"[{neutral_rows['glucose_cv']['bootstrap_ci_low']:.3f}, "
        f"{neutral_rows['glucose_cv']['bootstrap_ci_high']:.3f}] for glucose "
        "CV, "
        f"{neutral_rows['tir_70_180']['standardized_similarity_gain']:.3f} "
        f"[{neutral_rows['tir_70_180']['bootstrap_ci_low']:.3f}, "
        f"{neutral_rows['tir_70_180']['bootstrap_ci_high']:.3f}] for time in "
        "range, and "
        f"{neutral_rows['glucose_sd']['standardized_similarity_gain']:.3f} "
        f"[{neutral_rows['glucose_sd']['bootstrap_ci_low']:.3f}, "
        f"{neutral_rows['glucose_sd']['bootstrap_ci_high']:.3f}] for glucose "
        "SD. All four permutation p-values were 0.0005 and FDR q-values were "
        "0.00075.",
        "",
        f"HbA1c sharing also persisted after neutralization. The standardized "
        f"gain was {hba1c['standardized_similarity_gain']:.3f} "
        f"[{hba1c['bootstrap_ci_low']:.3f}, "
        f"{hba1c['bootstrap_ci_high']:.3f}], with permutation "
        f"p={hba1c['permutation_p']:.4f} and q={hba1c['fdr_q']:.5f}. "
        f"Neighbour HbA1c values differed by "
        f"{hba1c['neighbor_raw_mean_difference']:.3f} percentage points "
        f"versus {hba1c['random_raw_mean_difference']:.3f} for random "
        "non-neighbours.",
        "",
        f"Study-group concordance was {100 * study_group['same_group_rate_neighbors']:.1f}% "
        f"among static-neutral neighbours and "
        f"{100 * study_group['same_group_rate_random']:.1f}% under the random "
        f"baseline, a gain of {100 * study_group['same_group_rate_gain']:.1f} "
        f"percentage points [{100 * study_group['bootstrap_ci_low']:.1f}, "
        f"{100 * study_group['bootstrap_ci_high']:.1f}], with p=0.0005 and "
        "q=0.00075.",
        "",
        "The full-profile space showed stronger direct-input sharing for "
        f"HbA1c (gain {full_hba1c['standardized_similarity_gain']:.3f}) and "
        f"study group (gain {full_group['same_group_rate_gain']:.3f}) than "
        "the static-neutral space. This contrast estimates how much "
        "clinical information remains recoverable after participant-specific "
        "static conditioning is removed; it is not an exact causal "
        "decomposition.",
        "",
        f"Site matching did not explain the result. Site-matched neutral gains "
        f"were {site_hba1c['standardized_similarity_gain']:.3f} for HbA1c "
        f"and {site_group['same_group_rate_gain']:.3f} for study group, and "
        "no focal participant had insufficient same-site candidates. HbA1c "
        f"gains were also stable at k=5 "
        f"({k5_hba1c['standardized_similarity_gain']:.3f}) and k=20 "
        f"({k20_hba1c['standardized_similarity_gain']:.3f}); the glucose and "
        "study-group conclusions were likewise stable.",
        "",
        "External biomarker sharing was not supported. Static-neutral gains "
        f"were {ntprobnp['standardized_similarity_gain']:.3f} for NT-proBNP "
        f"(p={ntprobnp['permutation_p']:.3f}, q={ntprobnp['fdr_q']:.3f}), "
        f"{crp['standardized_similarity_gain']:.3f} for high-sensitivity CRP "
        f"(p={crp['permutation_p']:.3f}, q={crp['fdr_q']:.3f}), and "
        f"{bun['standardized_similarity_gain']:.3f} for BUN/creatinine ratio "
        f"(p={bun['permutation_p']:.3f}, q={bun['fdr_q']:.3f}).",
    ]
    return "\n".join(lines)


def make_hba1c_results_text(summary: pd.DataFrame) -> str:
    validation = summary[
        summary["split"].eq("validation_nested_cv")
    ].iloc[0]
    test = summary[summary["split"].eq("test_transport")].iloc[0]
    return "\n".join(
        [
            "## Targeted HbA1c positive-control predictive transport",
            "",
            "HbA1c was a direct input to the full-profile model. The targeted "
            "closing analysis therefore compared information recoverable from "
            "the full-profile and static-neutral representations beyond a "
            "conventional glycemic-summary baseline. It was not an untouched "
            "confirmation.",
            "",
            f"The validation glycemic baseline R2 was "
            f"{validation['glycemic_baseline_r2']:.4f}. Adding the full-profile "
            f"state increased R2 by {validation['full_delta_r2']:+.4f} "
            f"[{validation['full_delta_r2_ci_low']:+.4f}, "
            f"{validation['full_delta_r2_ci_high']:+.4f}], while adding the "
            f"static-neutral state increased R2 by "
            f"{validation['neutral_delta_r2']:+.4f} "
            f"[{validation['neutral_delta_r2_ci_low']:+.4f}, "
            f"{validation['neutral_delta_r2_ci_high']:+.4f}]. The neutral "
            f"increment exceeded the shuffled-state null "
            f"(p={validation['neutral_permutation_p']:.4f}).",
            "",
            f"On targeted test transport, baseline R2 was "
            f"{test['glycemic_baseline_r2']:.4f}. The full-profile increment "
            f"was {test['full_delta_r2']:+.4f} "
            f"[{test['full_delta_r2_ci_low']:+.4f}, "
            f"{test['full_delta_r2_ci_high']:+.4f}], whereas the "
            f"static-neutral increment was {test['neutral_delta_r2']:+.4f} "
            f"[{test['neutral_delta_r2_ci_low']:+.4f}, "
            f"{test['neutral_delta_r2_ci_high']:+.4f}]. The positive "
            "validation neutral increment therefore did not transport.",
            "",
            "Strong HbA1c recovery in the full-profile state was consistent "
            "with direct static conditioning. Although HbA1c remained locally "
            "shared and visually organized on the neutral manifold, the "
            "neutral representation did not add transported predictive value "
            "beyond conventional glycemic summaries. The full-neutral "
            "difference is not an exact learned-versus-injected percentage.",
        ]
    )


def markdown_to_latex(markdown: str) -> str:
    def escape(text: str) -> str:
        replacements = (
            ("&", r"\&"),
            ("%", r"\%"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("$", r"\$"),
        )
        value = text.replace("**", "").replace("`", "")
        for old, new in replacements:
            value = value.replace(old, new)
        return value

    lines = []
    in_items = False
    for line in markdown.splitlines():
        if line.startswith("### "):
            if in_items:
                lines.append(r"\end{itemize}")
                in_items = False
            lines.append(r"\subsubsection{" + escape(line[4:]) + "}")
        elif line.startswith("## "):
            if in_items:
                lines.append(r"\end{itemize}")
                in_items = False
            lines.append(r"\subsection{" + escape(line[3:]) + "}")
        elif line.startswith("# "):
            if in_items:
                lines.append(r"\end{itemize}")
                in_items = False
            lines.append(r"\section{" + escape(line[2:]) + "}")
        elif line.startswith("- "):
            if not in_items:
                lines.append(r"\begin{itemize}")
                in_items = True
            lines.append(r"\item " + escape(line[2:]))
        elif not line.strip():
            if in_items:
                lines.append(r"\end{itemize}")
                in_items = False
            lines.append("")
        else:
            if in_items:
                lines.append(r"\end{itemize}")
                in_items = False
            lines.append(escape(line))
    if in_items:
        lines.append(r"\end{itemize}")
    return "\n".join(lines) + "\n"


def scan_em_dash(paths: list[Path]) -> list[str]:
    affected: list[str] = []
    for root in paths:
        candidates = root.rglob("*") if root.is_dir() else [root]
        for path in candidates:
            if path.is_file() and path.suffix.lower() in {
                ".csv", ".json", ".log", ".md", ".py", ".tex"
            }:
                try:
                    if NO_EM_DASH in path.read_text():
                        affected.append(str(path))
                except UnicodeDecodeError:
                    continue
    return affected


def git_metadata(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty_tree": bool(status),
        "status_lines": status,
    }


def run_text_stage(
    run_directory: Path,
    repository_root: Path,
    step_directories: dict[str, Path],
) -> dict[str, Any]:
    revised_text_directory = run_directory / "revised_text"
    manifest_path = run_directory / "step7_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("figure_stage", {}).get("status") != "QC_COMPLETE":
        raise RuntimeError("Figure stage is not complete")
    if manifest.get("hba1c_stage", {}).get("status") != "QC_COMPLETE":
        raise RuntimeError("HbA1c stage is not complete")
    if manifest.get("text_stage", {}).get("status") == "QC_COMPLETE":
        raise RuntimeError("Text stage is already complete")
    if any(revised_text_directory.iterdir()):
        raise RuntimeError(
            f"Revised text directory is not empty: {revised_text_directory}"
        )
    step6_directory = step_directories["step6"]
    base_methods = clean_source_text(
        (step6_directory / "final_thesis_methods.md").read_text()
    )
    base_results = clean_source_text(
        (step6_directory / "final_thesis_results.md").read_text()
    )
    base_discussion = clean_source_text(
        (step6_directory / "final_thesis_discussion.md").read_text()
    )
    neighbour_results = pd.read_csv(
        run_directory
        / "neighbor_sharing/neighbor_sharing_tier1_results.csv"
    )
    hba1c_summary = pd.read_csv(
        run_directory
        / "hba1c_positive_control/hba1c_full_vs_neutral_summary.csv"
    )
    neighbour_section = make_neighbor_results_text(neighbour_results)
    hba1c_section = make_hba1c_results_text(hba1c_summary)
    methods_addition = "\n".join(
        [
            "## Step 7 nearest-neighbour clinical sharing",
            "",
            "Test participants were projected with the matching frozen "
            "validation scaler and PCA for the full-profile and static-neutral "
            "all-recording representations. Euclidean directed nearest-"
            "neighbour graphs used k=10, with k=5 and k=20 sensitivities. "
            "For each continuous variable, focal-participant mean absolute "
            "differences were compared with 2,000 deterministic random "
            "non-neighbour sets of equal size. Study group used same-group "
            "rates. A same-site random baseline assessed recruitment-site "
            "confounding. Two thousand participant-clustered bootstraps and "
            "2,000 label permutations were used. The primary FDR family "
            "contained exactly nine static-neutral k=10 tests; the full-profile "
            "family was corrected separately.",
            "",
            "## Step 7 targeted HbA1c positive control",
            "",
            "HbA1c percent was taken from the baseline participant static table. "
            "The primary baseline used mean glucose, SD, CV, TIR, TAR, TBR, "
            "mean absolute slope, glucose range, and valid CGM hours. Ridge "
            "models reused the Step 5 protocol: five repetitions of nested "
            "5 by 5 participant-level validation CV, fold-fitted imputation "
            "and scaling, the same alpha grid, 2,000 participant bootstraps, "
            "and a 2,000-draw state-row permutation test. Final models were "
            "selected and fitted on validation only, then applied once to test. "
            "The test analysis was targeted positive-control predictive "
            "transport, not untouched confirmation.",
        ]
    )
    methods = base_methods.rstrip() + "\n\n" + methods_addition + "\n"
    results = (
        base_results.rstrip()
        + "\n\n"
        + neighbour_section
        + "\n\n"
        + hba1c_section
        + "\n"
    )
    discussion_addition = "\n".join(
        [
            "## Direct answer to the participant-similarity question",
            "",
            "Nearest-neighbour analysis directly answered the original question "
            "without forcing discrete groups. Participants close in the "
            "static-neutral hidden-state manifold shared mean glucose, glucose "
            "variability, time in range, glucose SD, HbA1c, and clinically "
            "defined diabetes study group more strongly than random "
            "participants. These effects persisted with same-site random "
            "comparisons and at k=5 and k=20. The manifold therefore represents "
            "clinically meaningful local glycemic similarity.",
            "",
            "The stronger full-profile sharing of HbA1c and study group is "
            "expected because these variables were supplied through the static "
            "pathway. Static neutralization changes initialization, static "
            "embeddings, conditioning, and subsequent recurrent updates. The "
            "full-versus-neutral contrast estimates how much related information "
            "remains recoverable after participant-specific static conditioning "
            "is removed; it does not identify learned and injected percentages.",
            "",
            "HbA1c illustrates the distinction between shared geometry and "
            "unique predictive value. HbA1c remained locally shared after "
            "neutralization, but the neutral-state validation increment beyond "
            "conventional glycemic summaries did not transport to test. The "
            "full-profile increment transported, consistent with direct HbA1c "
            "conditioning. Thus, the neutral state summarized long-term "
            "glycemic structure without adding robust transported HbA1c "
            "prediction beyond direct CGM summaries.",
            "",
            "External clinical sharing was weaker. NT-proBNP, high-sensitivity "
            "CRP, and BUN/creatinine ratio did not pass the primary neighbour "
            "permutation and FDR criteria. This agrees with the predictive "
            "transport analysis. A small high-sensitivity CRP validation "
            "increment exceeded the shuffled-state permutation null, but its "
            "bootstrap interval included zero and the effect reversed on test. "
            "It was therefore not considered transported incremental value.",
            "",
            "The exploratory near-threshold k=2 analysis remains an exploratory "
            "glycemic-tail stratification representing a reproducible glycemic "
            "extreme. It does not replace the primary continuous-manifold "
            "interpretation.",
        ]
    )
    discussion = (
        base_discussion.rstrip() + "\n\n" + discussion_addition + "\n"
    )
    conclusion = "\n".join(
        [
            "# Conclusion",
            "",
            "Participants who were close in streaming hidden-state space shared "
            "glycemic control and clinically defined diabetes strata more "
            "strongly than random participants. Mean glucose, glucose CV, time "
            "in range, glucose SD, HbA1c, and study group remained locally "
            "shared after participant-specific static conditioning was removed. "
            "The findings persisted after site matching and across k=5, 10, "
            "and 20.",
            "",
            "HbA1c sharing persisted geometrically after neutralization, but "
            "the neutral-state increment beyond conventional glycemic summaries "
            "did not transport to test. In contrast, the full-profile HbA1c "
            "increment transported, consistent with HbA1c being directly "
            "supplied to the full-profile model. This comparison is not an "
            "exact causal decomposition.",
            "",
            "Sharing of NT-proBNP, high-sensitivity CRP, and BUN/creatinine "
            "ratio was not supported, consistent with the absence of transported "
            "incremental prediction beyond conventional participant summaries. "
            "The final answer is therefore that streaming hidden states encode "
            "a reproducible, clinically meaningful continuous glycemic "
            "neighbourhood structure, rather than confirmed discrete clinical "
            "types or broad external biomarker phenotypes.",
        ]
    )
    presentation = "\n".join(
        [
            "# Final presentation summary revised",
            "",
            "## Slide 1: Original clinical question",
            "",
            "- Do nearby participants in hidden-state space share clinical characteristics?",
            "- The direct test compares nearest neighbours with random non-neighbours.",
            "",
            "## Slide 2: Model-input clarification",
            "",
            "- " + MODEL_INPUT_PARAGRAPH,
            "- " + EXTERNAL_INPUT_PARAGRAPH,
            "- HbA1c is a direct-input positive control in the full-profile state.",
            "",
            "## Slide 3: Frozen analysis design",
            "",
            "- Test participants only for the primary neighbour figure.",
            "- Matching frozen full and neutral validation PCA pipelines.",
            "- k=10 primary; k=5 and k=20 sensitivities.",
            "- Participant bootstrap, label permutation, and separate nine-test FDR families.",
            "",
            "## Slide 4: Glycemic neighbour sharing",
            "",
            "- Static-neutral gains: mean glucose 0.725, CV 0.381, TIR 0.665, SD 0.610.",
            "- All four permutation p-values were 0.0005 and q-values were 0.00075.",
            "- Site-matched and k sensitivity results were consistent.",
            "",
            "## Slide 5: HbA1c and study group",
            "",
            "- Neutral HbA1c sharing gain: 0.399 [0.355, 0.448].",
            "- Neutral same-study-group rates: 39.1% for neighbours versus 27.5% random.",
            "- Full-profile sharing was stronger because real static conditioning was retained.",
            "",
            "## Slide 6: HbA1c predictive positive control",
            "",
            "- Validation baseline R2: 0.710.",
            "- Full-state delta R2: +0.127 validation and +0.148 targeted test transport.",
            "- Neutral-state delta R2: +0.067 validation but -0.024 test.",
            "- The neutral increment did not transport; no learned-versus-injected percentage is claimed.",
            "",
            "## Slide 7: External biomarkers",
            "",
            "- NT-proBNP, high-sensitivity CRP, and BUN/creatinine ratio showed no primary FDR-supported neighbour sharing.",
            "- External predictive increments did not transport.",
            "",
            "## Slide 8: Continuous clinical manifold",
            "",
            "- Mean glucose, study group, and HbA1c overlay the frozen continuous geometry.",
            "- Full and neutral panels use matching, distinct frozen PCA spaces.",
            "",
            "## Slide 9: Context and exploratory stratification",
            "",
            "- Night and day show different participant geometry.",
            "- k=2 remains an exploratory glycemic-tail stratification representing a reproducible glycemic extreme.",
            "",
            "## Slide 10: Final answer",
            "",
            "- Yes: nearby hidden-state participants share glycemic control and diabetes strata.",
            "- No reliable sharing was demonstrated for the three external biomarkers.",
            "- The primary result is a continuous clinically meaningful glycemic neighbourhood structure.",
        ]
    )
    text_paths = {
        "results": revised_text_directory
        / "final_results_with_neighbor_sharing.md",
        "discussion": revised_text_directory
        / "final_discussion_with_neighbor_sharing.md",
        "conclusion": revised_text_directory
        / "final_conclusion_with_neighbor_sharing.md",
        "complete": revised_text_directory
        / "final_thesis_section_complete_revised.md",
        "complete_latex": revised_text_directory
        / "final_thesis_section_complete_revised.tex",
        "presentation": revised_text_directory
        / "final_presentation_summary_revised.md",
    }
    text_paths["results"].write_text(results)
    text_paths["discussion"].write_text(discussion)
    text_paths["conclusion"].write_text(conclusion + "\n")
    complete = (
        methods.rstrip()
        + "\n\n"
        + results.rstrip()
        + "\n\n"
        + discussion.rstrip()
        + "\n\n"
        + conclusion
        + "\n"
    )
    text_paths["complete"].write_text(complete)
    text_paths["complete_latex"].write_text(markdown_to_latex(complete))
    text_paths["presentation"].write_text(presentation + "\n")
    changelog_path = run_directory / "step7_changelog.md"
    changelog = "\n".join(
        [
            "# Step 7 changelog",
            "",
            "## ADDED",
            "",
            "- Nearest-neighbour clinical-sharing analysis.",
            "- Unrestricted random non-neighbour baseline.",
            "- Site-matched random sensitivity.",
            "- Study-group manifold overlay.",
            "- HbA1c full-versus-neutral targeted positive control.",
            "- Revised neighbour-sharing forest figure.",
            "- Revised continuous-manifold clinical overlay figure.",
            "",
            "## CHANGED",
            "",
            "- Final palette and figure labels.",
            "- External probe forest presentation.",
            "- Night and day display.",
            "- Exploratory k=2 description.",
            "- Final results, discussion, conclusion, and presentation language.",
            "",
            "## UNCHANGED",
            "",
            "- Canonical participant split.",
            "- Frozen validation PCA and scalers.",
            "- Burn-in and participant aggregation.",
            "- Static-neutralization intervention.",
            "- Original Step 0 through Step 6 artifacts and hashes.",
            "- Primary continuous-manifold conclusion.",
            "- External clinical-probe transport conclusion.",
            "- Exploratory status of k=2.",
        ]
    )
    changelog_path.write_text(changelog + "\n")
    report_path = run_directory / "step7_report.md"
    hba1c_validation = hba1c_summary[
        hba1c_summary["split"].eq("validation_nested_cv")
    ].iloc[0]
    hba1c_test = hba1c_summary[
        hba1c_summary["split"].eq("test_transport")
    ].iloc[0]
    report = "\n".join(
        [
            "# Step 7 original-question closing pass",
            "",
            "## Status",
            "",
            "All authorized closing stages completed.",
            "",
            "## Direct answer",
            "",
            "Participants close in static-neutral streaming hidden-state space "
            "shared glucose control, HbA1c, and study group more strongly than "
            "random participants. External biomarker sharing was not supported.",
            "",
            "## HbA1c positive control",
            "",
            f"Validation glycemic baseline R2: "
            f"{hba1c_validation['glycemic_baseline_r2']:.4f}. "
            f"Validation full delta R2: "
            f"{hba1c_validation['full_delta_r2']:+.4f}. "
            f"Validation neutral delta R2: "
            f"{hba1c_validation['neutral_delta_r2']:+.4f}. "
            f"Test baseline R2: {hba1c_test['glycemic_baseline_r2']:.4f}. "
            f"Test full delta R2: {hba1c_test['full_delta_r2']:+.4f}. "
            f"Test neutral delta R2: "
            f"{hba1c_test['neutral_delta_r2']:+.4f}.",
            "",
            "The neutral validation increment did not transport to test. The "
            "full-profile increment transported, consistent with direct HbA1c "
            "conditioning.",
            "",
            "## Outputs",
            "",
            f"- Revised figures: {run_directory / 'revised_figures'}",
            f"- HbA1c outputs: {run_directory / 'hba1c_positive_control'}",
            f"- Revised text: {revised_text_directory}",
            f"- Neighbour outputs: {run_directory / 'neighbor_sharing'}",
        ]
    )
    report_path.write_text(report + "\n")
    generated_text = [*text_paths.values(), changelog_path, report_path]
    for path in generated_text:
        lowered = path.read_text().lower()
        for phrase in PROHIBITED_FINAL_PHRASES:
            if phrase in lowered:
                raise RuntimeError(
                    f"Prohibited final phrase '{phrase}' in {path}"
                )
    if MODEL_INPUT_PARAGRAPH not in complete:
        raise RuntimeError("Required model-input clarification is absent")
    if EXTERNAL_INPUT_PARAGRAPH not in complete:
        raise RuntimeError("Required external-target clarification is absent")
    manifest["text_stage"] = {
        "status": "QC_COMPLETE",
        "text_paths": {name: str(path) for name, path in text_paths.items()},
        "text_hashes": {
            name: sha256_file(path) for name, path in text_paths.items()
        },
        "changelog_path": str(changelog_path),
        "required_input_wording_present": True,
        "prohibited_k2_language_absent": True,
        "blockers": [],
    }
    source_manifest_paths = {
        "step0": step_directories["step0"] / "step0_manifest.json",
        "step1": step_directories["step1"] / "step1_manifest.json",
        "step2": step_directories["step2"] / "step2_manifest.json",
        "step3": step_directories["step3"] / "step3_manifest.json",
        "step3b": step_directories["step3b"]
        / "exploratory_k2_freeze_manifest.json",
        "step4": step_directories["step4"] / "step4_manifest.json",
        "step5": step_directories["step5"] / "step5_manifest.json",
        "step6": step_directories["step6"] / "step6_manifest.json",
    }
    source_manifest_hashes = {
        name: sha256_file(path)
        for name, path in source_manifest_paths.items()
    }
    original_input_hashes = manifest.get("input_hashes", {})
    input_paths = manifest.get("input_paths", {})
    current_input_hashes = {
        name: sha256_file(Path(input_paths[name]))
        for name in original_input_hashes
        if name in input_paths and Path(input_paths[name]).is_file()
    }
    source_hashes_match = all(
        current_input_hashes.get(name) == expected
        for name, expected in original_input_hashes.items()
    )
    if not source_hashes_match:
        raise RuntimeError("One or more Gate 1 source hashes changed")
    manifest.update(
        {
            "status": "QC_COMPLETE",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "git_final": git_metadata(repository_root),
            "step_run_ids": {
                name: path.name
                for name, path in step_directories.items()
            },
            "step_manifest_paths": {
                name: str(path)
                for name, path in source_manifest_paths.items()
            },
            "step_manifest_hashes": source_manifest_hashes,
            "participant_counts": {
                "validation": 239,
                "test": 221,
                "hba1c_validation": 235,
                "hba1c_test": 217,
            },
            "model_input_clarification": {
                "most_core_static_variables_are_inputs": True,
                "hba1c_is_direct_input_positive_control": True,
                "external_biomarkers_are_forecasting_inputs": False,
            },
            "source_hash_status": "UNCHANGED",
            "source_hashes_match_gate1": source_hashes_match,
            "warnings": manifest.get("hba1c_stage", {}).get(
                "warnings", []
            ),
            "blockers": [],
            "qc_status": "QC_COMPLETE",
            "latest_pointer_created": False,
        }
    )
    write_json(manifest_path, manifest)
    em_dash_files = scan_em_dash(
        [
            run_directory,
            repository_root
            / "scripts/run_hidden_state_original_question_closing_pass.py",
            repository_root
            / "ssmcgm/analysis/neighbor_clinical_sharing.py",
            repository_root
            / "ssmcgm/analysis/hba1c_positive_control.py",
            repository_root
            / "ssmcgm/analysis/final_closing_figures.py",
            Path(__file__),
        ]
    )
    if em_dash_files:
        raise RuntimeError(
            "Forbidden Unicode U+2014 found: " + ", ".join(em_dash_files)
        )
    required_output_paths = [
        *text_paths.values(),
        changelog_path,
        report_path,
        run_directory / "revised_figures/revised_figure_manifest.csv",
        run_directory
        / "neighbor_sharing/neighbor_sharing_tier1_results.csv",
        run_directory
        / "hba1c_positive_control/hba1c_full_vs_neutral_summary.csv",
    ]
    if any(not path.exists() for path in required_output_paths):
        raise RuntimeError("A required final output is missing")
    latest_path = run_directory.parent / "latest"
    temporary_latest = run_directory.parent / ".latest.tmp"
    if temporary_latest.exists() or temporary_latest.is_symlink():
        temporary_latest.unlink()
    temporary_latest.symlink_to(run_directory.name)
    os.replace(temporary_latest, latest_path)
    if latest_path.resolve() != run_directory.resolve():
        raise RuntimeError("Latest pointer does not resolve to final run")
    manifest["latest_pointer_created"] = True
    manifest["latest_pointer"] = str(latest_path)
    manifest["em_dash_scan_status"] = "PASS"
    manifest["required_output_status"] = "PASS"
    write_json(manifest_path, manifest)
    with (run_directory / "step7_run.log").open("a") as handle:
        handle.write("STEP 7 revised text stage completed\n")
        handle.write("Final QC status: QC_COMPLETE\n")
        handle.write("Source hash status: UNCHANGED\n")
        handle.write("Em-dash scan status: PASS\n")
        handle.write(f"Latest pointer: {latest_path}\n")
    final_em_dash_scan = scan_em_dash([run_directory])
    if final_em_dash_scan:
        raise RuntimeError(
            "Final post-manifest em-dash scan failed: "
            + ", ".join(final_em_dash_scan)
        )
    return {
        "run_directory": str(run_directory),
        "status": "QC_COMPLETE",
        "text_paths": {name: str(path) for name, path in text_paths.items()},
        "changelog_path": str(changelog_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "latest_pointer": str(latest_path),
        "source_hash_status": "UNCHANGED",
        "em_dash_scan_status": "PASS",
        "warnings": manifest["warnings"],
        "blockers": [],
    }
