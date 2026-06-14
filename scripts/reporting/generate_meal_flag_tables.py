#!/usr/bin/env python3
"""Generate LaTeX tables for the meal-flag report section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG = {
    "project_root": "/home/myriamcharfeddine/CGM/SSM-CGM",
    "meal_transfer_dir": "outputs/no_log_scenarios/meal_transfer",
    "online_causal_dir": "outputs/no_log_scenarios/meal_transfer/online_causal",
    "scientific_summary_dir": "outputs/no_log_scenarios/meal_transfer/scientific_summary",
    "table_dir": "report/tables/meal_flags",
}


def load_config(path: Path) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        import yaml

        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)
    except Exception:
        pass
    return cfg


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def fmt(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return latex_escape(value)


def metric_dict(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["value"]))


def write_table(path: Path, headers: list[str], rows: list[list[str]], align: str | None = None, resize: bool = True) -> None:
    align = align or ("l" + "r" * (len(headers) - 1))
    lines: list[str] = []
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    path.write_text("\n".join(lines) + "\n")


def table_pipeline_summary(meal_dir: Path, online_dir: Path, table_dir: Path, summary_dir: Path) -> str:
    m = metric_dict(meal_dir / "meal_transfer_metrics.csv")
    with open(online_dir / "online_generation_manifest.json") as f:
        online_manifest = json.load(f)
    rows_raw = [
        ("Participants", int(float(m["participants"]))),
        ("Rows", int(float(m["rows"]))),
        ("Teacher probability finite fraction", float(m["teacher_prob_finite_frac"])),
        ("Corrected teacher flag rate", float(m["teacher_flag_rate"])),
        ("Surviving old binary artifact rate", float(m["legacy_predmeal_flag_rate"])),
        ("Pseudo-label positives", int(float(m["pseudo_positive"]))),
        ("Pseudo-label negatives", int(float(m["pseudo_negative"]))),
        ("Pseudo-label uncertain", int(float(m["pseudo_uncertain"]))),
        ("Causal student validation AP", float(m["student_val_average_precision"])),
        ("Causal student validation AUC", float(m["student_val_auc"])),
        ("Retrospective decoded events", int(float(m["n_events"]))),
        ("Retrospective decoded flag rate", float(m["decoded_flag_rate"])),
        ("Online prefix invariance max absolute difference", online_manifest["prefix_invariance"]["max_abs_diff"]),
        ("Online response-size validation macro-F1", online_manifest["response_size_model"]["val_macro_f1"]),
        ("Independent evaluation events", int(online_manifest["eval_mask"]["n_eval_events"])),
    ]
    source = pd.DataFrame(rows_raw, columns=["quantity", "value"])
    source.to_csv(summary_dir / "meal_pipeline_summary.csv", index=False)
    rows = []
    for key, value in rows_raw:
        if isinstance(value, int):
            out = f"{value:,}"
        else:
            out = fmt(value, 3)
        rows.append([latex_escape(key), out])
    path = table_dir / "meal_pipeline_summary.tex"
    write_table(path, ["Quantity", "Value"], rows, align="lr", resize=False)
    return str(path)


def table_main_results(meal_dir: Path, table_dir: Path, summary_dir: Path) -> str:
    df = pd.read_csv(meal_dir / "meal_ablation_60min_metrics.csv")
    order = [
        "BASELINE_q50_uncorrected",
        "A_none",
        "B_old_predmeal_flag",
        "C_teacher_prob",
        "D_student_prob",
        "E_prob_phase_time",
        "F_full_state",
    ]
    labels = {
        "BASELINE_q50_uncorrected": "q50",
        "A_none": "A: none",
        "B_old_predmeal_flag": "B: old flag",
        "C_teacher_prob": "C: teacher prob.",
        "D_student_prob": "D: student prob.",
        "E_prob_phase_time": "E: prob.+phase",
        "F_full_state": "F: full state",
    }
    df["setup"] = pd.Categorical(df["setup"], categories=order, ordered=True)
    df = df.sort_values("setup")
    cols = ["setup", "MAE_30min", "MAE_60min", "MAE_meal_window", "peak_error_1h", "hyper_AUROC", "hyper_AUPRC", "hyper_brier"]
    source = df[cols].copy()
    source.to_csv(summary_dir / "meal_main_results.csv", index=False)
    rows = []
    for _, row in source.iterrows():
        rows.append(
            [
                latex_escape(labels[str(row["setup"])]),
                fmt(row["MAE_30min"]),
                fmt(row["MAE_60min"]),
                fmt(row["MAE_meal_window"]),
                fmt(row["peak_error_1h"]),
                fmt(row["hyper_AUROC"]),
                fmt(row["hyper_AUPRC"]),
                fmt(row["hyper_brier"]),
            ]
        )
    path = table_dir / "meal_main_results.tex"
    write_table(
        path,
        ["Setup", "MAE30", "MAE60", "Meal MAE", "Peak err.", "AUROC", "AUPRC", "Brier"],
        rows,
        align="lrrrrrrr",
    )
    return str(path)


def table_causal_results(online_dir: Path, table_dir: Path, summary_dir: Path) -> str:
    metrics = pd.read_csv(online_dir / "online_meal_ablation_metrics.csv")
    boot = pd.read_csv(online_dir / "online_meal_ablation_bootstrap.csv")
    labels = {
        "BASELINE_q50_uncorrected": "q50",
        "D_student_prob": "D: student prob.",
        "G_online_state": "G: online state",
        "H_online_full_state": "H: online full",
    }
    source = metrics[[
        "setup",
        "MAE_30min",
        "MAE_60min",
        "MAE_eval_meal_window",
        "peak_error_1h",
        "time_to_peak_error_min",
        "hyper_AUROC",
        "hyper_AUPRC",
        "hyper_brier",
    ]].copy()
    source.to_csv(summary_dir / "meal_causal_results.csv", index=False)
    rows = []
    for _, row in source.iterrows():
        rows.append(
            [
                latex_escape(labels[str(row["setup"])]),
                fmt(row["MAE_30min"]),
                fmt(row["MAE_60min"]),
                fmt(row["MAE_eval_meal_window"]),
                fmt(row["peak_error_1h"]),
                fmt(row["time_to_peak_error_min"]),
                fmt(row["hyper_AUROC"]),
                fmt(row["hyper_AUPRC"]),
                fmt(row["hyper_brier"]),
            ]
        )
    path = table_dir / "meal_causal_results.tex"
    write_table(
        path,
        ["Setup", "MAE30", "MAE60", "Eval meal MAE", "Peak err.", "TTP err.", "AUROC", "AUPRC", "Brier"],
        rows,
        align="lrrrrrrrr",
    )
    boot_keep = boot[
        boot["comparison"].isin(
            [
                "BASELINE_q50_uncorrected - H_online_full_state",
                "D_student_prob - H_online_full_state",
                "G_online_state - H_online_full_state",
            ]
        )
        & boot["metric"].isin(["MAE_60min", "MAE_eval_meal_window", "peak_error_1h", "hyper_brier"])
    ].copy()
    boot_keep.to_csv(summary_dir / "meal_causal_bootstrap_summary.csv", index=False)
    return str(path)


def table_negative_controls(meal_dir: Path, online_dir: Path, table_dir: Path, summary_dir: Path) -> str:
    retro = pd.read_csv(meal_dir / "meal_ablation_60min_controls.csv")
    online = pd.read_csv(online_dir / "online_meal_negative_controls.csv")
    retro = retro[retro["setup"].isin(["C_teacher_prob", "F_full_state"])].copy()
    retro["family"] = "retrospective"
    online = online[online["setup"].isin(["H_online_full_state"])].copy()
    online["family"] = "online"
    source = pd.concat([retro, online], ignore_index=True, sort=False)
    source.to_csv(summary_dir / "meal_negative_controls.csv", index=False)
    rows = []
    for _, row in source.iterrows():
        setup = str(row["setup"]).replace("C_teacher_prob", "C teacher").replace("F_full_state", "F full state").replace("H_online_full_state", "H online full")
        if row["family"] == "retrospective":
            gain = fmt(row["MAE_60min_gain_vs_A"])
            meal_col = "MAE_meal_window"
        else:
            gain = "--"
            meal_col = "MAE_eval_meal_window"
        rows.append(
            [
                latex_escape(row["family"]),
                latex_escape(setup),
                latex_escape(row["control"]),
                fmt(row["MAE_60min"]),
                gain,
                fmt(row[meal_col]),
                fmt(row["peak_error_1h"]),
                fmt(row["hyper_AUROC"]),
            ]
        )
    path = table_dir / "meal_negative_controls.tex"
    write_table(
        path,
        ["Family", "Setup", "Control", "MAE60", "Gain vs A", "Meal MAE", "Peak err.", "AUROC"],
        rows,
        align="lllrrrrr",
    )
    return str(path)


def table_subgroups(meal_dir: Path, online_dir: Path, table_dir: Path, summary_dir: Path) -> str:
    online = pd.read_csv(online_dir / "online_meal_subgroups.csv")
    dist = pd.read_csv(meal_dir / "diagnostics" / "distributions_by_med_insulin.csv")
    online = online[
        (online["setup"].isin(["BASELINE_q50_uncorrected", "H_online_full_state"]))
        & (online["subgroup"].isin(["med_insulin", "participants_study_group"]))
    ].copy()
    rate = dist[dist["metric"] == "predmeal_flag_clean"][["group", "mean"]].rename(columns={"group": "rate_group", "mean": "decoded_rate"})
    online.to_csv(summary_dir / "meal_subgroup_results.csv", index=False)
    rate.to_csv(summary_dir / "meal_insulin_decoded_rates.csv", index=False)
    rows = []
    label_map = {
        "BASELINE_q50_uncorrected": "q50",
        "H_online_full_state": "H online full",
    }
    for _, row in online.iterrows():
        rows.append(
            [
                latex_escape(row["subgroup"]),
                latex_escape(row["value"]),
                latex_escape(label_map[str(row["setup"])]),
                f"{int(row['n_participants']):,}",
                fmt(row["MAE_60min"]),
                fmt(row["MAE_eval_meal_window"]),
                fmt(row["peak_error_1h"]),
                fmt(row["hyper_brier"]),
            ]
        )
    path = table_dir / "meal_subgroup_results.tex"
    write_table(
        path,
        ["Subgroup", "Level", "Setup", "$n$", "MAE60", "Meal MAE", "Peak err.", "Brier"],
        rows,
        align="lllrrrrr",
    )
    return str(path)


def table_feature_provenance(online_dir: Path, table_dir: Path, summary_dir: Path) -> str:
    df = pd.read_csv(online_dir / "online_feature_provenance.csv")
    keep = [
        "student_meal_probability",
        "online_meal_probability",
        "online_phase_code",
        "online_time_since_onset",
        "online_confidence",
        "online_support_score",
        "expected_response_size_score",
        "predicted_response_size",
    ]
    sub = df[df["feature_name"].isin(keep)].copy()
    if len(sub) < len(keep):
        sub = df.head(12).copy()
    sub.to_csv(summary_dir / "meal_feature_provenance.csv", index=False)
    rows = []
    for _, row in sub.iterrows():
        rows.append(
            [
                latex_escape(row["feature_name"]),
                latex_escape(row["maximum_lookback"]),
                latex_escape(row["uses_future_glucose"]),
                latex_escape(row["uses_future_wearables"]),
                latex_escape(row["uses_teacher_output_at_inference"]),
                latex_escape(row["available_online"]),
            ]
        )
    path = table_dir / "meal_feature_provenance.tex"
    write_table(
        path,
        ["Feature", "Lookback", "Future glucose", "Future wearables", "Teacher at inference", "Online"],
        rows,
        align="llllll",
    )
    return str(path)


def main() -> None:
    raise SystemExit(
        "DEPRECATED: generate_meal_flag_tables.py reads the stale legacy ablation CSVs and "
        "would overwrite the new post-prandial tables. Use "
        "scripts/reporting/generate_meal_feature_report.py (and build_meal_flag_report_section.py).")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reporting/meal_flag_report.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    root = Path(cfg["project_root"])
    meal_dir = resolve(root, cfg["meal_transfer_dir"])
    online_dir = resolve(root, cfg["online_causal_dir"])
    table_dir = resolve(root, cfg["table_dir"])
    summary_dir = resolve(root, cfg["scientific_summary_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        table_pipeline_summary(meal_dir, online_dir, table_dir, summary_dir),
        table_main_results(meal_dir, table_dir, summary_dir),
        table_causal_results(online_dir, table_dir, summary_dir),
        table_negative_controls(meal_dir, online_dir, table_dir, summary_dir),
        table_subgroups(meal_dir, online_dir, table_dir, summary_dir),
        table_feature_provenance(online_dir, table_dir, summary_dir),
    ]
    manifest = {
        "script": "scripts/reporting/generate_meal_flag_tables.py",
        "config": str(cfg_path),
        "generated_files": generated,
        "summary_csvs": sorted(str(p) for p in summary_dir.glob("meal_*.csv")),
    }
    with open(summary_dir / "meal_flag_table_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(summary_dir / "meal_flag_table_generation.log", "w") as f:
        f.write("Generated meal-flag report tables.\n")
        for item in generated:
            f.write(f"- {item}\n")


if __name__ == "__main__":
    main()
