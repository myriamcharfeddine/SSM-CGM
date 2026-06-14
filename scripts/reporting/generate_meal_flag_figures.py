#!/usr/bin/env python3
"""Generate publication figures for the meal-flag report section."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


DEFAULT_CONFIG = {
    "project_root": "/home/myriamcharfeddine/CGM/SSM-CGM",
    "meal_transfer_dir": "outputs/no_log_scenarios/meal_transfer",
    "online_causal_dir": "outputs/no_log_scenarios/meal_transfer/online_causal",
    "scientific_summary_dir": "outputs/no_log_scenarios/meal_transfer/scientific_summary",
    "figure_dir": "report/figures/meal_flags",
    "figure_dpi": 300,
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


def metric_dict(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["value"]))


def as_float(metrics: dict[str, str], key: str) -> float:
    return float(metrics[key])


def save_source(df: pd.DataFrame, summary_dir: Path, name: str) -> None:
    df.to_csv(summary_dir / f"{name}_source_data.csv", index=False)


def save_figure(fig: plt.Figure, figure_dir: Path, name: str, dpi: int) -> list[str]:
    pdf = figure_dir / f"{name}.pdf"
    png = figure_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]


def setup_axes(ax: plt.Axes, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_pipeline(figure_dir: Path, summary_dir: Path, dpi: int) -> list[str]:
    steps = [
        ("Source audit", "5-min CGM\n72 steps = 6 h"),
        ("Teacher", "CGMacros checkpoint\ncontinuous probability"),
        ("Weak labels", "train quantiles\ninsulin-aware uncertainty"),
        ("Causal student", "past/current features\nparticipant split"),
        ("Meal states", "retrospective decoder\nonline forward filter"),
        ("Forecast test", "residual correction\nnegative controls\nsubgroups"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_axis_off()
    positions = [
        (0.055, 0.58),
        (0.370, 0.58),
        (0.685, 0.58),
        (0.055, 0.25),
        (0.370, 0.25),
        (0.685, 0.25),
    ]
    box_w = 0.245
    box_h = 0.22
    for idx, ((title, body), (x, y)) in enumerate(zip(steps, positions)):
        box = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.025,rounding_size=0.03",
            linewidth=1.1,
            edgecolor="#2f5d7c",
            facecolor="#eef5f9",
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + box_h * 0.68, title, ha="center", va="center", fontsize=10.0, fontweight="bold")
        ax.text(x + box_w / 2, y + box_h * 0.32, body, ha="center", va="center", fontsize=8.2, linespacing=1.25)

    arrow_pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    for start, end in arrow_pairs:
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        if start == 2 and end == 3:
            start_xy = (x0 + box_w / 2, y0 - 0.015)
            end_xy = (x1 + box_w / 2, y1 + box_h + 0.015)
            connectionstyle = "arc3,rad=-0.22"
        else:
            start_xy = (x0 + box_w + 0.022, y0 + box_h / 2)
            end_xy = (x1 - 0.022, y1 + box_h / 2)
            connectionstyle = "arc3,rad=0"
        arrow = FancyArrowPatch(
            start_xy,
            end_xy,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#555555",
            connectionstyle=connectionstyle,
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.09,
        "Teacher outputs are retrospective weak evidence; Track A online outputs are the deployable causal candidates.",
        ha="center",
        va="center",
        fontsize=8.3,
    )
    data = pd.DataFrame(steps, columns=["stage", "summary"])
    save_source(data, summary_dir, "meal_pipeline")
    return save_figure(fig, figure_dir, "meal_pipeline", dpi)


def fig_teacher_reconstruction(meal_dir: Path, figure_dir: Path, summary_dir: Path, dpi: int) -> list[str]:
    metrics = metric_dict(meal_dir / "meal_transfer_metrics.csv")
    rows = pd.DataFrame(
        [
            {"artifact": "Corrected probability finite", "rate": as_float(metrics, "teacher_prob_finite_frac")},
            {"artifact": "Corrected teacher flag", "rate": as_float(metrics, "teacher_flag_rate")},
            {"artifact": "Legacy ratio vote", "rate": as_float(metrics, "teacher_baseline_flag_rate")},
            {"artifact": "Surviving binary artifact", "rate": as_float(metrics, "legacy_predmeal_flag_rate")},
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    colors = ["#5c8eb0", "#2f5d7c", "#9bb6c8", "#b36b43"]
    bars = ax.bar(rows["artifact"], rows["rate"], color=colors)
    setup_axes(ax, "Teacher reconstruction preserves the lost probability signal", "fraction of rows")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=18)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.3f}", ha="center", fontsize=8)
    loss = int(float(metrics["artifact_coverage_loss_recon1_artifact0"]))
    ax.text(
        0.02,
        0.90,
        f"Reconstruction=1/artifact=0 rows: {loss:,}",
        transform=ax.transAxes,
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    save_source(rows, summary_dir, "teacher_reconstruction")
    return save_figure(fig, figure_dir, "teacher_reconstruction", dpi)


def fig_retrospective_state_validity(meal_dir: Path, figure_dir: Path, summary_dir: Path, dpi: int) -> list[str]:
    df = pd.read_csv(meal_dir / "meal_response_by_size.csv")
    order = ["small", "medium", "large"]
    df["response_size"] = pd.Categorical(df["response_size"], categories=order, ordered=True)
    df = df.sort_values("response_size")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    axes[0].bar(df["response_size"].astype(str), df["mean_peak_rise"], color="#2f5d7c")
    setup_axes(axes[0], "Peak rise", "mg/dL")
    axes[1].bar(df["response_size"].astype(str), df["mean_duration_min"], color="#5c8eb0")
    setup_axes(axes[1], "Event duration", "minutes")
    width = 0.36
    x = range(len(df))
    axes[2].bar([v - width / 2 for v in x], df["mean_dg_60"], width=width, label="60 min", color="#7aa36f")
    axes[2].bar([v + width / 2 for v in x], df["mean_dg_120"], width=width, label="120 min", color="#b36b43")
    axes[2].set_xticks(list(x), df["response_size"].astype(str))
    axes[2].legend(frameon=False, fontsize=8)
    setup_axes(axes[2], "Glucose change at onset", "mg/dL")
    fig.suptitle("Retrospective response-size proxy is ordered by realized glycemic excursion", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_source(df, summary_dir, "retrospective_state_validity")
    return save_figure(fig, figure_dir, "retrospective_state_validity", dpi)


def fig_forecast_ablation(meal_dir: Path, online_dir: Path, figure_dir: Path, summary_dir: Path, dpi: int) -> list[str]:
    retro = pd.read_csv(meal_dir / "meal_ablation_60min_metrics.csv")
    online = pd.read_csv(online_dir / "online_meal_ablation_metrics.csv")
    keep_retro = ["BASELINE_q50_uncorrected", "A_none", "B_old_predmeal_flag", "C_teacher_prob", "D_student_prob", "E_prob_phase_time", "F_full_state"]
    retro = retro[retro["setup"].isin(keep_retro)].copy()
    retro["family"] = "retrospective"
    online = online.copy()
    online["family"] = "online"
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    labels = {
        "BASELINE_q50_uncorrected": "q50",
        "A_none": "A",
        "B_old_predmeal_flag": "B",
        "C_teacher_prob": "C",
        "D_student_prob": "D",
        "E_prob_phase_time": "E",
        "F_full_state": "F",
        "G_online_state": "G",
        "H_online_full_state": "H",
    }
    colors = ["#707070", "#b8b8b8", "#8fb3c9", "#2f5d7c", "#c78f5c", "#bfcf8f", "#7aa36f"]
    axes[0].bar([labels[x] for x in retro["setup"]], retro["MAE_60min"], color=colors)
    setup_axes(axes[0], "Retrospective A-F", "60-min MAE")
    axes[1].bar([labels[x] for x in retro["setup"]], retro["peak_error_1h"], color=colors)
    setup_axes(axes[1], "Retrospective A-F", "1-h peak error")
    online_colors = ["#707070", "#c78f5c", "#7aa36f", "#2f5d7c"]
    axes[2].bar([labels[x] for x in online["setup"]], online["MAE_60min"], color=online_colors)
    setup_axes(axes[2], "Strict online Track A", "60-min MAE")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Forecast residual ablations separate retrospective upper bounds from online features", fontsize=11, fontweight="bold")
    fig.tight_layout()
    source = pd.concat([retro, online], ignore_index=True, sort=False)
    save_source(source, summary_dir, "forecast_ablation")
    return save_figure(fig, figure_dir, "forecast_ablation", dpi)


def fig_participant_effects(online_dir: Path, figure_dir: Path, summary_dir: Path, dpi: int) -> list[str]:
    df = pd.read_csv(online_dir / "online_meal_ablation_bootstrap.csv")
    metrics = ["MAE_60min", "MAE_eval_meal_window", "peak_error_1h", "hyper_brier"]
    comps = [
        "BASELINE_q50_uncorrected - G_online_state",
        "BASELINE_q50_uncorrected - H_online_full_state",
        "D_student_prob - H_online_full_state",
        "G_online_state - H_online_full_state",
    ]
    sub = df[df["metric"].isin(metrics) & df["comparison"].isin(comps)].copy()
    sub["label"] = sub["comparison"].str.replace("BASELINE_q50_uncorrected", "q50", regex=False)
    sub["label"] = sub["label"].str.replace("_", " ", regex=False)
    sub["metric_label"] = sub["metric"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    y = list(range(len(sub)))
    ax.errorbar(
        sub["participant_mean_diff_positive_improves"],
        y,
        xerr=[
            sub["participant_mean_diff_positive_improves"] - sub["ci95_lo"],
            sub["ci95_hi"] - sub["participant_mean_diff_positive_improves"],
        ],
        fmt="o",
        color="#2f5d7c",
        ecolor="#8fb3c9",
        capsize=2,
        markersize=4,
    )
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(y, sub["label"] + " / " + sub["metric_label"], fontsize=7.2)
    ax.set_xlabel("participant-bootstrap difference; positive favors right-hand setup")
    setup_axes(ax, "Paired participant effects for online causal representations")
    fig.tight_layout()
    save_source(sub, summary_dir, "participant_level_effects")
    return save_figure(fig, figure_dir, "participant_level_effects", dpi)


def fig_negative_controls(meal_dir: Path, online_dir: Path, figure_dir: Path, summary_dir: Path, dpi: int) -> list[str]:
    retro = pd.read_csv(meal_dir / "meal_ablation_60min_controls.csv")
    online = pd.read_csv(online_dir / "online_meal_negative_controls.csv")
    retro = retro[retro["setup"].isin(["F_full_state", "C_teacher_prob", "D_student_prob"])].copy()
    online = online[online["setup"].isin(["H_online_full_state", "G_online_state"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    pivot = retro.pivot(index="control", columns="setup", values="MAE_60min_gain_vs_A").loc[["real", "shuffle", "time_shift", "block_shuffle"]]
    pivot.plot(kind="bar", ax=axes[0], color=["#2f5d7c", "#c78f5c", "#7aa36f"], width=0.78)
    setup_axes(axes[0], "Retrospective controls", "MAE60 gain vs A")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].tick_params(axis="x", labelrotation=20, labelsize=8)
    pivot2 = online.pivot(index="control", columns="setup", values="MAE_60min").loc[["real", "shuffle", "time_shift", "block_shuffle"]]
    pivot2.plot(kind="bar", ax=axes[1], color=["#7aa36f", "#2f5d7c"], width=0.72)
    setup_axes(axes[1], "Online controls", "MAE60")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.suptitle("Negative controls test whether meal-state timing carries specific forecast signal", fontsize=11, fontweight="bold")
    fig.tight_layout()
    source = pd.concat(
        [
            retro.assign(family="retrospective"),
            online.assign(family="online", MAE_60min_gain_vs_A=pd.NA),
        ],
        ignore_index=True,
        sort=False,
    )
    save_source(source, summary_dir, "negative_controls")
    return save_figure(fig, figure_dir, "negative_controls", dpi)


def fig_state_examples(online_dir: Path, figure_dir: Path, summary_dir: Path, dpi: int) -> list[str]:
    src = online_dir / "online_state_examples.png"
    if not src.exists():
        raise FileNotFoundError(src)
    image = plt.imread(src)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title("Examples of strictly online decoded meal states", fontsize=11, fontweight="bold")
    save_source(pd.DataFrame([{"source_png": str(src)}]), summary_dir, "state_examples")
    return save_figure(fig, figure_dir, "state_examples", dpi)


def fig_insulin_sensitivity(meal_dir: Path, online_dir: Path, figure_dir: Path, summary_dir: Path, dpi: int) -> list[str]:
    dist = pd.read_csv(meal_dir / "diagnostics" / "distributions_by_med_insulin.csv")
    sub = pd.read_csv(online_dir / "online_meal_subgroups.csv")
    dist_keep = dist[dist["metric"].isin(["student_meal_probability", "predmeal_flag_clean", "cgmacros_teacher_flag"])].copy()
    med = sub[(sub["subgroup"] == "med_insulin") & (sub["setup"].isin(["BASELINE_q50_uncorrected", "H_online_full_state"]))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    pivot = dist_keep.pivot(index="group", columns="metric", values="mean")
    pivot.plot(kind="bar", ax=axes[0], color=["#2f5d7c", "#c78f5c", "#7aa36f"], width=0.75)
    setup_axes(axes[0], "Decoded meal-like rate by insulin status", "mean / row rate")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].tick_params(axis="x", labelrotation=0, labelsize=8)
    pivot2 = med.pivot(index="value", columns="setup", values="MAE_60min")
    pivot2.index = [f"med_insulin={v}" for v in pivot2.index]
    pivot2.plot(kind="bar", ax=axes[1], color=["#707070", "#2f5d7c"], width=0.72)
    setup_axes(axes[1], "Forecast error by insulin status", "60-min MAE")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].tick_params(axis="x", labelrotation=0, labelsize=8)
    fig.suptitle("Insulin users show inflated decoded meal-like rates and higher forecast error", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_source(pd.concat([dist_keep.assign(source="distribution"), med.assign(source="subgroup")], ignore_index=True, sort=False), summary_dir, "insulin_sensitivity")
    return save_figure(fig, figure_dir, "insulin_sensitivity", dpi)


def fig_retrospective_causal_gap(meal_dir: Path, online_dir: Path, figure_dir: Path, summary_dir: Path, dpi: int) -> list[str]:
    retro = pd.read_csv(meal_dir / "meal_ablation_60min_metrics.csv")
    online = pd.read_csv(online_dir / "online_meal_ablation_metrics.csv")
    rows = pd.DataFrame(
        [
            {"setup": "q50 baseline", "family": "baseline", "MAE_60min": retro.loc[retro["setup"] == "BASELINE_q50_uncorrected", "MAE_60min"].iloc[0], "peak_error_1h": retro.loc[retro["setup"] == "BASELINE_q50_uncorrected", "peak_error_1h"].iloc[0]},
            {"setup": "C teacher", "family": "retrospective", "MAE_60min": retro.loc[retro["setup"] == "C_teacher_prob", "MAE_60min"].iloc[0], "peak_error_1h": retro.loc[retro["setup"] == "C_teacher_prob", "peak_error_1h"].iloc[0]},
            {"setup": "F full state", "family": "retrospective", "MAE_60min": retro.loc[retro["setup"] == "F_full_state", "MAE_60min"].iloc[0], "peak_error_1h": retro.loc[retro["setup"] == "F_full_state", "peak_error_1h"].iloc[0]},
            {"setup": "D student", "family": "causal", "MAE_60min": online.loc[online["setup"] == "D_student_prob", "MAE_60min"].iloc[0], "peak_error_1h": online.loc[online["setup"] == "D_student_prob", "peak_error_1h"].iloc[0]},
            {"setup": "G online state", "family": "causal", "MAE_60min": online.loc[online["setup"] == "G_online_state", "MAE_60min"].iloc[0], "peak_error_1h": online.loc[online["setup"] == "G_online_state", "peak_error_1h"].iloc[0]},
            {"setup": "H online full", "family": "causal", "MAE_60min": online.loc[online["setup"] == "H_online_full_state", "MAE_60min"].iloc[0], "peak_error_1h": online.loc[online["setup"] == "H_online_full_state", "peak_error_1h"].iloc[0]},
        ]
    )
    color_map = {"baseline": "#707070", "retrospective": "#2f5d7c", "causal": "#7aa36f"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axes[0].bar(rows["setup"], rows["MAE_60min"], color=[color_map[x] for x in rows["family"]])
    setup_axes(axes[0], "60-min MAE", "mg/dL")
    axes[1].bar(rows["setup"], rows["peak_error_1h"], color=[color_map[x] for x in rows["family"]])
    setup_axes(axes[1], "1-h peak error", "mg/dL")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.suptitle("Retrospective signal is strong, but deployable online signal is modest", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_source(rows, summary_dir, "retrospective_causal_gap")
    return save_figure(fig, figure_dir, "retrospective_causal_gap", dpi)


def main() -> None:
    raise SystemExit(
        "DEPRECATED: generate_meal_flag_figures.py reads the stale legacy ablation CSVs and "
        "would overwrite the new post-prandial figures. Use "
        "scripts/reporting/generate_meal_feature_report.py (and build_meal_flag_report_section.py).")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reporting/meal_flag_report.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    root = Path(cfg["project_root"])
    meal_dir = resolve(root, cfg["meal_transfer_dir"])
    online_dir = resolve(root, cfg["online_causal_dir"])
    figure_dir = resolve(root, cfg["figure_dir"])
    summary_dir = resolve(root, cfg["scientific_summary_dir"])
    dpi = int(cfg.get("figure_dpi", 300))
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    generated += fig_pipeline(figure_dir, summary_dir, dpi)
    generated += fig_teacher_reconstruction(meal_dir, figure_dir, summary_dir, dpi)
    generated += fig_retrospective_state_validity(meal_dir, figure_dir, summary_dir, dpi)
    generated += fig_forecast_ablation(meal_dir, online_dir, figure_dir, summary_dir, dpi)
    generated += fig_participant_effects(online_dir, figure_dir, summary_dir, dpi)
    generated += fig_negative_controls(meal_dir, online_dir, figure_dir, summary_dir, dpi)
    generated += fig_state_examples(online_dir, figure_dir, summary_dir, dpi)
    generated += fig_insulin_sensitivity(meal_dir, online_dir, figure_dir, summary_dir, dpi)
    generated += fig_retrospective_causal_gap(meal_dir, online_dir, figure_dir, summary_dir, dpi)

    manifest = {
        "script": "scripts/reporting/generate_meal_flag_figures.py",
        "config": str(cfg_path),
        "generated_files": generated,
        "source_data_csvs": sorted(str(p) for p in summary_dir.glob("*_source_data.csv")),
    }
    with open(summary_dir / "meal_flag_figure_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(summary_dir / "meal_flag_figure_generation.log", "w") as f:
        f.write("Generated meal-flag report figures.\n")
        for item in generated:
            f.write(f"- {item}\n")


if __name__ == "__main__":
    main()
